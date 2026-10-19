"""Compare nine baseline features with an extra weight * horsepower feature.

Run from Git Bash: ./.venv/Scripts/python.exe ml/experiments/exp_interaction.py
Reuse the team's fixed split and preprocessing; keep production models untouched.
Both fitted pipelines and their provenance are saved to local MLflow storage.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.data_pipeline import V2_FEATURE_COLUMNS  # noqa: E402
from ml.train import TARGET, dvc_md5, make_fixed_split, prepare_improved  # noqa: E402

INTERACTION = "weight_x_horsepower"
EXPERIMENT = "AutoMPG_Team_Experiments"
SOURCE_FILES = (
    "ml/experiments/exp_interaction.py",
    "ml/train.py",
    "ml/data_pipeline.py",
    "requirements.txt",
    "requirements-dev.txt",
)


def build_model() -> Pipeline:
    """Fit missing-value statistics and scaling only on training records."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )


def augment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the product before imputation, matching the team's experiment."""
    out = df.copy()
    out[INTERACTION] = out["weight"] * out["horsepower"]
    return out


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def verified_data_hash(path: Path) -> str:
    """Reject data whose actual bytes do not match the adjacent DVC pointer."""
    expected = dvc_md5(path)
    actual = hashlib.md5(path.read_bytes()).hexdigest()
    if expected != actual:
        raise ValueError(f"Data version mismatch: DVC={expected}, actual={actual}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=_ROOT / "data/auto_mpg_v1.csv")
    args = parser.parse_args()
    data_path = args.data.resolve()
    data_hash = verified_data_hash(data_path)
    raw = pd.read_csv(data_path)
    c_train, c_test, missing = make_fixed_split(raw, 0.2, 42)
    train_fold, test_fold = prepare_improved(c_train, c_test, missing)

    train_ids = pd.concat([c_train, missing])["record_id"].astype(int).tolist()
    test_ids = c_test["record_id"].astype(int).tolist()
    if (len(train_ids), len(test_ids), len(missing)) != (319, 79, 6):
        raise ValueError("Expected 319 training rows, 79 test rows, and 6 missing-horsepower rows")
    if set(train_ids) & set(test_ids) or len(set(train_ids + test_ids)) != len(raw):
        raise ValueError("Training/test identities overlap or do not cover all raw records")
    split = {"train_record_ids": train_ids, "test_record_ids": test_ids}

    # A Git SHA alone does not identify uncommitted code: also save exact sources.
    code_version = git_output("rev-parse", "HEAD")
    worktree_dirty = bool(git_output("status", "--porcelain"))
    source_hashes = {
        name: hashlib.sha256((_ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES
    }
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "mlflow": mlflow.__version__,
    }

    mlflow.set_tracking_uri(f"sqlite:///{(_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT)
    print(f"data_version={data_hash}; n_train={len(train_ids)} n_test={len(test_ids)}")
    print(f"code_version={code_version}; worktree_dirty={worktree_dirty}")
    results = {}

    for tag in ("baseline_features", "with_interaction"):
        use_interaction = tag == "with_interaction"
        train_df = augment(train_fold) if use_interaction else train_fold
        test_df = augment(test_fold) if use_interaction else test_fold
        features = list(V2_FEATURE_COLUMNS) + ([INTERACTION] if use_interaction else [])
        model = build_model()
        model.fit(train_df[features], train_df[TARGET])
        pred = model.predict(test_df[features])
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(test_df[TARGET], pred))),
            "mae": float(mean_absolute_error(test_df[TARGET], pred)),
            "r2": float(r2_score(test_df[TARGET], pred)),
        }
        median_hp = float(model.named_steps["imputer"].statistics_[features.index("horsepower")])
        if not use_interaction and not np.allclose(
            list(metrics.values()), [3.2691, 2.4890, 0.7906], rtol=0, atol=0.00005
        ):
            raise ValueError(f"Baseline does not reproduce the team reference: {metrics}")

        with mlflow.start_run(run_name=f"exp-{tag}") as run:
            mlflow.log_params(
                {
                    "features": ",".join(features),
                    "n_features": len(features),
                    "model_type": "linear",
                    "fit_intercept": True,
                    "imputer_strategy": "median",
                    "scale": True,
                    "interaction": use_interaction,
                    "interaction_order": "product_before_columnwise_imputation",
                    "test_size": 0.2,
                    "random_state": 42,
                    "n_train_rows": len(train_ids),
                    "n_test_rows": len(test_ids),
                    "n_missing_horsepower": len(missing),
                    "train_fit_median_hp": median_hp,
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    "code_version": code_version,
                    "data_version": data_hash,
                    "data_file": data_path.name,
                    "worktree_dirty": str(worktree_dirty).lower(),
                    "experiment_script_sha256": source_hashes[SOURCE_FILES[0]],
                    "role": "A",
                    "stage": "team_interaction_comparison",
                }
            )
            mlflow.log_dict(split, "evidence/split.json")
            mlflow.log_dict(source_hashes, "evidence/source_sha256.json")
            mlflow.log_dict(versions, "evidence/versions.json")
            mlflow.log_dict(
                {
                    "features": features,
                    "imputation_medians": model.named_steps["imputer"].statistics_.tolist(),
                },
                "evidence/preprocessing.json",
            )
            for name in SOURCE_FILES:
                artifact_dir = (Path("source") / Path(name).parent).as_posix()
                mlflow.log_artifact(str(_ROOT / name), artifact_path=artifact_dir)
            mlflow.log_artifact(str(data_path.with_suffix(".csv.dvc")), artifact_path="evidence")
            with tempfile.TemporaryDirectory() as tmp:
                predictions = pd.DataFrame(
                    {
                        "record_id": test_ids,
                        "actual_mpg": test_df[TARGET].to_numpy(),
                        "predicted_mpg": pred,
                    }
                )
                predictions["residual"] = predictions["actual_mpg"] - predictions["predicted_mpg"]
                prediction_file = Path(tmp) / "test_predictions.csv"
                predictions.to_csv(prediction_file, index=False)
                mlflow.log_artifact(str(prediction_file), artifact_path="evidence")

            model_info = mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=test_df[features].head(3),
                skops_trusted_types=["numpy.dtype"],
            )
            # No register_model or export: the production pipeline still expects 9 features.
            mlflow.log_dict(
                {"run_id": run.info.run_id, "model_uri": model_info.model_uri, **metrics},
                "evidence/result.json",
            )
        results[tag] = metrics
        print(
            f"{tag:>18}: RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}"
            f"  R2={metrics['r2']:.4f}  train_median_hp={median_hp:.1f}"
        )
        print(f"  run_id={run.info.run_id}; model_uri={model_info.model_uri}")

    delta = results["baseline_features"]["rmse"] - results["with_interaction"]["rmse"]
    relative = 100 * delta / results["baseline_features"]["rmse"]
    print(f"Test RMSE reduction: {delta:.4f} mpg ({relative:.2f}%)")


if __name__ == "__main__":
    main()
