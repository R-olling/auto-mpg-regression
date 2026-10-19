from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 以 `python ml/train.py` 方式运行时，把仓库根目录加入 sys.path，保证能 import 包 ml
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.data_pipeline import (  # noqa: E402
    V2_FEATURE_COLUMNS,
    coerce_horsepower,
    encode_origin,
)

MODEL_REGISTRY_NAME = "AutoMPG_LinearRegression"  # MLflow 3.x 模型名不能带点号/斜杠/冒号
TARGET = "mpg"
BASELINE_FEATURES = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
]


def git_commit_sha() -> str:
    """取当前代码版本（git commit SHA）；不在 git 仓库时返回 unknown。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def dvc_md5(data_path: Path) -> str:
    """读数据文件旁边 .dvc 指针里的 md5；文件不存在时返回 not-tracked。"""
    dvc_file = data_path.with_suffix(data_path.suffix + ".dvc")
    if not dvc_file.exists():
        return "not-tracked"
    match = re.search(r"md5:\s*([0-9a-f]{32})", dvc_file.read_text(encoding="utf-8"))
    return match.group(1) if match else "not-tracked"


def build_model(model_type: str, alpha: float, scale: bool, impute: bool = False):
    """组装 Pipeline：可选「中位数填补 -> 标准化 -> 回归器」。"""
    steps = []
    if impute:
        steps.append(("imputer", SimpleImputer(strategy="median")))
    if scale:
        steps.append(("scaler", StandardScaler()))
    if model_type == "linear":
        steps.append(("regressor", LinearRegression()))
    else:
        steps.append(("regressor", Ridge(alpha=alpha)))
    return Pipeline(steps)


def log_residual_plot(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """画测试集残差图并作为 artifact 记录。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.6, s=24)
    ax.axhline(0.0, color="grey", linewidth=1)
    ax.set_xlabel("Predicted mpg")
    ax.set_ylabel("Residual (true - predicted)")
    ax.set_title("Test-set residuals")
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "residual_plot.png"
        fig.savefig(png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(png))


def make_fixed_split(raw: pd.DataFrame, test_size: float, random_state: int):
    """按记录身份固定划分：缺马力的 6 条一律进训练折，其余再随机切。"""
    df = raw.copy()
    df["record_id"] = np.arange(len(df))
    df = coerce_horsepower(df)
    missing = df[df["horsepower"].isna()].copy()
    complete = df[df["horsepower"].notna()].copy()
    c_train, c_test = train_test_split(complete, test_size=test_size, random_state=random_state)
    return c_train, c_test, missing


def prepare_baseline(c_train, c_test):
    """baseline：原始 7 特征（origin 仍是整数），不标准化。"""
    X_train = c_train[BASELINE_FEATURES]
    X_test = c_test[BASELINE_FEATURES]
    return X_train, c_train[TARGET], X_test, c_test[TARGET], BASELINE_FEATURES


def prepare_improved(c_train, c_test, missing):
    """improved：训练折 = 完整记录 + 6 条缺失记录，缺失交给 Pipeline 内填补。"""
    train_pool = pd.concat([c_train, missing], ignore_index=True)
    train_pool = encode_origin(train_pool)
    test = encode_origin(c_test)
    cols = ["mpg"] + V2_FEATURE_COLUMNS
    return train_pool[cols], test[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + track an Auto MPG linear model")
    parser.add_argument("--data", type=Path, required=True, help="原始 v1 CSV")
    parser.add_argument("--run-name", required=True, help="MLflow run 显示名")
    parser.add_argument("--stage", default="baseline", choices=["baseline", "improved"])
    parser.add_argument("--model-type", default="linear", choices=["linear", "ridge"])
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge 正则强度")
    parser.add_argument("--scale", action="store_true", help="是否标准化")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiment", default="AutoMPG_Regression")
    parser.add_argument("--export", type=Path, default=None, help="导出模型路径")
    args = parser.parse_args()

    raw = pd.read_csv(args.data)
    c_train, c_test, missing = make_fixed_split(raw, args.test_size, args.random_state)
    n_test = len(c_test)

    if args.stage == "baseline":
        X_train, y_train, X_test, y_test, feature_cols = prepare_baseline(c_train, c_test)
        impute = False
    else:
        train_fold, test_fold = prepare_improved(c_train, c_test, missing)
        feature_cols = V2_FEATURE_COLUMNS
        X_train, y_train = train_fold[feature_cols], train_fold[TARGET]
        X_test, y_test = test_fold[feature_cols], test_fold[TARGET]
        impute = True

    model = build_model(args.model_type, args.alpha, args.scale, impute=impute)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    train_median = None
    if impute:
        hp_index = feature_cols.index("horsepower")
        train_median = float(model.named_steps["imputer"].statistics_[hp_index])

    cv_rmse = None
    if impute:
        kf = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
        scores = -cross_val_score(
            model, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error"
        )
        cv_rmse = float(scores.mean())

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "model_type": args.model_type,
                "alpha": args.alpha if args.model_type == "ridge" else "n/a",
                "scale": args.scale,
                "fit_intercept": True,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "n_train_rows": len(X_train),
                "n_test_rows": n_test,
                "n_features": len(feature_cols),
            }
        )
        if train_median is not None:
            mlflow.log_param("train_fit_median_hp", round(train_median, 4))
        if cv_rmse is not None:
            mlflow.log_metric("cv_rmse", cv_rmse)
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.set_tags(
            {
                "code_version": git_commit_sha(),
                "data_version": dvc_md5(args.data),
                "data_file": str(args.data),
                "stage": args.stage,
                "feature_columns": ",".join(feature_cols),
            }
        )
        model_info = mlflow.sklearn.log_model(
            model, name="model", skops_trusted_types=["numpy.dtype"]
        )
        log_residual_plot(y_test.to_numpy(), y_pred)

        run_id = mlflow.active_run().info.run_id
        mv = mlflow.register_model(model_info.model_uri, MODEL_REGISTRY_NAME)

        if args.export:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, args.export)
            mlflow.log_param("exported_to", str(args.export))

    print("=" * 62)
    print(f"run_name : {args.run_name} (stage={args.stage})")
    print(f"run_id : {run_id}")
    print(f"code_version : {git_commit_sha()}")
    print(f"data_version : {dvc_md5(args.data)} ({args.data})")
    if train_median is not None:
        print(f"train median : {train_median:.1f}（仅在训练折拟合）")
    if cv_rmse is not None:
        print(f"CV RMSE : {cv_rmse:.4f}")
    print(f"features ({len(feature_cols)}) : {list(feature_cols)}")
    print(f"n_train={len(X_train)} n_test={n_test}")
    print(f"RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
    print(f"registered : {MODEL_REGISTRY_NAME} version {mv.version}")
    if args.export:
        print(f"exported to : {args.export}")
    print("=" * 62)


if __name__ == "__main__":
    main()
