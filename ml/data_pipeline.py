"""Auto MPG 数据流水线：..."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAW_COLUMNS = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "car_name",
]

V2_FEATURE_COLUMNS = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin_1",
    "origin_2",
    "origin_3",
]


def read_raw(raw_path: Path) -> pd.DataFrame:
    """读取无表头、空白分隔的原始数据。"""
    return pd.read_csv(raw_path, sep=r"\s+", header=None, names=RAW_COLUMNS)


def coerce_horsepower(df: pd.DataFrame) -> pd.DataFrame:
    """"?" -> NaN，再转成数值列。"""
    out = df.copy()
    out["horsepower"] = pd.to_numeric(out["horsepower"], errors="coerce")
    return out


def impute_horsepower(df: pd.DataFrame, median: float | None = None):
    """用中位数填补 horsepower；传入 median 时不再自行计算（无泄漏评估用）。"""
    out = df.copy()
    if median is None:
        median = float(out["horsepower"].median())
    out["horsepower"] = out["horsepower"].fillna(median)
    return out, float(median)


def encode_origin(df: pd.DataFrame) -> pd.DataFrame:
    """origin(1/2/3) one-hot 成 origin_1/2/3 三列。"""
    out = pd.get_dummies(df, columns=["origin"], prefix="origin", dtype=int)
    for col in ("origin_1", "origin_2", "origin_3"):
        if col not in out.columns:
            out[col] = 0
    return out


def iqr_bounds(df: pd.DataFrame):
    """按 mpg 计算 IQR 阈值 [Q1-1.5*IQR, Q3+1.5*IQR]。"""
    q1, q3 = df["mpg"].quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def apply_iqr(df: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """只保留 mpg 落在 [lower, upper] 内的行。"""
    return df[(df["mpg"] >= lower) & (df["mpg"] <= upper)]


def make_v1(raw_path: Path, out_path: Path) -> None:
    df = read_raw(raw_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[v1] 读取 {raw_path} -> 写出 {out_path}")
    print(f"[v1] 行数={len(df)}，列={list(df.columns)}")
    print(f"[v1] horsepower 含 '?' 的行数: {(df['horsepower'] == '?').sum()}")


def make_v2(v1_path: Path, out_path: Path) -> None:
    df = pd.read_csv(v1_path)
    n_raw = len(df)

    df = coerce_horsepower(df)
    n_missing = int(df["horsepower"].isna().sum())
    df, median_hp = impute_horsepower(df)

    df = encode_origin(df)

    lower, upper = iqr_bounds(df)
    n_outliers = int(((df["mpg"] < lower) | (df["mpg"] > upper)).sum())
    df = apply_iqr(df, lower, upper)

    df = df[["mpg"] + V2_FEATURE_COLUMNS]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[v2] 输入 {v1_path}（{n_raw} 行）-> 输出 {out_path}（{len(df)} 行）")
    print(f"[v2] 中位数填补 horsepower 缺失 {n_missing} 处（中位数={median_hp:.1f}）")
    print(f"[v2] IQR 剔除 mpg 异常行 {n_outliers} 行（区间 [{lower:.2f}, {upper:.2f}]）")
    print("[v2] origin 已 one-hot 编码为 origin_1/origin_2/origin_3")
    print(f"[v2] 最终列: {list(df.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto MPG 数据版本流水线")
    sub = parser.add_subparsers(dest="stage", required=True)

    p_v1 = sub.add_parser("v1", help="原始数据 -> v1 规范 CSV")
    p_v1.add_argument("--raw", type=Path, required=True)
    p_v1.add_argument("--out", type=Path, required=True)

    p_v2 = sub.add_parser("v2", help="v1 -> v2 清洗+编码")
    p_v2.add_argument("--in", dest="in_path", type=Path, required=True)
    p_v2.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.stage == "v1":
        make_v1(args.raw, args.out)
    else:
        make_v2(args.in_path, args.out)


if __name__ == "__main__":
    main()