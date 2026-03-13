#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 image_features.tsv + split.tsv 生成 1D-CNN / RNN 用的序列数据：
- train_seq.npz / test_seq.npz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def is_R_col(c: str) -> bool:
    if not c.lower().startswith("r_"):
        return False
    try:
        float(c[2:])
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True,
                    help="results/hsi/image_features.tsv")
    ap.add_argument("--split", required=True,
                    help="results/hsi/ml/split_image_seed42.tsv")
    ap.add_argument("--target", default="phase")
    ap.add_argument("--outdir", required=True,
                    help="results/hsi/ml/seq_dataset")
    args = ap.parse_args()

    df = pd.read_csv(args.images, sep="\t")
    split_df = pd.read_csv(args.split, sep="\t")

    # 统一列名小写
    df.columns = [c.strip().lower() for c in df.columns]
    split_df.columns = [c.strip().lower() for c in split_df.columns]

    if "sample_id" not in df.columns or "sample_id" not in split_df.columns:
        raise RuntimeError("image_features / split.tsv 里都需要 sample_id 列。")

    if args.target.lower() not in df.columns:
        raise RuntimeError(f"image_features 缺少 target 列 {args.target}。")

    # 只选 spectral 列
    spec_cols = [c for c in df.columns if is_R_col(c)]
    if not spec_cols:
        raise RuntimeError("未找到任何 R_* 光谱列。")

    # 按波长排序列
    wl = np.array([float(c[2:]) for c in spec_cols], float)
    order = np.argsort(wl)
    wl_sorted = wl[order]
    spec_cols_sorted = [spec_cols[i] for i in order]

    # 合并 split 信息
    df["sample_id"] = df["sample_id"].astype(str)
    split_df["sample_id"] = split_df["sample_id"].astype(str)
    df = df.merge(split_df[["sample_id", "split"]], on="sample_id", how="inner")

    y_raw = df[args.target.lower()].astype(str).to_numpy()
    X_all = df[spec_cols_sorted].to_numpy(dtype=float)

    # 去掉 NaN 行
    ok = np.isfinite(X_all).all(axis=1)
    X_all = X_all[ok]
    y_raw = y_raw[ok]
    split_all = df["split"].to_numpy()[ok]

    le = LabelEncoder()
    y_all = le.fit_transform(y_raw)
    classes = le.classes_

    # 拆 train / test
    train_mask = split_all == "train"
    test_mask = split_all == "test"

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        outdir / "train_seq.npz",
        X=X_train,
        y=y_train,
        classes=classes,
        wavelength=wl_sorted,
        spec_cols=spec_cols_sorted,
    )
    np.savez_compressed(
        outdir / "test_seq.npz",
        X=X_test,
        y=y_test,
        classes=classes,
        wavelength=wl_sorted,
        spec_cols=spec_cols_sorted,
    )
    print(f"[seq] train: {X_train.shape}, test: {X_test.shape}, classes={len(classes)}")


if __name__ == "__main__":
    main()

