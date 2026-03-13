#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单检查 physio patch 任务中是否存在明显的数据泄露风险：
- 针对 patch 特征表（例如 patch_features_with_physiological.tsv）：
  1. 统计每个 physiological_state 的 patch 数量（写出 physio_patch_stats.tsv）
  2. 检查在“行级随机划分”下，同一 source_sample_id 是否容易被拆到 train/test 两端

注意：
- 这里只是诊断脚本，本身不参与训练，也不会修改任何原始文件。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--patch-tsv",
        required=True,
        help="例如 results/hsi/patch_features_with_physiological.tsv",
    )
    ap.add_argument(
        "--target-col",
        default="physiological_state",
        help="标签列名，默认 physiological_state",
    )
    ap.add_argument(
        "--outdir",
        default="results/hsi",
        help="输出目录（统计结果会写到这里），默认 results/hsi",
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="行级 train/test 比例（仅用于诊断），默认 0.2",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42",
    )
    args = ap.parse_args()

    patch_path = Path(args.patch_tsv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[leak_check] 读取 patch 特征表: {patch_path}")
    df = pd.read_csv(patch_path, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]

    target_col = args.target_col.lower()
    if target_col not in df.columns:
        raise RuntimeError(f"找不到标签列 {target_col}")

    if "source_sample_id" not in df.columns:
        raise RuntimeError("patch 表缺少 source_sample_id 列，无法按原始样本聚合")

    # 1) 统计每个 physiological_state 的 patch 数
    stats = (
        df.groupby(target_col)
        .agg(
            n_patches=("sample_id", "count"),
            n_source_samples=("source_sample_id", "nunique"),
        )
        .reset_index()
    )
    stats = stats.sort_values("n_patches", ascending=False)

    stats_path = outdir / "physio_patch_stats.tsv"
    stats.to_csv(stats_path, sep="\t", index=False)
    print(f"[leak_check] 已写出每类 patch 数统计: {stats_path}")

    # 2) 行级划分下，同一 source_sample_id 是否同时出现在 train/test
    y = df[target_col].astype(str).to_numpy()
    X_dummy = np.zeros((len(df), 1), dtype=float)  # 只为调用 train_test_split，这里不关心特征

    _, _, idx_train, idx_test = train_test_split(
        X_dummy,
        np.arange(len(df)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    train_sources = set(df.iloc[idx_train]["source_sample_id"].astype(str))
    test_sources = set(df.iloc[idx_test]["source_sample_id"].astype(str))
    overlap_sources = sorted(train_sources & test_sources)

    print(
        f"[leak_check] 行级划分（test_size={args.test_size}）下："
        f"train source 样本数={len(train_sources)}, "
        f"test source 样本数={len(test_sources)}, "
        f"train/test 交集 source 数={len(overlap_sources)}"
    )

    overlap_path = outdir / "physio_patch_overlap_sources.tsv"
    pd.DataFrame({"source_sample_id": overlap_sources}).to_csv(
        overlap_path, sep="\t", index=False
    )
    print(f"[leak_check] train/test 同时出现的 source_sample_id 已写出: {overlap_path}")
    print("[leak_check] 说明：如果交集很大，说明传统 ML 的行级划分确实会在同一植株的不同 patch 上做 train/test。")


if __name__ == "__main__":
    main()

