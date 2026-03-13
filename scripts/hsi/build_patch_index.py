#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【HSI patch pipeline - 已废弃】
从 image_features.tsv + split.tsv 生成 patch 索引：
- 每行一个 patch（不保存 patch 数组，只保存位置和标签）
- patch 继承 image-level 的 train/test 划分，避免信息泄漏

DEPRECATED: 此脚本已被重构为两个脚本：
1. build_patch_index_base.py - 生成通用几何索引（不含 target/split）
2. make_patch_index_for_target.py - 基于 base 索引和 target 生成任务特定索引

保留此脚本仅用于兼容旧流程，新流程请使用上述两个脚本。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from imageio import v2 as imageio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True,
                    help="results/hsi/image_features.tsv")
    ap.add_argument("--split", required=True,
                    help="results/hsi/ml/split_phase_core_seed42.tsv")
    ap.add_argument("--out", required=True,
                    help="results/hsi/ml/patch_index_phase_core_seed42.tsv")
    ap.add_argument("--target", default="phase_core",
                    help="标签列名：phase / phase_core / metabo_state 等")
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--min-mask-frac", type=float, default=0.6,
                    help="patch 中 mask>0 像素占比至少多少才保留")
    ap.add_argument("--max-patches-per_image", type=int, default=200,
                    help="每张图最多保留多少 patch，避免爆炸")
    args = ap.parse_args()

    df = pd.read_csv(args.images, sep="\t")
    split_df = pd.read_csv(args.split, sep="\t")

    # 列名统一小写
    df.columns = [c.strip().lower() for c in df.columns]
    split_df.columns = [c.strip().lower() for c in split_df.columns]

    needed_cols = ["sample_id", "cube_npz", args.target.lower()]
    for col in needed_cols:
        if col not in df.columns:
            raise RuntimeError(f"image_features 缺少列: {col}")
    if "sample_id" not in split_df.columns or "split" not in split_df.columns:
        raise RuntimeError("split.tsv 必须包含 sample_id 和 split 列")

    df["sample_id"] = df["sample_id"].astype(str)
    split_df["sample_id"] = split_df["sample_id"].astype(str)

    # 合并 train/test 信息
    df = df.merge(split_df[["sample_id", "split"]], on="sample_id", how="inner")

    rows = []
    ps = args.patch_size
    stride = args.stride

    rng = np.random.default_rng(42)

    for i, row in df.iterrows():
        sid = row["sample_id"]
        cube_npz = row["cube_npz"]
        split = row["split"]
        target = str(row[args.target.lower()])

        cube_npz_path = Path(cube_npz)
        if not cube_npz_path.is_file():
            print(f"[WARN] cube_npz 不存在: {cube_npz_path}")
            continue

        data = np.load(cube_npz_path, allow_pickle=True)
        R = data["R"]               # (H, W, B)
        mask = data["mask"].astype(bool)  # 同样大小的 ROI mask

        H, W, _ = R.shape
        if mask.shape[:2] != (H, W):
            print(f"[WARN] mask 尺寸不匹配 {sid}: cube={R.shape}, mask={mask.shape}")
            continue

        candidates = []
        for y0 in range(0, H - ps + 1, stride):
            for x0 in range(0, W - ps + 1, stride):
                sub_mask = mask[y0:y0+ps, x0:x0+ps]
                frac = sub_mask.mean()
                if frac >= args.min_mask_frac:
                    candidates.append((y0, x0))

        if not candidates:
            print(f"[INFO] {sid} 没有满足条件的 patch")
            continue

        # 控制每图最多 patch 数量
        if len(candidates) > args.max_patches_per_image:
            idx = rng.choice(len(candidates),
                             size=args.max_patches_per_image,
                             replace=False)
            candidates = [candidates[j] for j in idx]

        for (y0, x0) in candidates:
            patch_id = f"{sid}_{y0}_{x0}"
            rows.append({
                "patch_id": patch_id,
                "sample_id": sid,
                "split": split,
                "target": target,
                "cube_npz": str(cube_npz_path),
                "y0": int(y0),
                "x0": int(x0),
                "size": ps,
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    patch_df = pd.DataFrame(rows)
    patch_df.to_csv(out_path, sep="\t", index=False)
    print(f"[patch_index] wrote {len(patch_df)} patches -> {out_path}")


if __name__ == "__main__":
    main()

