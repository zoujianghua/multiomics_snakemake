#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSI patch pipeline - Patch 特征表多组学导出（与 target 解耦版）

从 patch_features_*.tsv + image_features.tsv 生成可用于多组学关联分析的
patch 特征表 patch_features_*_omics.tsv（附加 phase / temp / time / session_id 等 image-level 信息）

功能：
1. 读取 patch_features 和 image_features
2. 通过 source_sample_id 匹配，附加 image-level 元信息（phase / phase_core / temp / time / session_id 等）
3. 输出列顺序：先元信息列，再特征列
4. 仅检查匹配质量，不关心具体 target 列
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="从 patch_features + image_features 生成多组学导出表"
    )
    ap.add_argument(
        "--patch",
        required=True,
        help="输入 patch 特征表路径，例如：results/hsi/patch_features_phase_core.tsv",
    )
    ap.add_argument(
        "--images",
        required=True,
        help="输入 image 特征表路径：results/hsi/image_features.tsv",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="输出 TSV 路径，例如：results/hsi/patch_features_phase_core_omics.tsv",
    )
    args = ap.parse_args()

    # ==================== 读取数据 ====================
    print(f"[patch_omics] 读取 patch 特征表: {args.patch}")
    patch_df = pd.read_csv(args.patch, sep="\t")
    patch_df.columns = [c.strip().lower() for c in patch_df.columns]
    print(f"[patch_omics] patch 表: {len(patch_df)} 行 × {len(patch_df.columns)} 列")

    print(f"[patch_omics] 读取 image 特征表: {args.images}")
    img_df = pd.read_csv(args.images, sep="\t")
    img_df.columns = [c.strip().lower() for c in img_df.columns]
    print(f"[patch_omics] image 表: {len(img_df)} 行 × {len(img_df.columns)} 列")

    # ==================== 必要列检查 ====================
    # patch_features 至少要有：
    #   - sample_id: patch 的 ID（稍后重命名为 patch_id）
    #   - source_sample_id: 对应的 image-level sample_id
    required_patch_cols = ["sample_id", "source_sample_id"]
    missing_patch_cols = [c for c in required_patch_cols if c not in patch_df.columns]
    if missing_patch_cols:
        raise RuntimeError(
            f"patch_features 缺少必需列: {missing_patch_cols}\n"
            f"当前列: {list(patch_df.columns)[:20]}..."
        )

    # image_features 至少要有：
    #   - sample_id
    #   - phase（你的实验设计里是核心分层变量）
    required_img_cols = ["sample_id", "phase"]
    missing_img_cols = [c for c in required_img_cols if c not in img_df.columns]
    if missing_img_cols:
        raise RuntimeError(
            f"image_features 缺少必需列: {missing_img_cols}\n"
            f"当前列: {list(img_df.columns)[:20]}..."
        )

    # ==================== 列重命名与 meta 选择 ====================
    # patch_df: sample_id -> patch_id
    patch_df = patch_df.rename(columns={"sample_id": "patch_id"})

    # img_df: 选择 meta 列（后续可以在这里扩展 metabo/rnaseq 的设计列）
    preferred_meta_cols = [
        "sample_id",
        "phase",
        "phase_core",
        "temp",
        "time",
        "session_id",
    ]
    meta_cols = [c for c in preferred_meta_cols if c in img_df.columns]
    meta_df = img_df[meta_cols].copy()
    print(f"[patch_omics] 从 image_features 选择 meta 列: {meta_cols}")

    # ==================== merge 逻辑 ====================
    print(f"[patch_omics] 开始 merge: patch.source_sample_id -> image.sample_id")
    merged = patch_df.merge(
        meta_df,
        left_on="source_sample_id",
        right_on="sample_id",
        how="left",
    )

    # ==================== 匹配质量检查 ====================
    total_patches = len(merged)
    matched_mask = merged["sample_id"].notna()
    matched_count = matched_mask.sum()
    unmatched_count = total_patches - matched_count
    unmatched_ratio = unmatched_count / total_patches if total_patches > 0 else 0.0

    print("[patch_omics] 匹配统计:")
    print(f"  总 patch 数: {total_patches}")
    print(f"  匹配成功: {matched_count} ({matched_count / total_patches * 100:.2f}%)")
    print(f"  匹配失败: {unmatched_count} ({unmatched_ratio * 100:.2f}%)")

    if unmatched_ratio > 0.01:
        # 显示一些未匹配的 source_sample_id 示例
        unmatched_samples = merged[~matched_mask]["source_sample_id"].unique()[:10]
        print("[patch_omics][WARN] 匹配失败比例 > 1%，示例未匹配的 source_sample_id:")
        for sid in unmatched_samples:
            print(f"    - {sid}")
        raise RuntimeError(
            f"匹配失败比例过高 ({unmatched_ratio * 100:.2f}% > 1%)，"
            "请检查 patch_features 中的 source_sample_id 是否与 image_features 中的 sample_id 一致"
        )

    # phase 缺失简单提醒一下
    if matched_count > 0:
        phase_missing = merged[matched_mask]["phase"].isna().sum()
        if phase_missing > 0:
            print(f"[patch_omics][WARN] 匹配成功的行中有 {phase_missing} 行 phase 为空")

    # ==================== 输出列顺序设计 ====================
    # 元信息列顺序（存在则保留）
    meta_cols_order = [
        "patch_id",         # patch-level ID
        "source_sample_id", # 对应的 image-level 样本
        "sample_id",        # merge 后 image 的 sample_id
        "phase",
        "phase_core",
        "temp",
        "time",
        "session_id",
    ]
    meta_cols_order = [c for c in meta_cols_order if c in merged.columns]

    # 其他列（特征列 + 任何额外列）
    all_cols = list(merged.columns)
    other_cols = [c for c in all_cols if c not in meta_cols_order]

    final_cols = meta_cols_order + other_cols
    merged = merged[final_cols]

    # ==================== 写出文件 ====================
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(out_path, sep="\t", index=False)

    print(f"[patch_omics] 输出: {args.out}")
    print(f"[patch_omics] 输出表: {len(merged)} 行 × {len(merged.columns)} 列")
    print(f"[patch_omics] 元信息列 ({len(meta_cols_order)}): {meta_cols_order}")
    print(f"[patch_omics] 特征列数量: {len(other_cols)}")
    print("[patch_omics] 完成！")


if __name__ == "__main__":
    main()

