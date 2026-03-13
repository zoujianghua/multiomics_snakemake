#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将生理状态标签添加到 patch_features.tsv

从 patch_features_{patch_target}.tsv 读取数据，通过 source_sample_id -> image_features.phase -> 
phase_physiological_state_mapping.physiological_state 的路径，添加 physiological_state 列。

不修改 image_features.tsv，只生成新的 patch_features_with_physiological.tsv
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="将生理状态标签添加到 patch_features.tsv"
    )
    ap.add_argument(
        "--patch-features",
        required=True,
        help="输入的 patch_features_{patch_target}.tsv 路径",
    )
    ap.add_argument(
        "--image-features",
        required=True,
        help="image_features.tsv 路径（只读，不修改）",
    )
    ap.add_argument(
        "--mapping",
        required=True,
        help="phase_physiological_state_mapping.tsv 路径",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="输出的 patch_features_with_physiological.tsv 路径",
    )
    args = ap.parse_args()

    # ==================== 读取数据 ====================
    print(f"[add_physio] 读取 patch features: {args.patch_features}")
    patch_df = pd.read_csv(args.patch_features, sep="\t")
    print(f"[add_physio] patch 表: {len(patch_df)} 行 × {len(patch_df.columns)} 列")

    print(f"[add_physio] 读取 image features: {args.image_features}")
    image_df = pd.read_csv(args.image_features, sep="\t")
    print(f"[add_physio] image 表: {len(image_df)} 行 × {len(image_df.columns)} 列")

    print(f"[add_physio] 读取映射表: {args.mapping}")
    mapping_df = pd.read_csv(args.mapping, sep="\t")
    print(f"[add_physio] 映射表: {len(mapping_df)} 行")

    # ==================== 检查必需列 ====================
    # patch_features 需要 source_sample_id（或 sample_id）
    if "source_sample_id" not in patch_df.columns:
        if "sample_id" in patch_df.columns:
            # 如果只有 sample_id，假设它就是 source_sample_id
            patch_df["source_sample_id"] = patch_df["sample_id"]
        else:
            raise RuntimeError(
                f"patch_features 缺少必需列: source_sample_id 或 sample_id\n"
                f"当前列: {list(patch_df.columns)[:20]}..."
            )

    # image_features 需要 sample_id 和 phase
    if "sample_id" not in image_df.columns:
        raise RuntimeError("image_features.tsv 中缺少 'sample_id' 列")
    if "phase" not in image_df.columns:
        raise RuntimeError("image_features.tsv 中缺少 'phase' 列")

    # mapping 需要 phase 和 physiological_state
    if "phase" not in mapping_df.columns:
        raise RuntimeError("映射表中缺少 'phase' 列")
    if "physiological_state" not in mapping_df.columns:
        raise RuntimeError("映射表中缺少 'physiological_state' 列")

    # ==================== 合并逻辑 ====================
    # 步骤 1: patch_features.source_sample_id -> image_features.sample_id -> image_features.phase
    print(f"[add_physio] 步骤 1: 合并 image_features 的 phase 信息")
    image_phase = image_df[["sample_id", "phase"]].copy()
    patch_with_phase = patch_df.merge(
        image_phase,
        left_on="source_sample_id",
        right_on="sample_id",
        how="left",
        suffixes=("", "_image"),
    )

    # 检查匹配情况
    matched_phase = patch_with_phase["phase"].notna().sum()
    total_patches = len(patch_with_phase)
    print(
        f"[add_physio] phase 匹配: {matched_phase}/{total_patches} "
        f"({matched_phase/total_patches*100:.1f}%)"
    )

    # 步骤 2: phase -> physiological_state
    print(f"[add_physio] 步骤 2: 合并 physiological_state 映射")
    mapping_subset = mapping_df[["phase", "physiological_state"]].copy()
    patch_with_physio = patch_with_phase.merge(
        mapping_subset,
        on="phase",
        how="left",
        suffixes=("", "_physio"),
    )

    # 检查最终匹配情况
    matched_physio = patch_with_physio["physiological_state"].notna().sum()
    print(
        f"[add_physio] physiological_state 匹配: {matched_physio}/{total_patches} "
        f"({matched_physio/total_patches*100:.1f}%)"
    )

    # 检查未匹配的情况
    if matched_physio < total_patches:
        unmatched = patch_with_physio[
            patch_with_physio["physiological_state"].isna()
        ]
        unmatched_phases = unmatched["phase"].dropna().unique()
        if len(unmatched_phases) > 0:
            print(f"[add_physio] 警告: 以下 phase 未在映射表中找到:")
            for p in unmatched_phases[:10]:
                count = (unmatched["phase"] == p).sum()
                print(f"    - {p}: {count} 个 patch")
            if len(unmatched_phases) > 10:
                print(f"    ... 还有 {len(unmatched_phases) - 10} 个 phase")

    # ==================== 清理列名 ====================
    # 移除可能重复的列（如 sample_id_image）
    cols_to_drop = [c for c in patch_with_physio.columns if c.endswith("_image")]
    if cols_to_drop:
        patch_with_physio = patch_with_physio.drop(columns=cols_to_drop)

    # 确保 physiological_state 列存在（即使有些是 NaN）
    if "physiological_state" not in patch_with_physio.columns:
        raise RuntimeError("合并后未找到 physiological_state 列")

    # ==================== 保存结果 ====================
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patch_with_physio.to_csv(output_path, sep="\t", index=False)
    print(f"[add_physio] 已保存: {output_path}")
    print(f"[add_physio] 输出表: {len(patch_with_physio)} 行 × {len(patch_with_physio.columns)} 列")

    # ==================== 统计信息 ====================
    if matched_physio > 0:
        physio_counts = patch_with_physio["physiological_state"].value_counts()
        print(f"[add_physio] physiological_state 分布:")
        for physio, count in physio_counts.head(15).items():
            print(f"    {physio}: {count} 个 patch")
        if len(physio_counts) > 15:
            print(f"    ... 还有 {len(physio_counts) - 15} 个类别")


if __name__ == "__main__":
    main()
