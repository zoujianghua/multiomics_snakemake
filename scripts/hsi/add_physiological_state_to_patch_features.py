#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将生理状态标签添加到 patch_features.tsv

功能：
1. 读取已有的 patch_features_{patch_target}.tsv（包含所有 patch 的光谱特征）
2. 读取 phase_physiological_state_mapping.tsv（phase -> physiological_state 映射）
3. 通过 source_sample_id -> image_features.tsv -> phase -> physiological_state 链式合并
4. 输出 patch_features_physiological_state.tsv

注意：
- 不修改 image_features.tsv，避免触发整个项目重跑
- 通过 source_sample_id 和 phase 进行链式匹配
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="将生理状态标签添加到 patch_features.tsv"
    )
    ap.add_argument(
        "--patch-features",
        required=True,
        help="已有的 patch_features 文件，例如：results/hsi/patch_features_phase.tsv"
    )
    ap.add_argument(
        "--image-features",
        required=True,
        help="image_features.tsv：results/hsi/image_features.tsv"
    )
    ap.add_argument(
        "--mapping",
        required=True,
        help="phase -> physiological_state 映射表：config/phase_physiological_state_mapping.tsv"
    )
    ap.add_argument(
        "--out",
        required=True,
        help="输出路径：results/hsi/patch_features_physiological_state.tsv"
    )
    args = ap.parse_args()

    # 读取 patch_features
    print(f"[add_physio] 读取 patch_features: {args.patch_features}")
    patch_df = pd.read_csv(args.patch_features, sep="\t")
    patch_df.columns = [c.strip().lower() for c in patch_df.columns]
    print(f"[add_physio] patch_features: {len(patch_df)} 行 × {len(patch_df.columns)} 列")

    # 检查必需的列
    if "source_sample_id" not in patch_df.columns:
        raise RuntimeError(
            f"patch_features 缺少 'source_sample_id' 列。"
            f"当前列：{list(patch_df.columns)[:20]}..."
        )

    # 读取 image_features
    print(f"[add_physio] 读取 image_features: {args.image_features}")
    img_df = pd.read_csv(args.image_features, sep="\t")
    img_df.columns = [c.strip().lower() for c in img_df.columns]
    print(f"[add_physio] image_features: {len(img_df)} 行 × {len(img_df.columns)} 列")

    if "sample_id" not in img_df.columns or "phase" not in img_df.columns:
        raise RuntimeError(
            f"image_features 缺少必需列（sample_id 或 phase）。"
            f"当前列：{list(img_df.columns)[:20]}..."
        )

    # 读取映射表
    print(f"[add_physio] 读取映射表: {args.mapping}")
    mapping_df = pd.read_csv(args.mapping, sep="\t")
    mapping_df.columns = [c.strip().lower() for c in mapping_df.columns]
    print(f"[add_physio] 映射表: {len(mapping_df)} 行")

    if "phase" not in mapping_df.columns or "physiological_state" not in mapping_df.columns:
        raise RuntimeError(
            f"映射表缺少必需列（phase 或 physiological_state）。"
            f"当前列：{list(mapping_df.columns)}"
        )

    # 统一 sample_id 类型
    patch_df["source_sample_id"] = patch_df["source_sample_id"].astype(str)
    img_df["sample_id"] = img_df["sample_id"].astype(str)
    mapping_df["phase"] = mapping_df["phase"].astype(str)

    # 链式合并：patch_features -> image_features (添加 phase)
    print(f"[add_physio] 合并 patch_features 和 image_features...")
    merged = patch_df.merge(
        img_df[["sample_id", "phase"]],
        left_on="source_sample_id",
        right_on="sample_id",
        how="left",
        suffixes=("", "_img")
    )

    # 检查匹配情况
    matched_phase = merged["phase"].notna().sum()
    total = len(merged)
    print(f"[add_physio] phase 匹配: {matched_phase}/{total} ({matched_phase/total*100:.1f}%)")

    # 链式合并：添加 physiological_state
    print(f"[add_physio] 合并 physiological_state...")
    merged = merged.merge(
        mapping_df[["phase", "physiological_state", "cluster_id"]],
        on="phase",
        how="left"
    )

    # 检查匹配情况
    matched_physio = merged["physiological_state"].notna().sum()
    print(f"[add_physio] physiological_state 匹配: {matched_physio}/{total} ({matched_physio/total*100:.1f}%)")

    # 检查未匹配的样本
    if matched_physio < total:
        unmatched = merged[merged["physiological_state"].isna()]
        unmatched_samples = unmatched["source_sample_id"].unique()[:10]
        print(f"[add_physio] 警告: {total - matched_physio} 个 patch 未匹配到 physiological_state")
        print(f"[add_physio] 未匹配的样本示例（前10个）:")
        for s in unmatched_samples:
            phase_val = unmatched[unmatched["source_sample_id"] == s]["phase"].iloc[0] if len(unmatched[unmatched["source_sample_id"] == s]) > 0 else "N/A"
            print(f"    {s}: phase={phase_val}")

    # 移除临时列（sample_id_img）
    if "sample_id_img" in merged.columns:
        merged = merged.drop(columns=["sample_id_img"])

    # 检查 physiological_state 的分布
    if matched_physio > 0:
        physio_counts = merged["physiological_state"].value_counts()
        print(f"[add_physio] physiological_state 分布:")
        for state, count in physio_counts.items():
            print(f"  {state}: {count} patches")

    # 保存结果
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, sep="\t", index=False)
    print(f"[add_physio] 已保存: {out_path}")
    print(f"[add_physio] 输出: {len(merged)} 行 × {len(merged.columns)} 列")
    print("[add_physio] 完成！")


if __name__ == "__main__":
    main()
