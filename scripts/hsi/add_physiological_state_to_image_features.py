#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将生理状态标签添加到 image_features.tsv（临时版本，不覆盖原文件）

功能：
1. 读取 image_features.tsv
2. 读取 phase_physiological_state_mapping.tsv
3. 通过 phase 列合并，生成临时的 image_features_with_physio.tsv
4. 用于生成 split 和 patch_index，不覆盖原文件

注意：
- 输出文件是临时文件，仅用于 physiological.smk 流程
- 不修改原始的 image_features.tsv，避免触发整个项目重跑
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="将生理状态标签添加到 image_features.tsv（临时版本）"
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
        help="输出路径：results/hsi/image_features_with_physio.tsv（临时文件）"
    )
    args = ap.parse_args()

    # 读取 image_features
    print(f"[add_physio_to_image] 读取 image_features: {args.image_features}")
    img_df = pd.read_csv(args.image_features, sep="\t")
    img_df.columns = [c.strip().lower() for c in img_df.columns]
    print(f"[add_physio_to_image] image_features: {len(img_df)} 行 × {len(img_df.columns)} 列")

    if "phase" not in img_df.columns:
        raise RuntimeError(
            f"image_features 缺少 'phase' 列。"
            f"当前列：{list(img_df.columns)[:20]}..."
        )

    # 读取映射表
    print(f"[add_physio_to_image] 读取映射表: {args.mapping}")
    mapping_df = pd.read_csv(args.mapping, sep="\t")
    mapping_df.columns = [c.strip().lower() for c in mapping_df.columns]
    print(f"[add_physio_to_image] 映射表: {len(mapping_df)} 行")

    if "phase" not in mapping_df.columns or "physiological_state" not in mapping_df.columns:
        raise RuntimeError(
            f"映射表缺少必需列（phase 或 physiological_state）。"
            f"当前列：{list(mapping_df.columns)}"
        )

    # 统一 phase 类型
    img_df["phase"] = img_df["phase"].astype(str)
    mapping_df["phase"] = mapping_df["phase"].astype(str)

    # 合并
    print(f"[add_physio_to_image] 合并 physiological_state...")
    merged = img_df.merge(
        mapping_df[["phase", "physiological_state", "cluster_id"]],
        on="phase",
        how="left"
    )

    # 检查匹配情况
    matched = merged["physiological_state"].notna().sum()
    total = len(merged)
    unmatched = total - matched
    print(f"[add_physio_to_image] 匹配统计:")
    print(f"  总样本数: {total}")
    print(f"  匹配成功: {matched} ({matched/total*100:.1f}%)")
    print(f"  未匹配: {unmatched} ({unmatched/total*100:.1f}%)")

    # 检查未匹配的 phase
    if unmatched > 0:
        unmatched_phases = merged[merged["physiological_state"].isna()]["phase"].unique()
        print(f"[add_physio_to_image] 警告: 以下 phase 未在映射表中找到:")
        for p in unmatched_phases[:10]:  # 只显示前10个
            count = len(merged[merged["phase"] == p])
            print(f"    {p}: {count} samples")
        if len(unmatched_phases) > 10:
            print(f"    ... 还有 {len(unmatched_phases) - 10} 个 phase")

    # 检查 physiological_state 的分布
    if matched > 0:
        physio_counts = merged["physiological_state"].value_counts()
        print(f"[add_physio_to_image] physiological_state 分布:")
        for state, count in physio_counts.items():
            print(f"  {state}: {count} samples")

    # 保存结果
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, sep="\t", index=False)
    print(f"[add_physio_to_image] 已保存: {out_path}")
    print(f"[add_physio_to_image] 输出: {len(merged)} 行 × {len(merged.columns)} 列")
    print("[add_physio_to_image] 完成！")


if __name__ == "__main__":
    main()
