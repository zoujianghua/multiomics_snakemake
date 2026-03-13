#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 patch_features.tsv 中提取 image-level 信息（sample_id 和 target）

功能：
1. 读取 patch_features.tsv
2. 提取唯一的 source_sample_id 和 target 列
3. 输出 image-level 表（sample_id, target）

用于为 physiological_state 生成 split 和 patch_index
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="从 patch_features 中提取 image-level 信息"
    )
    ap.add_argument(
        "--patch-features",
        required=True,
        help="patch_features 文件路径"
    )
    ap.add_argument(
        "--target-col",
        required=True,
        help="目标列名（例如：physiological_state）"
    )
    ap.add_argument(
        "--out",
        required=True,
        help="输出路径（image-level TSV）"
    )
    args = ap.parse_args()

    # 读取 patch_features
    print(f"[extract_image] 读取 patch_features: {args.patch_features}")
    patch_df = pd.read_csv(args.patch_features, sep="\t")
    patch_df.columns = [c.strip().lower() for c in patch_df.columns]
    print(f"[extract_image] patch_features: {len(patch_df)} 行 × {len(patch_df.columns)} 列")

    # 检查必需的列
    if "source_sample_id" not in patch_df.columns:
        raise RuntimeError(
            f"patch_features 缺少 'source_sample_id' 列。"
            f"当前列：{list(patch_df.columns)[:20]}..."
        )

    target_col_lower = args.target_col.lower()
    if target_col_lower not in patch_df.columns:
        raise RuntimeError(
            f"patch_features 缺少目标列 '{target_col_lower}'。"
            f"当前列：{list(patch_df.columns)[:20]}..."
        )

    # 提取唯一的 sample_id 和 target
    img_level = patch_df[["source_sample_id", target_col_lower]].drop_duplicates()
    img_level = img_level.rename(columns={"source_sample_id": "sample_id"})
    img_level = img_level.dropna(subset=[target_col_lower])

    # 检查类别分布
    target_counts = img_level[target_col_lower].value_counts()
    print(f"[extract_image] 目标列 '{target_col_lower}' 的类别分布：")
    for cls, count in target_counts.items():
        print(f"  {cls}: {count} samples")

    # 保存结果
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_level.to_csv(out_path, sep="\t", index=False)
    print(f"[extract_image] 已保存: {out_path}")
    print(f"[extract_image] 输出: {len(img_level)} 行 × {len(img_level.columns)} 列")
    print("[extract_image] 完成！")


if __name__ == "__main__":
    main()
