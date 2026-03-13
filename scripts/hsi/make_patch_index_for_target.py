#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【HSI patch pipeline - 任务特定索引生成】

从 base 索引（包含 cube_patch_npz）和 image metadata + split 生成任务特定索引。

功能：
1. 读取 base_index（包含 patch_id, sample_id, cube_patch_npz, patch_idx, y0, x0, size）
2. 从 image_features.tsv 中提取 target_col（例如 phase_core, metabo_state）
3. 从 split_<target>_seed<seed>.tsv 中提取 split（train/test）
4. 合并生成最终的 patch_index_<target>_seed<seed>.tsv，包含：
   - patch_id, sample_id, cube_patch_npz, patch_idx
   - split, target
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="从 base 索引生成任务特定索引（添加 target 和 split）"
    )
    ap.add_argument(
        "--base-index",
        required=True,
        help="base 索引文件：results/hsi/ml/patch_index_base_cubes.tsv"
    )
    ap.add_argument(
        "--image-meta",
        required=True,
        help="image metadata：results/hsi/image_features.tsv"
    )
    ap.add_argument(
        "--split",
        required=True,
        help="split 文件：results/hsi/ml/split_<target>_seed<seed>.tsv"
    )
    ap.add_argument(
        "--target-col",
        required=True,
        help="目标列名：phase_core / metabo_state / rnaseq_state 等"
    )
    ap.add_argument(
        "--out",
        required=True,
        help="输出路径：results/hsi/ml/patch_index_<target>_seed<seed>.tsv"
    )
    args = ap.parse_args()

    # 读取 base 索引
    print(f"[make_patch_index] 读取 base 索引: {args.base_index}")
    df_base = pd.read_csv(args.base_index, sep="\t")
    df_base.columns = [c.strip().lower() for c in df_base.columns]

    required_base_cols = ["sample_id", "cube_patch_npz", "patch_idx"]
    missing_cols = [col for col in required_base_cols if col not in df_base.columns]
    if missing_cols:
        raise RuntimeError(f"base_index 缺少必需的列: {missing_cols}")

    # 读取 image metadata
    print(f"[make_patch_index] 读取 image metadata: {args.image_meta}")
    df_meta = pd.read_csv(args.image_meta, sep="\t")
    df_meta.columns = [c.strip().lower() for c in df_meta.columns]

    target_col_lower = args.target_col.lower()
    if target_col_lower not in df_meta.columns:
        raise RuntimeError(f"image_meta 缺少目标列: {target_col_lower}")

    # 读取 split
    print(f"[make_patch_index] 读取 split: {args.split}")
    df_split = pd.read_csv(args.split, sep="\t")
    df_split.columns = [c.strip().lower() for c in df_split.columns]

    if "sample_id" not in df_split.columns or "split" not in df_split.columns:
        raise RuntimeError("split 文件必须包含 sample_id 和 split 列")

    # 统一 sample_id 类型
    df_base["sample_id"] = df_base["sample_id"].astype(str)
    df_meta["sample_id"] = df_meta["sample_id"].astype(str)
    df_split["sample_id"] = df_split["sample_id"].astype(str)

    # 合并：base -> meta (添加 target)
    print(f"[make_patch_index] 合并 base 和 metadata...")
    df_result = df_base.merge(
        df_meta[["sample_id", target_col_lower]],
        on="sample_id",
        how="inner"
    )

    # 合并：添加 split
    print(f"[make_patch_index] 合并 split...")
    df_result = df_result.merge(
        df_split[["sample_id", "split"]],
        on="sample_id",
        how="inner"
    )

    # 重命名 target 列为 "target"
    df_result = df_result.rename(columns={target_col_lower: "target"})

    # 确保 target 是字符串类型
    df_result["target"] = df_result["target"].astype(str)

    # 选择输出列
    output_cols = ["patch_id", "sample_id", "cube_patch_npz", "patch_idx", "split", "target"]
    
    # 如果 base 中有 cube_npz, y0, x0, size，也保留（cube_npz 用于获取 mask，y0/x0/size 方便 debug）
    optional_cols = ["cube_npz", "y0", "x0", "size"]
    for col in optional_cols:
        if col in df_result.columns:
            output_cols.append(col)

    # 确保所有必需的列都存在
    missing_output_cols = [col for col in output_cols if col not in df_result.columns]
    if missing_output_cols:
        raise RuntimeError(f"合并后的数据缺少输出列: {missing_output_cols}")

    df_result = df_result[output_cols].copy()

    # 保存结果
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(out_path, sep="\t", index=False)

    print(f"[make_patch_index] 完成！")
    print(f"[make_patch_index] 总 patch 数: {len(df_result)}")
    print(f"[make_patch_index] train: {len(df_result[df_result['split'] == 'train'])}")
    print(f"[make_patch_index] test: {len(df_result[df_result['split'] == 'test'])}")
    print(f"[make_patch_index] 输出文件: {out_path}")


if __name__ == "__main__":
    main()

