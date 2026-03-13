#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【HSI patch pipeline - 任务特定数据划分】

为指定的 target（phase_core / metabo_state / rnaseq_state 等）生成 train/test 划分。

功能：
1. 从 image_features.tsv 读取 sample_id 和 target_col
2. 按 target_col 做分层采样（stratified split），防止类别极度不平衡
3. 输出 split_{target}_seed{seed}.tsv，包含 sample_id 和 split（train/test）

注意：
- 这是 sample 级别的划分，后续 patch 会继承所属 sample 的 split
- 默认 80/20 划分，可通过参数调整
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser(
        description="为指定 target 生成 train/test 划分（分层采样）"
    )
    ap.add_argument(
        "--image-meta",
        required=True,
        help="image metadata：results/hsi/image_features.tsv"
    )
    ap.add_argument(
        "--target-col",
        required=True,
        help="目标列名：phase_core / metabo_state / rnaseq_state 等"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）"
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="测试集比例（默认 0.2，即 80/20 划分）"
    )
    ap.add_argument(
        "--out",
        required=True,
        help="输出路径：results/hsi/ml/split_{target}_seed{seed}.tsv"
    )
    args = ap.parse_args()

    # 读取 image metadata
    print(f"[make_split] 读取 image metadata: {args.image_meta}")
    df = pd.read_csv(args.image_meta, sep="\t")
    
    # 检查必要的列
    if "sample_id" not in df.columns:
        raise RuntimeError(f"输入文件缺少 sample_id 列")
    
    if args.target_col not in df.columns:
        raise RuntimeError(
            f"输入文件缺少目标列 '{args.target_col}'。"
            f"可用列：{list(df.columns)}"
        )
    
    # 提取 sample_id 和 target
    df_split = df[["sample_id", args.target_col]].copy()
    df_split = df_split.dropna(subset=[args.target_col]).reset_index(drop=True)
    
    # 检查类别分布
    target_counts = df_split[args.target_col].value_counts()
    print(f"[make_split] 目标列 '{args.target_col}' 的类别分布：")
    for cls, count in target_counts.items():
        print(f"  {cls}: {count} samples")
    
    # 如果某个类别样本数太少，给出警告
    min_samples = target_counts.min()
    if min_samples < 2:
        raise RuntimeError(
            f"类别 '{target_counts.idxmin()}' 只有 {min_samples} 个样本，"
            f"无法进行分层划分。请检查数据或调整 target_col。"
        )
    
    # 分层划分
    print(f"[make_split] 进行分层划分（test_size={args.test_size}, seed={args.seed}）...")
    train_samples, test_samples = train_test_split(
        df_split["sample_id"].values,
        stratify=df_split[args.target_col].values,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True
    )
    
    # 构建输出 DataFrame
    result_rows = []
    for sample_id in train_samples:
        result_rows.append({"sample_id": sample_id, "split": "train"})
    for sample_id in test_samples:
        result_rows.append({"sample_id": sample_id, "split": "test"})
    
    df_result = pd.DataFrame(result_rows)
    df_result = df_result.sort_values("sample_id").reset_index(drop=True)
    
    # 验证划分结果
    train_count = len(df_result[df_result["split"] == "train"])
    test_count = len(df_result[df_result["split"] == "test"])
    print(f"[make_split] 划分结果：train={train_count}, test={test_count}")
    
    # 检查每个类别的 train/test 分布
    df_result_with_target = df_result.merge(
        df_split[["sample_id", args.target_col]],
        on="sample_id",
        how="left"
    )
    print(f"[make_split] 各类别在 train/test 中的分布：")
    for cls in sorted(df_result_with_target[args.target_col].unique()):
        cls_df = df_result_with_target[df_result_with_target[args.target_col] == cls]
        train_cls = len(cls_df[cls_df["split"] == "train"])
        test_cls = len(cls_df[cls_df["split"] == "test"])
        print(f"  {cls}: train={train_cls}, test={test_cls}")
    
    # 保存结果
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_result.to_csv(out_path, sep="\t", index=False)
    print(f"[make_split] 保存划分结果: {out_path}")
    print("[make_split] 完成！")


if __name__ == "__main__":
    main()
