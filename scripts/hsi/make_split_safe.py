#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSI 防泄露数据划分脚本
采用基于 Image ID (sample_id) 的独立分层划分，自然解决 Leaf 级的 Group 泄露问题。
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--level", required=True, choices=["image", "leaf"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[Split Safe] 读取特征表: {args.features}")
    df = pd.read_csv(args.features, sep="\t")
    
    if "sample_id" not in df.columns or args.target_col not in df.columns:
        raise ValueError(f"输入文件缺少 sample_id 或 {args.target_col} 列。")

    # 去除目标列为空的行
    df = df.dropna(subset=[args.target_col])

    # 核心逻辑：无论输入是 image 还是 leaf，分类目标 (phase/physio) 都是绑定在母图像上的。
    # 我们只针对唯一的 sample_id 进行划分，这就保证了同一 sample_id 的所有叶片必定被分配到同一个数据集。
    df_unique = df[["sample_id", args.target_col]].drop_duplicates(subset=["sample_id"])
    
    # 类别平衡检查
    target_counts = df_unique[args.target_col].value_counts()
    min_count = target_counts.min()
    if min_count < 2:
        raise ValueError(
            f"独立图像中，类别 '{target_counts.idxmin()}' 样本数过少 ({min_count})，无法进行分层划分。"
        )

    print(f"[Split Safe] 对 {len(df_unique)} 个独立 Image 进行分层划分 (test_size={args.test_size})...")
    
    train_ids, test_ids = train_test_split(
        df_unique["sample_id"].values,
        stratify=df_unique[args.target_col].values,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True
    )

    # 构建标准格式的 split 字典表
    res = [{"sample_id": sid, "split": "train"} for sid in train_ids]
    res.extend([{"sample_id": sid, "split": "test"} for sid in test_ids])
    
    df_res = pd.DataFrame(res).sort_values("sample_id")
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_path, sep="\t", index=False)
    
    print(f"[Split Safe] 划分成功。Train images: {len(train_ids)}, Test images: {len(test_ids)}")
    print(f"[Split Safe] 结果已保存至: {out_path}")

if __name__ == "__main__":
    main()
