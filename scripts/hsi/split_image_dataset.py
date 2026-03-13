#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 image_features.tsv 生成统一的 train/test 划分：
- 每一行是一个 image-level 样本（和 image_features 一一对应）
- 按 phase 分层，避免某些 phase 全跑到 train 或 test
输出：split_phase_seed{seed}.tsv，列：sample_id, phase, split
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True,
                    help="results/hsi/image_features.tsv")
    ap.add_argument("--out", required=True,
                    help="results/hsi/ml/split_phase_seed42.tsv")
    ap.add_argument("--target", default="phase")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.images, sep="\t")
    if "sample_id" not in df.columns:
        raise RuntimeError("image_features.tsv 中缺少 sample_id 列")
    if args.target not in df.columns:
        raise RuntimeError(f"image_features.tsv 中缺少 target 列 {args.target}")

    y = df[args.target].astype(str)
    sid = df["sample_id"].astype(str)

    # 分层划分
    sid_train, sid_test, y_train, y_test = train_test_split(
        sid, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    split_df = pd.DataFrame({
        "sample_id": list(sid_train) + list(sid_test),
        "phase":    list(y_train)   + list(y_test),
        "split":    ["train"] * len(sid_train) + ["test"] * len(sid_test),
    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(out_path, sep="\t", index=False)
    print(f"[split] wrote {len(split_df)} rows -> {out_path}")

if __name__ == "__main__":
    main()

