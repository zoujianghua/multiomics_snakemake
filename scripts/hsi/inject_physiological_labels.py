#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSI 标签注入脚本
将 physiological_state 从 mapping 文件注入到 image 和 leaf 特征表中。
"""

import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-in", required=True)
    ap.add_argument("--leaf-in", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--image-out", required=True)
    ap.add_argument("--leaf-out", required=True)
    args = ap.parse_args()

    print(f"[Inject] 读取 Mapping 表: {args.mapping}")
    df_map = pd.read_csv(args.mapping, sep="\t")
    
    if "phase" not in df_map.columns or "physiological_state" not in df_map.columns:
        raise ValueError("Mapping文件必须包含 'phase' 和 'physiological_state' 列。")

    # 去重，防止 mapping 表有重复导致 merge 后数据行数爆炸
    df_map = df_map[["phase", "physiological_state"]].drop_duplicates()

    for path_in, path_out, level in [
        (args.image_in, args.image_out, "Image"),
        (args.leaf_in, args.leaf_out, "Leaf")
    ]:
        print(f"[Inject] 处理 {level} 级特征表...")
        df = pd.read_csv(path_in, sep="\t")
        
        # 丢弃旧的 physiological_state 列（如果存在）避免冲突
        if "physiological_state" in df.columns:
            df = df.drop(columns=["physiological_state"])
            
        # Left Join 注入新标签
        df_merged = df.merge(df_map, on="phase", how="left")
        
        # 统计缺失情况
        missing = df_merged["physiological_state"].isna().sum()
        if missing > 0:
            print(f"[WARN] {level} 表中有 {missing} 行未能匹配到 physiological_state。")

        out_file = Path(path_out)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df_merged.to_csv(out_file, sep="\t", index=False)
        print(f"[Inject] 写入完成: {out_file}")

if __name__ == "__main__":
    main()
