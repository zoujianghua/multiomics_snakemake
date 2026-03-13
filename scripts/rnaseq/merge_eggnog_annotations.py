#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 reference 和 novel 的 eggNOG 注释表

功能：
1. 读取 reference 和 novel 的 eggNOG 注释表
2. 检查 gene_id 重复情况
3. 合并两个表，避免重复（如果 novel 的 gene_id 与 reference 重复，记录警告）
4. 输出合并后的注释表

输入：
- reference eggNOG 注释表
- novel eggNOG 注释表

输出：
- 合并后的 eggNOG 注释表
"""

import argparse
import pandas as pd
import sys


def main():
    parser = argparse.ArgumentParser(
        description="合并 reference 和 novel 的 eggNOG 注释表"
    )
    parser.add_argument(
        "--ref-annot",
        required=True,
        help="reference eggNOG 注释表路径"
    )
    parser.add_argument(
        "--novel-annot",
        required=True,
        help="novel eggNOG 注释表路径"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出合并后的注释表路径"
    )
    
    args = parser.parse_args()
    
    try:
        # 读取两个注释表
        print(f"[merge_eggnog] 读取 reference 注释: {args.ref_annot}")
        df_ref = pd.read_csv(args.ref_annot, sep="\t")
        
        print(f"[merge_eggnog] 读取 novel 注释: {args.novel_annot}")
        df_novel = pd.read_csv(args.novel_annot, sep="\t")
        
        # 检查列名是否一致
        if list(df_ref.columns) != list(df_novel.columns):
            print("[WARN] reference 和 novel 注释表的列名不一致", file=sys.stderr)
            print(f"  reference 列: {list(df_ref.columns)}", file=sys.stderr)
            print(f"  novel 列: {list(df_novel.columns)}", file=sys.stderr)
        
        # 检查 gene_id 重复
        ref_genes = set(df_ref["gene_id"].dropna())
        novel_genes = set(df_novel["gene_id"].dropna())
        overlap = ref_genes & novel_genes
        
        if overlap:
            print(f"[WARN] 发现 {len(overlap)} 个重复的 gene_id:", file=sys.stderr)
            for gene_id in sorted(list(overlap))[:10]:  # 只显示前10个
                print(f"  {gene_id}", file=sys.stderr)
            if len(overlap) > 10:
                print(f"  ... 还有 {len(overlap) - 10} 个", file=sys.stderr)
            print("[WARN] novel 的重复 gene_id 将被跳过（保留 reference 的注释）", file=sys.stderr)
        
        # 合并：先取 reference，再添加 novel 中不重复的部分
        df_merged = df_ref.copy()
        
        # 只添加 novel 中不在 reference 中的 gene_id
        df_novel_unique = df_novel[~df_novel["gene_id"].isin(ref_genes)]
        
        if len(df_novel_unique) > 0:
            df_merged = pd.concat([df_merged, df_novel_unique], ignore_index=True)
            print(f"[merge_eggnog] 添加了 {len(df_novel_unique)} 个 novel 注释")
        else:
            print("[merge_eggnog] 没有新的 novel 注释需要添加（所有 gene_id 都已存在于 reference）")
        
        # 按 gene_id 排序
        df_merged = df_merged.sort_values("gene_id").reset_index(drop=True)
        
        # 保存结果
        print(f"[merge_eggnog] 保存合并后的注释表: {args.output}")
        df_merged.to_csv(args.output, sep="\t", index=False, encoding="utf-8")
        
        print(f"[merge_eggnog] 完成！")
        print(f"[merge_eggnog] reference: {len(df_ref)} genes")
        print(f"[merge_eggnog] novel: {len(df_novel)} genes")
        print(f"[merge_eggnog] 合并后: {len(df_merged)} genes")
        print(f"[merge_eggnog] 新增: {len(df_merged) - len(df_ref)} genes")
        
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

