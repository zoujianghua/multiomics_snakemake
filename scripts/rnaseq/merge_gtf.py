#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 reference GTF 和 novel GTF

功能：
1. 读取 reference GTF 和 novel GTF
2. 检查 gene_id 重复情况
3. 合并两个 GTF，确保格式正确
4. 输出合并后的 GTF

输入：
- reference GTF 文件
- novel GTF 文件

输出：
- 合并后的 GTF 文件
"""

import argparse
import re
import sys
from pathlib import Path


def extract_gene_ids(gtf_file):
    """
    从 GTF 文件中提取所有 gene_id
    
    返回：
        set: gene_id 集合
    """
    gene_ids = set()
    
    with open(gtf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            match = re.search(r'gene_id\s+"([^"]+)"', line)
            if match:
                gene_ids.add(match.group(1))
    
    return gene_ids


def merge_gtf(ref_gtf, novel_gtf, output_gtf):
    """
    合并 reference GTF 和 novel GTF
    
    参数：
        ref_gtf: reference GTF 文件路径
        novel_gtf: novel GTF 文件路径
        output_gtf: 输出 GTF 文件路径
    """
    print(f"[merge_gtf] 读取 reference GTF: {ref_gtf}")
    ref_genes = extract_gene_ids(ref_gtf)
    print(f"[merge_gtf] reference GTF 包含 {len(ref_genes)} 个 genes")
    
    print(f"[merge_gtf] 读取 novel GTF: {novel_gtf}")
    novel_genes = extract_gene_ids(novel_gtf)
    print(f"[merge_gtf] novel GTF 包含 {len(novel_genes)} 个 genes")
    
    # 检查重复
    overlap = ref_genes & novel_genes
    if overlap:
        print(f"[WARN] 发现 {len(overlap)} 个重复的 gene_id:", file=sys.stderr)
        for gene_id in sorted(list(overlap))[:10]:  # 只显示前10个
            print(f"  {gene_id}", file=sys.stderr)
        if len(overlap) > 10:
            print(f"  ... 还有 {len(overlap) - 10} 个", file=sys.stderr)
        print("[WARN] novel GTF 中的重复 gene_id 将被跳过（保留 reference 的注释）", file=sys.stderr)
    
    # 创建输出目录
    output_path = Path(output_gtf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 合并 GTF
    print(f"[merge_gtf] 合并 GTF 文件...")
    novel_genes_set = novel_genes
    
    with open(output_gtf, 'w') as out_f:
        # 先写入 reference GTF
        with open(ref_gtf, 'r') as ref_f:
            for line in ref_f:
                out_f.write(line)
        
        # 再写入 novel GTF（跳过重复的 gene_id）
        skipped = 0
        added = 0
        current_gene_id = None
        
        with open(novel_gtf, 'r') as novel_f:
            for line in novel_f:
                if line.startswith('#'):
                    # 跳过注释行（避免重复）
                    continue
                
                # 提取 gene_id
                match = re.search(r'gene_id\s+"([^"]+)"', line)
                if match:
                    current_gene_id = match.group(1)
                
                # 如果 gene_id 不在 reference 中，写入
                if current_gene_id and current_gene_id not in ref_genes:
                    out_f.write(line)
                    added += 1
                elif current_gene_id and current_gene_id in ref_genes:
                    skipped += 1
    
    print(f"[merge_gtf] 完成！")
    print(f"[merge_gtf] reference: {len(ref_genes)} genes")
    print(f"[merge_gtf] novel: {len(novel_genes)} genes")
    print(f"[merge_gtf] 合并后: {len(ref_genes) + len(novel_genes) - len(overlap)} genes")
    print(f"[merge_gtf] 新增: {len(novel_genes) - len(overlap)} genes")
    print(f"[merge_gtf] 输出: {output_gtf}")


def main():
    parser = argparse.ArgumentParser(
        description="合并 reference GTF 和 novel GTF"
    )
    parser.add_argument(
        "--ref-gtf",
        required=True,
        help="reference GTF 文件路径"
    )
    parser.add_argument(
        "--novel-gtf",
        required=True,
        help="novel GTF 文件路径"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出合并后的 GTF 文件路径"
    )
    
    args = parser.parse_args()
    
    try:
        merge_gtf(args.ref_gtf, args.novel_gtf, args.output)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

