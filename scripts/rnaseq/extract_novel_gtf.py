#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 gffcompare 输出的 annotated GTF 中提取 novel transcripts/genes

功能：
1. 读取 gffcompare 输出的 annotated GTF（包含 class_code 属性）
2. 根据 class_code 筛选 novel 转录本（默认 u，可通过参数配置）
3. 规范化 gene_id 和 transcript_id 命名（gene:novel000001, transcript:novel000001.1）
4. 确保输出 GTF 格式符合 featureCounts 要求（gene_id, transcript_id 属性完整）

输入：
- annotated GTF 文件（gffcompare 输出）
- class_codes：要提取的 class_code 列表（默认 "u"）

输出：
- 规范化的 novel GTF 文件
"""

import argparse
import re
import sys
from pathlib import Path


def parse_class_code(line):
    """
    从 GTF 行中提取 class_code
    
    返回：
        str: class_code（如 "u", "x", "i" 等），如果不存在返回 None
    """
    # gffcompare 会在 attribute 中添加 class_code
    match = re.search(r'class_code\s+"([^"]+)"', line)
    if match:
        return match.group(1)
    return None


def extract_gene_transcript_ids(line):
    """
    从 GTF 行中提取 gene_id 和 transcript_id
    
    返回：
        tuple: (gene_id, transcript_id) 或 (None, None)
    """
    gene_match = re.search(r'gene_id\s+"([^"]+)"', line)
    transcript_match = re.search(r'transcript_id\s+"([^"]+)"', line)
    
    gene_id = gene_match.group(1) if gene_match else None
    transcript_id = transcript_match.group(1) if transcript_match else None
    
    return gene_id, transcript_id


def normalize_novel_ids(gene_id, transcript_id, novel_counter):
    """
    规范化 novel gene/transcript ID
    
    参数：
        gene_id: 原始 gene_id
        transcript_id: 原始 transcript_id
        novel_counter: novel gene 计数器（dict，用于生成唯一编号）
    
    返回：
        tuple: (normalized_gene_id, normalized_transcript_id)
    """
    # 如果 gene_id 不在计数器中，添加并分配新编号
    if gene_id not in novel_counter:
        novel_counter[gene_id] = len(novel_counter) + 1
    
    gene_num = novel_counter[gene_id]
    
    # 规范化 gene_id：gene:novel000001
    normalized_gene_id = f"gene:novel{str(gene_num).zfill(6)}"
    
    # 规范化 transcript_id：transcript:novel000001.1
    # 提取 transcript 编号（如果有）
    transcript_num = "1"
    if transcript_id:
        # 尝试从 transcript_id 中提取编号
        match = re.search(r'\.(\d+)$', transcript_id)
        if match:
            transcript_num = match.group(1)
    
    normalized_transcript_id = f"transcript:novel{str(gene_num).zfill(6)}.{transcript_num}"
    
    return normalized_gene_id, normalized_transcript_id


def extract_novel_gtf(annotated_gtf, output_gtf, class_codes):
    """
    从 annotated GTF 中提取 novel transcripts
    
    参数：
        annotated_gtf: gffcompare 输出的 annotated GTF 文件路径
        output_gtf: 输出 novel GTF 文件路径
        class_codes: 要提取的 class_code 列表（如 ["u", "x", "i"]）
    """
    class_codes_set = set(class_codes)
    
    # novel gene 计数器（沿用原来的逻辑）
    novel_counter = {}  # {original_gene_id: novel_index}
    
    # 用来记录"哪些原始 transcript 是 novel"
    novel_tx_ids = set()       # 原始 transcript_id 集合
    novel_gene_ids = set()     # 原始 gene_id 集合
    
    # 映射：原始 ID -> 规范化 ID
    gene_id_to_norm = {}       # {orig_gene_id: normalized_gene_id}
    tx_id_to_norm = {}         # {orig_tx_id: normalized_tx_id}
    
    print(f"[extract_novel] 读取 annotated GTF: {annotated_gtf}")
    print(f"[extract_novel] 提取 class_code: {', '.join(class_codes)}")
    
    # 先把所有行读进内存，方便两遍遍历
    with open(annotated_gtf, "r") as f:
        lines = f.readlines()
    
    # ---------- 第一遍：只看 transcript 行，找出 novel transcripts ----------
    for line in lines:
        if line.startswith("#"):
            continue
        
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 9:
            continue
        
        feature_type = fields[2]
        if feature_type != "transcript":
            # 只在 transcript 行上判断 class_code
            continue

        class_code = parse_class_code(line)
        if class_code not in class_codes_set:
            # 不是我们要的 u/x/i 这些，跳过
            continue

        # 提取原始 gene_id / transcript_id
        gene_id, transcript_id = extract_gene_transcript_ids(line)
        if transcript_id is None:
            # 正常情况下 gffcompare 的 transcript 行应该有 transcript_id，这里防御性处理
            continue
        
        # 如果 gene_id 缺失，可以跳过或用 transcript_id 代替，这里按防御性处理
        if gene_id is None:
            gene_id = transcript_id
        
        # 记录这是一个 novel transcript / gene
        novel_tx_ids.add(transcript_id)
        novel_gene_ids.add(gene_id)
        
        # 利用已有的 normalize_novel_ids() 为该原始 gene/transcript 分配规范化 ID
        norm_gene_id, norm_tx_id = normalize_novel_ids(
                            gene_id, transcript_id, novel_counter
                        )
        gene_id_to_norm[gene_id] = norm_gene_id
        tx_id_to_norm[transcript_id] = norm_tx_id
    
    # ---------- 第二遍：输出属于 novel transcripts 的所有行 ----------
    novel_lines = []
    
    for line in lines:
        if line.startswith("#"):
            # 注释行可以全部保留
            novel_lines.append(line)
            continue
        
        gene_id, transcript_id = extract_gene_transcript_ids(line)
        
        # 判断这一行是否属于 novel transcript / novel gene
        keep = False
        if transcript_id is not None and transcript_id in novel_tx_ids:
            keep = True
        elif (transcript_id is None) and (gene_id is not None) and (gene_id in novel_gene_ids):
            # 有些 gene 行可能只有 gene_id，没有 transcript_id
            keep = True
        
        if not keep:
            continue

        # 根据映射替换 gene_id 和 transcript_id
        normalized_line = line
        if gene_id is not None and gene_id in gene_id_to_norm:
            norm_gene_id = gene_id_to_norm[gene_id]
            normalized_line = re.sub(
                r'gene_id\s+"[^"]+"',
                f'gene_id "{norm_gene_id}"',
                normalized_line
            )
        if transcript_id is not None and transcript_id in tx_id_to_norm:
            norm_tx_id = tx_id_to_norm[transcript_id]
            normalized_line = re.sub(
                r'transcript_id\s+"[^"]+"',
                f'transcript_id "{norm_tx_id}"',
                normalized_line
            )
        novel_lines.append(normalized_line)
    
    # ---------- 打印统计信息并写出 ----------
    print(f"[extract_novel] 从 {annotated_gtf} 中识别到 {len(novel_gene_ids)} 个 novel genes，"
          f"{len(novel_tx_ids)} 个 novel transcripts")
    print(f"[extract_novel] 输出 GTF 行数: {len(novel_lines)}")
    
    output_path = Path(output_gtf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.writelines(novel_lines)
    
    print(f"[extract_novel] 输出 novel GTF: {output_gtf}")


def main():
    parser = argparse.ArgumentParser(
        description="从 gffcompare 输出的 annotated GTF 中提取 novel transcripts/genes"
    )
    parser.add_argument(
        "--annotated-gtf",
        required=True,
        help="gffcompare 输出的 annotated GTF 文件路径"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 novel GTF 文件路径"
    )
    parser.add_argument(
        "--class-codes",
        default="u",
        help="要提取的 class_code（逗号分隔，默认 u）"
    )
    
    args = parser.parse_args()
    
    # 解析 class_codes
    class_codes = [c.strip() for c in args.class_codes.split(",")]
    
    try:
        extract_novel_gtf(args.annotated_gtf, args.output, class_codes)
        print("[extract_novel] 完成！")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

