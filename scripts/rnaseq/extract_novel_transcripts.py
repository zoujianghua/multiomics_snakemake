#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 gffcompare 输出的 annotated GTF 中提取 novel transcripts

功能：
1. 读取 gffcompare 输出的 annotated GTF（包含 class_code 信息）
2. 根据 class_code 筛选 novel transcripts（默认 "u"，可通过 config 扩展）
3. 规范化 gene_id 和 transcript_id（确保与原始 GTF 风格一致）
4. 输出标准化的 novel GTF 文件

输入：
- annotated GTF 文件（gffcompare 输出）
- class_codes：要提取的 class_code 列表（默认 ["u"]）

输出：
- 标准化的 novel GTF 文件
"""

import argparse
import re
import sys
from pathlib import Path


def parse_gtf_line(line):
    """
    解析 GTF 行，提取字段和属性
    
    返回：
        (fields_dict, attributes_dict) 或 None
    """
    if line.startswith('#') or not line.strip():
        return None
    
    fields = line.strip().split('\t')
    if len(fields) < 9:
        return None
    
    # 解析属性字段
    attributes_str = fields[8]
    attributes = {}
    
    # 提取 key="value" 格式的属性
    for match in re.finditer(r'(\w+)\s+"([^"]+)"', attributes_str):
        key, value = match.groups()
        attributes[key] = value
    
    return {
        'seqname': fields[0],
        'source': fields[1],
        'feature': fields[2],
        'start': fields[3],
        'end': fields[4],
        'score': fields[5],
        'strand': fields[6],
        'frame': fields[7],
        'attributes': attributes,
        'raw_line': line
    }


def normalize_novel_ids(transcript_id, gene_id, novel_counter):
    """
    规范化 novel transcript 和 gene 的 ID
    
    参数：
        transcript_id: 原始 transcript_id
        gene_id: 原始 gene_id（可能来自 gffcompare）
        novel_counter: 用于生成唯一 ID 的计数器
    
    返回：
        (normalized_gene_id, normalized_transcript_id)
    """
    # 如果 gene_id 已经符合格式，直接使用；否则生成新的
    if gene_id and not gene_id.startswith('gene:'):
        # 生成新的 gene_id
        normalized_gene_id = f"gene:novel{novel_counter['gene']:06d}"
        novel_counter['gene'] += 1
    else:
        normalized_gene_id = gene_id if gene_id else f"gene:novel{novel_counter['gene']:06d}"
        if not gene_id:
            novel_counter['gene'] += 1
    
    # 规范化 transcript_id
    if transcript_id and not transcript_id.startswith('transcript:'):
        # 从 gene_id 派生 transcript_id
        gene_base = normalized_gene_id.replace('gene:', '')
        normalized_transcript_id = f"transcript:{gene_base}.1"
    else:
        normalized_transcript_id = transcript_id if transcript_id else f"transcript:novel{novel_counter['transcript']:06d}.1"
        if not transcript_id:
            novel_counter['transcript'] += 1
    
    return normalized_gene_id, normalized_transcript_id


def extract_novel_transcripts(annotated_gtf, output_gtf, class_codes):
    """
    从 annotated GTF 中提取 novel transcripts
    
    参数：
        annotated_gtf: gffcompare 输出的 annotated GTF 文件路径
        output_gtf: 输出 novel GTF 文件路径
        class_codes: 要提取的 class_code 列表
    """
    novel_counter = {'gene': 1, 'transcript': 1}
    gene_id_map = {}  # 用于跟踪 transcript -> gene 映射
    
    novel_lines = []
    current_transcript = None
    current_gene = None
    
    print(f"[extract_novel] 读取 annotated GTF: {annotated_gtf}")
    print(f"[extract_novel] 筛选 class_code: {class_codes}")
    
    with open(annotated_gtf, 'r') as f:
        for line in f:
            parsed = parse_gtf_line(line)
            if parsed is None:
                continue
            
            attributes = parsed['attributes']
            class_code = attributes.get('class_code', '')
            
            # 只保留指定的 class_code
            if class_code not in class_codes:
                continue
            
            # 获取 transcript_id 和 gene_id
            transcript_id = attributes.get('transcript_id', '')
            gene_id = attributes.get('gene_id', '')
            
            # 如果是新转录本，规范化 ID
            if transcript_id and transcript_id not in gene_id_map:
                normalized_gene_id, normalized_transcript_id = normalize_novel_ids(
                    transcript_id, gene_id, novel_counter
                )
                gene_id_map[transcript_id] = normalized_gene_id
                current_transcript = normalized_transcript_id
                current_gene = normalized_gene_id
            elif transcript_id in gene_id_map:
                current_gene = gene_id_map[transcript_id]
                # 从 gene_id 派生 transcript_id
                gene_base = current_gene.replace('gene:', '')
                current_transcript = f"transcript:{gene_base}.1"
            
            # 更新属性
            new_attributes = attributes.copy()
            new_attributes['gene_id'] = current_gene
            new_attributes['transcript_id'] = current_transcript
            
            # 移除 class_code（可选，保留也可以）
            # new_attributes.pop('class_code', None)
            
            # 重建属性字符串
            attr_str = '; '.join([f'{k} "{v}"' for k, v in new_attributes.items()])
            
            # 重建 GTF 行
            new_line = '\t'.join([
                parsed['seqname'],
                parsed['source'],
                parsed['feature'],
                parsed['start'],
                parsed['end'],
                parsed['score'],
                parsed['strand'],
                parsed['frame'],
                attr_str
            ]) + '\n'
            
            novel_lines.append(new_line)
    
    # 写入输出文件
    output_path = Path(output_gtf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # 写入注释头
        f.write(f"# Novel transcripts extracted from {annotated_gtf}\n")
        f.write(f"# Class codes: {', '.join(class_codes)}\n")
        f.write(f"# Total novel features: {len(novel_lines)}\n")
        
        # 写入 novel GTF 行
        for line in novel_lines:
            f.write(line)
    
    print(f"[extract_novel] 提取了 {len(novel_lines)} 个 novel features")
    print(f"[extract_novel] 输出到: {output_gtf}")


def main():
    parser = argparse.ArgumentParser(
        description="从 gffcompare annotated GTF 中提取 novel transcripts"
    )
    parser.add_argument(
        '--annotated-gtf',
        required=True,
        help='gffcompare 输出的 annotated GTF 文件'
    )
    parser.add_argument(
        '--output-gtf',
        required=True,
        help='输出的 novel GTF 文件路径'
    )
    parser.add_argument(
        '--class-codes',
        nargs='+',
        default=['u'],
        help='要提取的 class_code 列表（默认: u）'
    )
    
    args = parser.parse_args()
    
    extract_novel_transcripts(
        args.annotated_gtf,
        args.output_gtf,
        args.class_codes
    )


if __name__ == '__main__':
    main()

