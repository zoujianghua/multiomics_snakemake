#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整理 eggNOG-mapper 的注释结果，按 gene_id 聚合

功能：
1. 读取 eggNOG 的 .emapper.annotations 文件
2. 从 GTF 文件中建立 transcript_id -> gene_id 的映射
3. 将 transcript_id 转换为 gene_id（因为 gffread 生成的蛋白质序列 ID 是 transcript_id）
4. 按 gene_id 聚合注释信息（KEGG、GO、描述等）
5. 输出标准 TSV 格式，与 featureCounts 的 gene_id 格式一致

输入：
- eggNOG annotations 文件（TSV 格式）
- GTF 文件（用于 transcript -> gene 映射）

输出：
- 按 gene_id 汇总的注释表（TSV 格式，UTF-8 编码）
"""

import argparse
import pandas as pd
import re
import sys
from collections import defaultdict


def parse_gtf_for_transcript_gene_mapping(gtf_file):
    """
    从 GTF 文件中提取 transcript_id -> gene_id 的映射
    
    返回：
        dict: {transcript_id: gene_id}
    """
    mapping = {}
    
    with open(gtf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            # 提取 transcript_id 和 gene_id
            # 格式示例: transcript_id "transcript:Mm01G000001"; gene_id "gene:Mm01G000001"
            transcript_match = re.search(r'transcript_id\s+"([^"]+)"', line)
            gene_match = re.search(r'gene_id\s+"([^"]+)"', line)
            
            if transcript_match and gene_match:
                transcript_id_raw = transcript_match.group(1)
                gene_id = gene_match.group(1)

                 # 对 transcript_id 做同样的规范化处理
                transcript_id = extract_transcript_id_from_query(transcript_id_raw)

                mapping[transcript_id] = gene_id

    
    return mapping


def read_eggnog_annotations(annot_file):
    """
    读取 eggNOG annotations 文件

    - 跳过前面的 "##" 元数据行
    - 保留以 "#query" 开头的表头行
    """
    skip_rows = 0
    with open(annot_file, "r", encoding="utf-8") as f:
        for line in f:
            # 只跳过以 "##" 开头的元数据行
            if line.startswith("##"):
                skip_rows += 1
            else:
                # 碰到 "#query" 或数据行就停，这一行交给 pandas 做表头
                break

    df = pd.read_csv(
        annot_file,
        sep="\t",
        skiprows=skip_rows,
        low_memory=False,
    )

    # 安全检查一下列名
    if "#query" not in df.columns:
        raise RuntimeError(
            f"在 {annot_file} 中找不到 '#query' 列，实际列名为: {list(df.columns)}"
        )

    return df



def extract_transcript_id_from_query(query):
    """
    将 query/transcript ID 规范化，便于和 GTF 中的 transcript_id 匹配：

    典型输入示例：
    - 'transcript:Mm01G000001'
    - 'transcript:Mm01G000001.1'
    - 'transcript:Mm01G000001.p1'
    - 'transcript:novel000753.1.p1'
    - 'novel000753.1.p1'
    - 'TRINITY_DN100_c0_g1_i1.p1'

    规范化策略：
    1. 去掉前缀 'transcript:' 或 'gene:'
    2. 去掉空格后的所有内容（只保留第一个 token）
    3. 去掉 '|' 后面的内容（有些软件会写成 foo|bar）
    4. 去掉 TransDecoder 样式的蛋白后缀，例如 '.p1', '.p2'
    5. 去掉简单的数字版本号后缀，例如 '.1', '.2'
    """
    q = str(query).strip()

    # 1. 去前缀
    if q.startswith('transcript:'):
        q = q[len('transcript:'):]
    elif q.startswith('gene:'):
        q = q[len('gene:'):]

    # 2. 只保留第一个空格前的 token
    q = q.split()[0]

    # 3. 只保留 '|' 前面的部分
    q = q.split('|')[0]

    # 4. 去掉 TransDecoder 的蛋白后缀 .p1 / .p2 / …
    q = re.sub(r'\.p\d+.*$', '', q)

    # 5. 再去掉纯数字版本号，如 Mm01G000001.1 -> Mm01G000001
    q = re.sub(r'\.\d+$', '', q)

    return q



def aggregate_annotations_by_gene(df, transcript_gene_mapping):
    """
    按 gene_id 聚合注释信息
    
    参数：
        df: eggNOG 注释数据框
        transcript_gene_mapping: transcript_id -> gene_id 映射字典
    
    返回：
        pd.DataFrame: 按 gene_id 聚合后的数据框
    """
    # 添加 gene_id 列
    df['transcript_id'] = df['#query'].apply(extract_transcript_id_from_query)
    df['gene_id'] = df['transcript_id'].map(transcript_gene_mapping)
    
    # 移除无法映射到 gene_id 的行（可能是格式问题）
    df = df[df['gene_id'].notna()].copy()
    
    if len(df) == 0:
        raise ValueError("No transcripts could be mapped to gene_id. Please check GTF format.")
    
    # 定义需要聚合的列
    # 根据 eggNOG-mapper v2 的输出格式调整列名
    kegg_ko_col = None
    kegg_pathway_col = None
    go_terms_col = None
    description_col = None
    
    # 尝试匹配可能的列名
    for col in df.columns:
        col_lower = col.lower()
        if 'kegg_ko' in col_lower or col_lower == 'kegg_ko':
            kegg_ko_col = col
        elif 'kegg_pathway' in col_lower or 'pathway' in col_lower:
            kegg_pathway_col = col
        elif col in ["GOs", "Gene_Ontology_terms", "Gene_Ontology", "GO", "go_terms"]:
            go_terms_col = col
        elif 'description' in col_lower or 'predicted' in col_lower:
            description_col = col
    
    # 如果找不到标准列名，使用常见的列名（更宽松的匹配）
    if kegg_ko_col is None:
        # 尝试查找包含 KEGG KO 信息的列
        for col in df.columns:
            col_lower = col.lower()
            if 'kegg' in col_lower and ('ko' in col_lower or col_lower.endswith('ko')):
                kegg_ko_col = col
                break
    
    if kegg_pathway_col is None:
        # 尝试查找 KEGG 通路列
        for col in df.columns:
            col_lower = col.lower()
            if 'pathway' in col_lower and 'kegg' in col_lower:
                kegg_pathway_col = col
                break
    
    if go_terms_col is None:
        # 尝试查找 GO 相关列
        for col in df.columns:
            col_lower = col.lower()
            if 'go' in col_lower and ('term' in col_lower or 'ontology' in col_lower):
                go_terms_col = col
                break
    
    if description_col is None:
        # 使用 eggNOG 描述列（常见列名）
        for col in df.columns:
            col_lower = col.lower()
            if 'description' in col_lower or ('predicted' in col_lower and 'name' in col_lower):
                description_col = col
                break
        # 如果还是找不到，尝试使用 eggNOG 的标准列名
        if description_col is None:
            for col in ['eggNOG_free_text_description', 'Preferred_name', 'Description']:
                if col in df.columns:
                    description_col = col
                    break
    
    # 聚合函数：对于每个 gene_id，合并所有 transcript 的注释
    def aggregate_values(series):
        """聚合一个系列的值，去重并用分号分隔"""
        values = series.dropna().unique()
        values = [str(v).strip() for v in values if str(v).strip() and str(v).strip() != 'nan']
        return ';'.join(sorted(set(values))) if values else ''
    
    # 手动聚合各个字段（不使用 groupby.agg，因为需要自定义聚合逻辑）
    result_rows = []
    for gene_id, group in df.groupby('gene_id'):
        row = {'gene_id': gene_id}
        
        # 聚合 KEGG KO
        if kegg_ko_col and kegg_ko_col in group.columns:
            row['kegg_ko'] = aggregate_values(group[kegg_ko_col])
        else:
            row['kegg_ko'] = ''
        
        # 聚合 KEGG Pathway
        if kegg_pathway_col and kegg_pathway_col in group.columns:
            row['kegg_pathway'] = aggregate_values(group[kegg_pathway_col])
        else:
            row['kegg_pathway'] = ''
        
        # 聚合 GO terms
        if go_terms_col and go_terms_col in group.columns:
            row['go_terms'] = aggregate_values(group[go_terms_col])
        else:
            row['go_terms'] = ''
        
        # 聚合描述（取第一个非空描述）
        if description_col and description_col in group.columns:
            descs = group[description_col].dropna().unique()
            descs = [str(d).strip() for d in descs if str(d).strip() and str(d).strip() != 'nan']
            row['description'] = descs[0] if descs else ''
        else:
            row['description'] = ''
        
        result_rows.append(row)
    
    result_df = pd.DataFrame(result_rows)
    
    # 确保列顺序
    columns_order = ['gene_id', 'kegg_ko', 'kegg_pathway', 'go_terms', 'description']
    result_df = result_df[columns_order]
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description='整理 eggNOG-mapper 注释结果，按 gene_id 聚合'
    )
    parser.add_argument(
        '--annotations',
        required=True,
        help='eggNOG annotations 文件路径 (.emapper.annotations)'
    )
    parser.add_argument(
        '--gtf',
        required=True,
        help='GTF 文件路径（用于 transcript -> gene 映射）'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='输出文件路径（TSV 格式）'
    )
    
    args = parser.parse_args()
    
    try:
        # 1. 从 GTF 提取 transcript -> gene 映射
        print(f"Reading GTF file: {args.gtf}")
        transcript_gene_mapping = parse_gtf_for_transcript_gene_mapping(args.gtf)
        print(f"Found {len(transcript_gene_mapping)} transcript->gene mappings")
        
        # 2. 读取 eggNOG 注释
        print(f"Reading eggNOG annotations: {args.annotations}")
        df_annot = read_eggnog_annotations(args.annotations)
        print(f"Found {len(df_annot)} annotation records")
        
        # 3. 按 gene_id 聚合
        print("Aggregating annotations by gene_id...")
        df_result = aggregate_annotations_by_gene(df_annot, transcript_gene_mapping)
        print(f"Generated {len(df_result)} gene-level annotations")
        
        # 4. 保存结果（UTF-8 编码，无 BOM）
        print(f"Writing output to: {args.output}")
        df_result.to_csv(
            args.output,
            sep='\t',
            index=False,
            encoding='utf-8',
            na_rep=''
        )
        
        # 调试输出：打印示例 ID 用于验证规范化逻辑
        print("\n[DEBUG] ID normalization examples:")
        print(f"  GTF transcript_id keys (first 5, normalized): {list(transcript_gene_mapping.keys())[:5]}")
        
        raw_queries = df_annot['#query'].head(5).tolist()
        normalized_queries = [extract_transcript_id_from_query(q) for q in raw_queries]
        print(f"  eggNOG #query (first 5, raw): {raw_queries}")
        print(f"  eggNOG #query (first 5, normalized): {normalized_queries}")
        print(f"  Final result: {len(df_result)} gene-level annotations")
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)



if __name__ == '__main__':
    main()

