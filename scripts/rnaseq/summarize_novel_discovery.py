#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总 novel transcripts/genes 发现与注释规模：

输入：
  --novel-gtf         references/annotation/genome_novel.gtf
  --novel-pep         results/rnaseq/novel/novel_transcripts.pep
  --novel-eggnog      results/rnaseq/eggnog/novel_eggnog_annotations.tsv

输出：
  --output            results/rnaseq/novel/novel_discovery_summary.tsv

表格列示例：
  n_novel_genes
  n_novel_transcripts
  n_transcripts_with_orf
  n_novel_eggnog_annotated
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def count_genes_transcripts(gtf_path: Path) -> tuple[int, int]:
    """
    从 novel GTF 中统计基因数与转录本数。
    依赖 gene_id / transcript_id 属性。
    """
    gene_ids = set()
    transcript_ids = set()

    with gtf_path.open() as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            attrs = fields[8]
            g_match = re.search(r'gene_id\s+"([^"]+)"', attrs)
            t_match = re.search(r'transcript_id\s+"([^"]+)"', attrs)
            if g_match:
                gene_ids.add(g_match.group(1))
            if t_match:
                transcript_ids.add(t_match.group(1))

    return len(gene_ids), len(transcript_ids)


def count_fasta_seqs(fa_path: Path) -> int:
    """
    统计 FASTA 中的序列条数（以 '>' 行计）。
    """
    if not fa_path.is_file() or fa_path.stat().st_size == 0:
        return 0
    n = 0
    with fa_path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                n += 1
    return n


def count_eggnog_genes(eggnog_tsv: Path) -> int:
    """
    统计 novel eggNOG 注释中具有注释的 gene 数量。
    直接按 gene_id 去重计数。
    """
    if not eggnog_tsv.is_file() or eggnog_tsv.stat().st_size == 0:
        return 0
    df = pd.read_csv(eggnog_tsv, sep="\t")
    if "gene_id" not in df.columns:
        return 0
    return df["gene_id"].nunique()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--novel-gtf",
        required=True,
        help="novel GTF（references/annotation/genome_novel.gtf）",
    )
    ap.add_argument(
        "--novel-pep",
        required=True,
        help="TransDecoder 预测的 novel 蛋白 FASTA（novel_transcripts.pep）",
    )
    ap.add_argument(
        "--novel-eggnog",
        required=True,
        help="整理后的 novel eggNOG 注释表（novel_eggnog_annotations.tsv）",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="输出 TSV 路径（novel_discovery_summary.tsv）",
    )
    args = ap.parse_args()

    novel_gtf = Path(args.novel_gtf)
    novel_pep = Path(args.novel_pep)
    novel_eggnog = Path(args.novel_eggnog)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_genes, n_transcripts = count_genes_transcripts(novel_gtf)
    n_orf = count_fasta_seqs(novel_pep)
    n_eggnog = count_eggnog_genes(novel_eggnog)

    summary = pd.DataFrame(
        [
            {
                "n_novel_genes": n_genes,
                "n_novel_transcripts": n_transcripts,
                "n_transcripts_with_orf": n_orf,
                "n_novel_eggnog_annotated": n_eggnog,
            }
        ]
    )
    summary.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main()

