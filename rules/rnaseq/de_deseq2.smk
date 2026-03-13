# rules/rnaseq/de_deseq2.smk
# -*- coding: utf-8 -*-

# 1) 生成设计矩阵（由脚本完成；不再使用 here-doc）
rule rnaseq_make_design:
    threads: 4
    resources:
        mem_mb=4000,
        runtime=20

    input:
        csv ="config/samples_rnaseq.csv",
        code = "scripts/make_rnaseq_design.py",
    output:
        tsv ="config/rnaseq_design.tsv"
    log:
        "logs/rnaseq/make_design.log"
    shell:
        """
        # 补充 <in.csv> <out.tsv> 参数，匹配脚本的手动运行逻辑
        python {input.code} {input.csv} {output.tsv} > {log} 2>&1
        """

rule rnaseq_deseq2:
    priority: 1000
    resources: mem_mb=8000, runtime=60
    input:
        counts    = "results/rnaseq/counts/featurecounts_hisat2.tsv",
        design    = "config/rnaseq_design.tsv",
        contrasts = "config/contrasts.tsv",
        code      = "scripts/rnaseq/rnaseq_deseq2_contrasts.R",
    output:
        norm = "results/rnaseq/deseq2/normalized_counts.tsv",
        deg  = "results/rnaseq/deseq2/DEG_results.tsv"
    threads: 8
    log: "logs/rnaseq/deseq2.log"
    benchmark: "logs/rnaseq/deseq2.benchmark.txt"
    shell:
        r"""
        mkdir -p results/rnaseq/deseq2 logs/rnaseq
        /public/home/zoujianghua/miniconda3/bin/conda run -n r \
        Rscript {input.code} \
          --counts {input.counts} \
          --design {input.design} \
          --contrasts {input.contrasts} \
          --outdir results/rnaseq/deseq2 \
          --min_per_grp 2 --lfc_th 1 --fdr_th 0.05 \
          > {log} 2>&1
        """


