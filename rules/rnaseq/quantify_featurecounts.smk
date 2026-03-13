# rules/rnaseq/quantify_featurecounts.smk


rule rnaseq_featurecounts_hisat2:
    input:
        gtf=config["references"]["gtf"],
        bams=expand(
            "results/rnaseq/align_hisat2/{sample}.sorted.bam", sample=RNASEQ_IDS
        ),
    output:
        tsv="results/rnaseq/counts/featurecounts_hisat2.tsv",
        summary="results/rnaseq/counts/featurecounts_hisat2.tsv.summary",
    threads: 8
    resources:
        mem_mb=32000,
        runtime=240,
    conda:
        "../../envs/rnaseq_envs/count_featurecounts.yaml"
    params:
        stranded=lambda wc: str(config["rnaseq"]["stranded"]),
        extra=lambda wc: config["rnaseq"].get("featurecounts_extra", ""),
    log:
        "logs/rnaseq/featurecounts/hisat2.log",
    shell:
        r"""
        mkdir -p results/rnaseq/counts logs/rnaseq/featurecounts
        featureCounts -T {threads} -a {input.gtf} -t exon -g gene_id \
                      -p -B -C -s {params.stranded} -Q 10 {params.extra} \
                      -o {output.tsv} {input.bams} > {log} 2>&1
        # 注意：featureCounts 会自动生成 {output.tsv}.summary，我们已在 output: 显式声明为 {output.summary}
        """


############################################
# Novel gene 重新计数（使用 genes_with_novel.gtf）
############################################
rule rnaseq_featurecounts_hisat2_with_novel:
    """
    使用包含 novel genes 的合并 GTF 重新计数
    输入 BAM 与原规则完全一致，唯一区别是使用 genes_with_novel.gtf
    """
    input:
        gtf="references/annotation/genes_with_novel.gtf",
        bams=expand(
            "results/rnaseq/align_hisat2/{sample}.sorted.bam", sample=RNASEQ_IDS
        ),
    output:
        tsv="results/rnaseq/counts/featurecounts_hisat2_with_novel.tsv",
        summary="results/rnaseq/counts/featurecounts_hisat2_with_novel.tsv.summary",
    threads: 8
    resources:
        mem_mb=32000,
        runtime=240,
    conda:
        "../../envs/rnaseq_envs/count_featurecounts.yaml"
    params:
        stranded=lambda wc: str(config["rnaseq"]["stranded"]),
        extra=lambda wc: config["rnaseq"].get("featurecounts_extra", ""),
    log:
        "logs/rnaseq/featurecounts/hisat2_with_novel.log",
    shell:
        r"""
        mkdir -p results/rnaseq/counts logs/rnaseq/featurecounts
        featureCounts -T {threads} -a {input.gtf} -t exon -g gene_id \
                      -p -B -C -s {params.stranded} -Q 10 {params.extra} \
                      -o {output.tsv} {input.bams} > {log} 2>&1
        # 注意：featureCounts 会自动生成 {output.tsv}.summary，我们已在 output: 显式声明为 {output.summary}
        """
