# rules/rnaseq/rnaseq_all.smk


include: "ingest_samples.smk"
include: "qc_fastp.smk"
include: "align_hisat2.smk"
include: "quantify_featurecounts.smk"
include: "de_deseq2.smk"
include: "annot_eggnog.smk"
include: "novel_transcripts.smk"

rule all_rnaseq:
    input:
        "config/samples_rnaseq.csv",
        "config/rnaseq_design.tsv",   # ← 新增这一行
        # æææ ·æ¬ç BAMï¼ä¾¿äºäººå·¥æ½æ¥ï¼
        expand("results/rnaseq/align_hisat2/{sample}.sorted.bam", sample=RNASEQ_IDS),
        # è®¡æ°ä¸æ±æ» QC
        "results/rnaseq/counts/featurecounts_hisat2.tsv",
        "results/rnaseq/counts/featurecounts_hisat2.tsv.summary",
        "results/rnaseq/qc/multiqc/multiqc_report.html",
        "results/rnaseq/qc/upstream_qc_summary.tsv",

        "results/rnaseq/eggnog/eggnog_annotations.tsv",


rule rnaseq_eggnog_all:
    """
    eggNOG 注释流程的汇总目标
    包括：蛋白质 FASTA 生成 + eggNOG-mapper 注释 + 结果整理
    """
    input:
        "results/rnaseq/eggnog/eggnog_annotations.tsv",


rule rnaseq_novel_all:
    """
    Novel transcripts/genes 发现与注释流程的汇总目标（不含规模统计表，便于与统计解耦）
    包括：
    1. StringTie 单样本组装 + merge
    2. gffcompare 标注新转录本
    3. 提取并规范化 novel GTF
    4. TransDecoder 预测 novel 蛋白
    5. eggNOG-mapper 注释 novel 蛋白
    6. 合并 reference + novel 的 GTF 和 eggNOG 注释
    7. 基于合并 GTF 重新计数
    规模统计表 novel_discovery_summary.tsv 单独作为目标，见 rnaseq_novel_summary_only。
    """
    input:
        # StringTie 流程
        expand("results/rnaseq/stringtie/{sample}.gtf", sample=RNASEQ_IDS),
        "results/rnaseq/stringtie/merged.gtf",
        "results/rnaseq/stringtie/merged.annotated.gtf",
        # Novel GTF
        "references/annotation/genome_novel.gtf",
        "references/annotation/genes_with_novel.gtf",
        # TransDecoder
        "results/rnaseq/novel/novel_transcripts.pep",
        # Novel eggNOG
        "results/rnaseq/eggnog/novel_eggnog_annotations.tsv",
        # 合并结果
        "results/rnaseq/eggnog/eggnog_annotations_with_novel.tsv",
        # 重新计数
        "results/rnaseq/counts/featurecounts_hisat2_with_novel.tsv",
        "results/rnaseq/counts/featurecounts_hisat2_with_novel.tsv.summary",


# 与“计算”解耦：仅基于已有结果文件生成规模统计表，不触发 TransDecoder/eggNOG 等重跑。
# 上游（novel_gtf、novel_pep、novel_eggnog）已存在且无需更新时，只运行 rnaseq_novel_summary 一步。
rule rnaseq_novel_summary_only:
    """
    仅生成 novel 发现与注释规模汇总表；依赖 genome_novel.gtf、novel_transcripts.pep、novel_eggnog_annotations。
    若上述三文件已存在，仅执行本规则，不触发 StringTie/TransDecoder/eggNOG 等重算。
    """
    input:
        "results/rnaseq/novel/novel_discovery_summary.tsv",
