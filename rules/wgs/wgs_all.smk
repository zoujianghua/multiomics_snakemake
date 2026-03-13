# rules/wgs/wgs_all.smk  (pure ASCII)
include: "ingest_samples.smk"
include: "qc_fastp.smk"
include: "align_bwa.smk"
include: "postbam.smk"
include: "gatk.smk"


FASTQC_RAW = bool(config.get("wgs", {}).get("fastqc_raw", True))
FASTQC_CLEAN = bool(config.get("wgs", {}).get("fastqc_clean", True))

opt_fastqc_raw = (
    expand("results/wgs/qc/fastqc_raw/{sample}", sample=WGS_IDS) if FASTQC_RAW else []
)
opt_fastqc_clean = (
    expand("results/wgs/qc/fastqc_clean/{sample}", sample=WGS_IDS)
    if FASTQC_CLEAN
    else []
)


rule all_wgs:
    input:
        opt_fastqc_raw,
        opt_fastqc_clean,
        # 建议把 fastp 的合并后 clean 明确列为总目标
        expand("results/wgs/clean/{sample}_R1.merged.fq.gz", sample=WGS_IDS),
        expand("results/wgs/clean/{sample}_R2.merged.fq.gz", sample=WGS_IDS),
        # （可选）把 fastp 报告也纳入
        expand("results/wgs/qc/fastp/{sample}.fastp.html", sample=WGS_IDS),
        expand("results/wgs/qc/fastp/{sample}.fastp.json", sample=WGS_IDS),
        "results/wgs/qc/multiqc/multiqc_report.html",
        expand("results/wgs/align_bwa/{sample}.sorted.bam", sample=WGS_IDS),
        expand("results/wgs/align_bwa/{sample}.sorted.bam.bai", sample=WGS_IDS),
        expand("results/wgs/bamstats/{sample}.stats", sample=WGS_IDS),
        expand("results/wgs/bamstats/{sample}.flagstat", sample=WGS_IDS),
        expand("results/wgs/gatk/gvcf/{sample}.g.vcf.gz", sample=WGS_IDS),
        expand("results/wgs/gatk/gvcf/{sample}.g.vcf.gz.tbi", sample=WGS_IDS),
        "results/wgs/gatk/cohort.pass.vcf.gz",
        "results/wgs/gatk/cohort.pass.vcf.gz.tbi",
