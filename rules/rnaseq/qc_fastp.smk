# rules/rnaseq/qc_fastp.smk
# 作用：
#   1) FastQC（原始 FASTQ）
#   2) fastp（修剪 + 报告）
#   3) FastQC（修剪后 FASTQ）
#   4) MultiQC 汇总 fastp / FastQC / HISAT2 / featureCounts
#
# 依赖：
#   - RNASEQ / RNASEQ_IDS 来自 ingest_samples.smk
#   - HISAT2 规则需把 stderr 到 logs/rnaseq/hisat2/{sample}.log
#   - featureCounts 产出 results/rnaseq/counts/featurecounts_hisat2.tsv.summary


############################################
# 1) FastQC（原始 FASTQ）——目录型输出
############################################
rule rnaseq_fastqc_raw:
    input:
        r1=lambda wc: RNASEQ[wc.sample]["r1"],
        r2=lambda wc: RNASEQ[wc.sample]["r2"],
    output:
        outdir=directory("results/rnaseq/qc/fastqc_raw/{sample}"),
    conda:
        "../../envs/rnaseq_envs/qc.yaml"
    threads: 4
    resources:
        mem_mb=8000,
        runtime=60,
    log:
        "logs/rnaseq/fastqc_raw/{sample}.log",
    shell:
        r"""
        set -euo pipefail
        mkdir -p {output.outdir} logs/rnaseq/fastqc_raw
        fastqc -t {threads} -o {output.outdir} {input.r1} {input.r2} > {log} 2>&1
        """


############################################
# 2) fastp（修剪 + 报告）
############################################
rule rnaseq_fastp:
    input:
        r1=lambda wc: RNASEQ[wc.sample]["r1"],
        r2=lambda wc: RNASEQ[wc.sample]["r2"],
    output:
        # 修剪后 FASTQ（供对齐用）
        r1="results/rnaseq/clean/{sample}_R1.fastq.gz",
        r2="results/rnaseq/clean/{sample}_R2.fastq.gz",
        # fastp 报告
        html="results/rnaseq/qc/fastp/{sample}.fastp.html",
        json="results/rnaseq/qc/fastp/{sample}.fastp.json",
    conda:
        "../../envs/rnaseq_envs/qc.yaml"
    threads: 8
    resources:
        mem_mb=16000,
        runtime=60,
    log:
        "logs/rnaseq/fastp/{sample}.log",
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/rnaseq/clean results/rnaseq/qc/fastp logs/rnaseq/fastp
        fastp -i {input.r1} -I {input.r2} \
              -o {output.r1} -O {output.r2} \
              -w {threads} \
              -q 20 -u 30 -n 5 -l 25 --detect_adapter_for_pe \
              -h {output.html} -j {output.json} \
              > {log} 2>&1
        """


############################################
# 3) FastQC（修剪后 FASTQ）——目录型输出
############################################
rule rnaseq_fastqc_clean:
    input:
        r1="results/rnaseq/clean/{sample}_R1.fastq.gz",
        r2="results/rnaseq/clean/{sample}_R2.fastq.gz",
    output:
        outdir=directory("results/rnaseq/qc/fastqc_clean/{sample}"),
    conda:
        "../../envs/rnaseq_envs/qc.yaml"
    threads: 4
    resources:
        mem_mb=8000,
        runtime=60,
    log:
        "logs/rnaseq/fastqc_clean/{sample}.log",
    shell:
        r"""
        set -euo pipefail
        mkdir -p {output.outdir} logs/rnaseq/fastqc_clean
        fastqc -t {threads} -o {output.outdir} {input.r1} {input.r2} > {log} 2>&1
        """


############################################
# 4) MultiQC（汇总 fastp / FastQC / HISAT2 / featureCounts）
# 与重测序（WGS）完全解耦：仅依赖本模块的 results/rnaseq 与 logs/rnaseq，
# 不依赖任何 results/wgs 或 logs/wgs。运行后生成 multiqc_report.html 及
# multiqc_data/（内含 multiqc_general_stats.txt 等），供 rnaseq_qc_summary 使用。
############################################
rule rnaseq_multiqc:
    input:
        # fastp 报告（确保执行 fastp）
        expand("results/rnaseq/qc/fastp/{sample}.fastp.json", sample=RNASEQ_IDS),
        expand("results/rnaseq/qc/fastp/{sample}.fastp.html", sample=RNASEQ_IDS),
        # FastQC 的目录（确保执行 raw/clean 两次 FastQC）
        expand("results/rnaseq/qc/fastqc_raw/{sample}", sample=RNASEQ_IDS),
        expand("results/rnaseq/qc/fastqc_clean/{sample}", sample=RNASEQ_IDS),
        # HISAT2 日志（请在比对规则里 2> logs/rnaseq/hisat2/{sample}.log）
        expand("logs/rnaseq/hisat2/{sample}.log", sample=RNASEQ_IDS),
        # featureCounts 的 summary（单个文件）
        "results/rnaseq/counts/featurecounts_hisat2.tsv.summary",
    output:
        html="results/rnaseq/qc/multiqc/multiqc_report.html",
    conda:
        "../../envs/rnaseq_envs/qc.yaml"
    threads: 16
    resources:
        mem_mb=32000,
        runtime=600,
    log:
        "logs/rnaseq/multiqc/multiqc.log",
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/rnaseq/qc/multiqc logs/rnaseq/multiqc
        # 仅扫描 RNA-seq 相关结果与日志，避免与 WGS 等其它模块的 QC 混在一起
        multiqc results/rnaseq logs/rnaseq \
          -o results/rnaseq/qc/multiqc \
          -n multiqc_report.html -f > {log} 2>&1
        """


############################################
# 5) 上游 QC 汇总表（整合 MultiQC + featureCounts summary）
############################################
rule rnaseq_qc_summary:
    input:
        multiqc_html="results/rnaseq/qc/multiqc/multiqc_report.html",
        fc_summary="results/rnaseq/counts/featurecounts_hisat2.tsv.summary",
        code="scripts/rnaseq/summarize_rnaseq_upstream_qc.py",
    output:
        tsv="results/rnaseq/qc/upstream_qc_summary.tsv",
    conda:
        "../../envs/hsi_env.yaml"
    threads: 1
    resources:
        mem_mb=2000,
        runtime=30,
    log:
        "logs/rnaseq/qc/upstream_qc_summary.log",
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/rnaseq/qc logs/rnaseq/qc
        python {input.code} \
          --multiqc-data results/rnaseq/qc/multiqc/multiqc_data \
          --featurecounts-summary {input.fc_summary} \
          --output {output.tsv} \
          > {log} 2>&1
        """
