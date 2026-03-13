# -*- coding: utf-8 -*-
import os

# 是否做 raw / clean 的 FastQC（可按需改 True/False）
FASTQC_RAW = True
FASTQC_CLEAN = True

# 从 ingest 获取样本列表
if "WGS" in globals() and isinstance(WGS, dict) and WGS:
    WGS_IDS = sorted(WGS.keys())
else:
    WGS_IDS = [r["sample_id"] for r in config.get("wgs_samples", [])]


def _as_list(v):
    return v if isinstance(v, (list, tuple)) else [v]


# ---------------- 1) FastQC on RAW（可选；一次可给多个文件） ----------------
if FASTQC_RAW:

    rule wgs_fastqc_raw:
        input:
            r1=lambda wc: _as_list(WGS[wc.sample]["r1"]),
            r2=lambda wc: _as_list(WGS[wc.sample]["r2"]),
        output:
            outdir=directory("results/wgs/qc/fastqc_raw/{sample}"),
        conda:
            "../../envs/wgs_envs/qc.yaml"
        threads: 2
        resources:
            mem_mb=8000,
            runtime=60,
            netio=1,
        log:
            "logs/wgs/fastqc_raw/{sample}.log",
        shell:
            r"""

            export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
            mkdir -p {output.outdir} logs/wgs/fastqc_raw
            fastqc -t {threads} -o {output.outdir} {input.r1} {input.r2} > {log} 2>&1
            """
# ---------------- 2) fastp（安全合并后再跑；输出 *.merged.fq.gz） ----------------


rule wgs_fastp:
    priority: 800
    input:
        r1=lambda wc: _as_list(WGS[wc.sample]["r1"]),
        r2=lambda wc: _as_list(WGS[wc.sample]["r2"]),
    output:
        r1="results/wgs/clean/{sample}_R1.merged.fq.gz",
        r2="results/wgs/clean/{sample}_R2.merged.fq.gz",
        html="results/wgs/qc/fastp/{sample}.fastp.html",
        json="results/wgs/qc/fastp/{sample}.fastp.json",
    conda:
        "../../envs/wgs_envs/qc.yaml"
    threads: 16
    resources:
        mem_mb=32000,
        runtime=600,
        netio=1,
    log:
        "logs/wgs/fastp/{sample}.log"
    shell:
        r"""

        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
        mkdir -p results/wgs/clean results/wgs/qc/fastp logs/wgs/fastp

        # 压缩器：优先 pigz
        if command -v pigz >/dev/null 2>&1; then
          COMP="pigz -p {threads} -c"
        else
          COMP="gzip -c"
        fi

        # 如果 R1/R2 输入有多个文件则合并，否则直接用原文件
        if [ $(echo {input.r1} | wc -w) -gt 1 ]; then
          zcat {input.r1} | eval "$COMP" > "{output.r1}"
          IN1="{output.r1}"
        else
          IN1={input.r1}
        fi
        if [ $(echo {input.r2} | wc -w) -gt 1 ]; then
          zcat {input.r2} | eval "$COMP" > "{output.r2}"
          IN2="{output.r2}"
        else
          IN2={input.r2}
        fi

        # 直接在结果目录运行 fastp，输出覆盖目标文件
        fastp -i "$IN1" -I "$IN2" \
              -o "{output.r1}" -O "{output.r2}" \
              -w {threads} \
              -q 20 -u 30 -n 5 -l 25 --detect_adapter_for_pe \
              -h "{output.html}" -j "{output.json}" \
              > {log} 2>&1
        """




# ---------------- 3) FastQC on CLEAN（可选；对 merged 输出做 QC） ----------------
if FASTQC_CLEAN:

    rule wgs_fastqc_clean:
        input:
            r1="results/wgs/clean/{sample}_R1.merged.fq.gz",
            r2="results/wgs/clean/{sample}_R2.merged.fq.gz",
        output:
            outdir=directory("results/wgs/qc/fastqc_clean/{sample}"),
        conda:
            "../../envs/wgs_envs/qc.yaml"
        threads: 2
        resources:
            mem_mb=8000,
            runtime=60,
            netio=2,
        log:
            "logs/wgs/fastqc_clean/{sample}.log",
        shell:
            r"""
            set -euo pipefail
            export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
            mkdir -p {output.outdir} logs/wgs/fastqc_clean
            fastqc -t {threads} -o {output.outdir} {input.r1} {input.r2} > {log} 2>&1
            """

# ---------------- 4) MultiQC ----------------



rule wgs_multiqc:
    input:
        expand("results/wgs/qc/fastp/{sample}.fastp.json", sample=WGS_IDS),
        expand("results/wgs/qc/fastp/{sample}.fastp.html", sample=WGS_IDS),
        expand("results/wgs/qc/fastqc_clean/{sample}", sample=WGS_IDS)
        if FASTQC_CLEAN
        else [],
        expand("results/wgs/qc/fastqc_raw/{sample}", sample=WGS_IDS)
        if FASTQC_RAW
        else [],
    output:
        html="results/wgs/qc/multiqc/multiqc_report.html",
    conda:
        "../../envs/wgs_envs/qc.yaml"
    threads: 2
    resources:
        mem_mb=4000,
        runtime=30,
        netio=1,
    log:
        "logs/wgs/multiqc/multiqc.log",
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/wgs/qc/multiqc logs/wgs/multiqc
        # 仅扫描 WGS 相关结果与日志，避免与 RNA-seq 等其它模块的 QC 混在一起
        multiqc results/wgs logs/wgs \
          -o results/wgs/qc/multiqc \
          -n multiqc_report.html -f > {log} 2>&1
        """
