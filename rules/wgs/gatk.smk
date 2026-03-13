# -*- coding: utf-8 -*-

import os

REF_FASTA = config["references"]["fasta"]
REF_FAI = REF_FASTA + ".fai"
REF_DICT = os.path.splitext(REF_FASTA)[0] + ".dict"


rule wgs_markdup_gatk:
    priority: 1
    input:
        bam="results/wgs/align_bwa/{sample}.sorted.bam",
        bai="results/wgs/align_bwa/{sample}.sorted.bam.bai",
    output:
        bam="results/wgs/gatk/{sample}.markdup.bam",
        bai="results/wgs/gatk/{sample}.markdup.bam.bai",
        metrics="results/wgs/gatk/{sample}.markdup.metrics.txt",
    conda:
        "../../envs/wgs_envs/gatk.yaml"
    threads: 16
    resources:
        mem_mb=64000, runtime=72000, netio=5,
    params:
        validation="LENIENT",
    log:
        "logs/wgs/gatk/markdup_{sample}.log",
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/wgs/gatk logs/wgs/gatk

        # 选本地临时目录
        SCR_CAND=()
        [[ -n "${{SLURM_TMPDIR:-}}" ]] && SCR_CAND+=("${{SLURM_TMPDIR}}")
        [[ -n "${{SLURM_JOB_ID:-}}" && -d "/scratch/${{USER}}/${{SLURM_JOB_ID}}" ]] && SCR_CAND+=("/scratch/${{USER}}/${{SLURM_JOB_ID}}")
        SCR_CAND+=("${{TMPDIR:-/tmp}}")
        SCR=""
        for d in "${{SCR_CAND[@]}}"; do
          if mkdir -p "$d/smk_{wildcards.sample}" 2>/dev/null; then
            SCR="$d/smk_{wildcards.sample}"
            break
          fi
        done
        trap 'rm -rf "$SCR"' EXIT

        # Java 堆大小按 mem_mb 的 85%
        jmx=$(( {resources.mem_mb} * 85 / 100 ))

        BAM_IN="{input.bam}"
        BAM_OUT="{output.bam}"
        MET_OUT="{output.metrics}"
        LOG="{log}"

        BAM_TMP="$SCR/{wildcards.sample}.markdup.tmp.bam"
        MET_TMP="$SCR/{wildcards.sample}.markdup.tmp.metrics.txt"

        gatk --java-options "-Djava.io.tmpdir=$SCR -Xms2g -Xmx${{jmx}}m" MarkDuplicates \
          -I "$BAM_IN" -O "$BAM_TMP" -M "$MET_TMP" \
          --ASSUME_SORT_ORDER coordinate \
          --VALIDATION_STRINGENCY {params.validation} \
          --CREATE_INDEX true \
          2> "$LOG"

        mv -f "$BAM_TMP" "$BAM_OUT"
        mv -f "$MET_TMP" "$MET_OUT"

        # 用 samtools 统一建索引
        samtools index -@ {threads} "$BAM_OUT" 2>> "$LOG"

        # 健康检查
        if [[ ! -s "$BAM_OUT.bai" ]]; then
          echo "[ERR] index missing for $BAM_OUT" >> "$LOG"
          exit 1
        fi
        """




rule wgs_haplotypecaller_gvcf:
    priority: 1
    input:
        bam="results/wgs/gatk/{sample}.markdup.bam",
        bai="results/wgs/gatk/{sample}.markdup.bam.bai",
        ref=REF_FASTA,
        fai=REF_FAI,
        dict=REF_DICT,
    output:
        gvcf="results/wgs/gatk/gvcf/{sample}.g.vcf.gz",
        gvcf_tbi="results/wgs/gatk/gvcf/{sample}.g.vcf.gz.tbi",
    conda:
        "../../envs/wgs_envs/gatk.yaml"
    threads: 32
    resources:
        mem_mb=64000,
        runtime=72000,
        netio=5,
    params:
        # 保留接口；内部使用本地临时盘
        tmp="results/tmp/gatk/hc/{sample}",
        extra=config.get("gatk", {}).get("hc_extra", ""),
    log:
        "logs/wgs/gatk/HC_{sample}.log",
    shell:
        r"""
    set -euo pipefail
    mkdir -p results/wgs/gatk/gvcf logs/wgs/gatk {params.tmp}

    # 选本地临时目录（优先 SLURM_TMPDIR，其次 /scratch/$USER/$SLURM_JOB_ID，最后 TMPDIR 或 /tmp）
    SCR=""
    for d in "${{SLURM_TMPDIR:-}}" "/scratch/${{USER:-}}/${{SLURM_JOB_ID:-}}" "${{TMPDIR:-/tmp}}"; do
      [ -z "$d" ] && continue
      if mkdir -p "$d/smk_{wildcards.sample}" 2>/dev/null; then
        SCR="$d/smk_{wildcards.sample}"
        break
      fi
    done
    [ -n "$SCR" ] || SCR="${{TMPDIR:-/tmp}}/smk_{wildcards.sample}"
    trap 'rm -rf "$SCR"' EXIT

    # 尝试把 BAM/BAI 拷到本地（空间够才拷），否则直接从共享盘读
    BAM_IN="{input.bam}"
    BAI_IN="{input.bai}"
    BAM_LOCAL="$BAM_IN"

    if command -v stat >/dev/null 2>&1; then
      SZ=$(stat -c %s "$BAM_IN" 2>/dev/null || echo 0)
    else
      SZ=0
    fi
    FREE=$(df -B1 "$SCR" 2>/dev/null | awk 'NR==2{{print $4}}')
    NEED=$(( SZ + SZ/2 ))
    if [ "$SZ" -gt 0 ] && [ -n "${{FREE:-0}}" ] && [ "$FREE" -gt "$NEED" ]; then
      cp -f --reflink=auto "$BAM_IN" "$SCR/in.bam" 2>> {log} || cp -f "$BAM_IN" "$SCR/in.bam" 2>> {log}
      cp -f --reflink=auto "$BAI_IN" "$SCR/in.bam.bai" 2>> {log} || cp -f "$BAI_IN" "$SCR/in.bam.bai" 2>> {log} || true
      BAM_LOCAL="$SCR/in.bam"
    fi

    # 线程与内存（适度保守更稳）
    PAIR_THREADS=$(( {threads} > 8 ? 8 : {threads} ))
    export OMP_NUM_THREADS=1
    jmx=$(( {resources.mem_mb} * 90 / 100 ))

    GVCF_TMP="$SCR/{wildcards.sample}.g.tmp.vcf.gz"

    gatk --java-options "-Djava.io.tmpdir=$SCR -Xms2g -Xmx${{jmx}}m" HaplotypeCaller \
      -R {input.ref} \
      -I "$BAM_LOCAL" \
      -O "$GVCF_TMP" \
      -ERC GVCF \
      --native-pair-hmm-threads $PAIR_THREADS \
      {params.extra} \
      2> {log}

    # 索引与回写
    if [ ! -s "$GVCF_TMP.tbi" ]; then
      gatk IndexFeatureFile -I "$GVCF_TMP" 2>> {log} || tabix -p vcf "$GVCF_TMP" 2>> {log}
    fi
    mv -f "$GVCF_TMP"     "{output.gvcf}"
    mv -f "$GVCF_TMP.tbi" "{output.gvcf_tbi}"
    """


rule wgs_genotypegvcfs:
    priority: 1
    input:
        ref=REF_FASTA,
        fai=REF_FAI,
        dict=REF_DICT,
        gvcfs=expand("results/wgs/gatk/gvcf/{sample}.g.vcf.gz", sample=WGS_IDS),
    output:
        vcf="results/wgs/gatk/cohort.raw.vcf.gz",
        vcf_tbi="results/wgs/gatk/cohort.raw.vcf.gz.tbi",
    conda:
        "../../envs/wgs_envs/gatk.yaml"
    threads: 32
    resources:
        mem_mb=64000,
        runtime=72000,
        netio=5,
    params:
        # 保留接口；内部使用本地临时盘
        tmp="results/tmp/gatk/genotype/cohort",
        extra=config.get("gatk", {}).get("genotype_extra", ""),
    log:
        "logs/wgs/gatk/GenotypeGVCFs.log",
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/wgs/gatk logs/wgs/gatk {params.tmp}

        # 本地临时目录选择
        SCR_CAND=()
        [ -n "${{SLURM_TMPDIR:-}}" ] && SCR_CAND+=("${{SLURM_TMPDIR}}")
        [ -n "${{SLURM_JOB_ID:-}}" ] && [ -d "/scratch/${{USER}}/${{SLURM_JOB_ID}}" ] && SCR_CAND+=("/scratch/${{USER}}/${{SLURM_JOB_ID}}")
        SCR_CAND+=("${{TMPDIR:-/tmp}}")
        for d in "${{SCR_CAND[@]}}"; do
          if mkdir -p "$d/smk_genotype" 2>/dev/null; then
            SCR="$d/smk_genotype"
            break
          fi
        done
        trap 'rm -rf "$SCR"' EXIT

        jmx=$(( {resources.mem_mb} * 85 / 100 ))
        VLIST=$(printf " -V %s" {input.gvcfs})

        # 本地输出，再回写
        VCF_TMP="$SCR/cohort.raw.tmp.vcf.gz"

        gatk --java-options "-Djava.io.tmpdir=$SCR -Xms2g -Xmx${{jmx}}m" GenotypeGVCFs \
          -R {input.ref} \
          $VLIST \
          -O "$VCF_TMP" \
          {params.extra} \
          2> {log}

        # 索引在本地完成
        if [ ! -s "$VCF_TMP.tbi" ]; then
          gatk IndexFeatureFile -I "$VCF_TMP" 2>> {log} || tabix -p vcf "$VCF_TMP" 2>> {log}
        fi

        # 回写到既定产物路径
        mv -f "$VCF_TMP" "{output.vcf}"
        mv -f "$VCF_TMP.tbi" "{output.vcf_tbi}"
        """


SNP_EXPR = config["wgs"]["variant_filters"]["snp"]
INDEL_EXPR = config["wgs"]["variant_filters"]["indel"]


rule wgs_variant_filtration:
    priority: 1
    input:
        ref=REF_FASTA,
        fai=REF_FAI,
        dict=REF_DICT,
        vcf="results/wgs/gatk/cohort.raw.vcf.gz",
    output:
        snp="results/wgs/gatk/cohort.snp.filtered.vcf.gz",
        indel="results/wgs/gatk/cohort.indel.filtered.vcf.gz",
        merged="results/wgs/gatk/cohort.filtered.vcf.gz",
    conda:
        "../../envs/wgs_envs/gatk.yaml"
    threads:16
    resources:
        mem_mb=32000,
        runtime=60000,
        netio=5,
    params:
        # 保留接口；内部使用本地临时盘
        tmp="results/tmp/gatk/varflt/cohort",
    log:
        "logs/wgs/gatk/VariantFiltration.log",
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/wgs/gatk logs/wgs/gatk {params.tmp}

        # 本地临时目录选择
        SCR_CAND=()
        [ -n "${{SLURM_TMPDIR:-}}" ] && SCR_CAND+=("${{SLURM_TMPDIR}}")
        [ -n "${{SLURM_JOB_ID:-}}" ] && [ -d "/scratch/${{USER}}/${{SLURM_JOB_ID}}" ] && SCR_CAND+=("/scratch/${{USER}}/${{SLURM_JOB_ID}}")
        SCR_CAND+=("${{TMPDIR:-/tmp}}")
        for d in "${{SCR_CAND[@]}}"; do
          if mkdir -p "$d/smk_varflt" 2>/dev/null; then
            SCR="$d/smk_varflt"
            break
          fi
        done
        trap 'rm -rf "$SCR"' EXIT

        # 在本地做子集与过滤
        SNP_TMP="$SCR/tmp.snp.vcf.gz"
        INDEL_TMP="$SCR/tmp.indel.vcf.gz"

        gatk SelectVariants -R {input.ref} -V {input.vcf} --select-type-to-include SNP   -O "$SNP_TMP" 2> {log}
        gatk SelectVariants -R {input.ref} -V {input.vcf} --select-type-to-include INDEL -O "$INDEL_TMP" 2>> {log}

        gatk VariantFiltration -R {input.ref} -V "$SNP_TMP" \
          --filter-expression "{SNP_EXPR}"   --filter-name "SNP_filters"   -O "{output.snp}" 2>> {log}

        gatk VariantFiltration -R {input.ref} -V "$INDEL_TMP" \
          --filter-expression "{INDEL_EXPR}" --filter-name "INDEL_filters" -O "{output.indel}" 2>> {log}

        gatk MergeVcfs -I "{output.snp}" -I "{output.indel}" -O "{output.merged}" 2>> {log} \
          || bcftools concat -a -O z -o "{output.merged}" "{output.snp}" "{output.indel}" 2>> {log}

        # 清理本地临时
        rm -f "$SNP_TMP" "$SNP_TMP.tbi" "$INDEL_TMP" "$INDEL_TMP.tbi" || true
        """


rule wgs_select_pass:
    priority: 1
    input:
        vcf = "results/wgs/gatk/cohort.filtered.vcf.gz"
    output:
        vcf = "results/wgs/gatk/cohort.pass.vcf.gz",
        tbi = "results/wgs/gatk/cohort.pass.vcf.gz.tbi"
    conda:
        "../../envs/wgs_envs/gatk.yaml"
    threads: 16
    resources:
        mem_mb = 32000,
        runtime=60000,
        netio=5,
    log: "logs/wgs/gatk/SelectPASS.log"
    shell:
        r"""
        set -euo pipefail
        # 轻 I/O，直接在共享盘操作即可；如需也可按同样模式加 SCR
        gatk SelectVariants -V {input.vcf} --exclude-filtered true -O {output.vcf} 2> {log}
        # 建立索引（优先 GATK，失败则用 tabix）
        if [ ! -s {output.tbi} ]; then
          gatk IndexFeatureFile -I {output.vcf} 2>> {log} || tabix -p vcf {output.vcf} 2>> {log}
        fi
        """

