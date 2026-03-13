# -*- coding: utf-8 -*-

BWA_IDX_FILES = expand(
    "{ref}.{ext}",
    ref=config["references"]["fasta"],
    ext=["amb", "ann", "bwt", "pac", "sa"],
)


rule wgs_align_bwa:
    input:
        idx=BWA_IDX_FILES,
        r1q="results/wgs/clean/{sample}_R1.merged.fq.gz",
        r2q="results/wgs/clean/{sample}_R2.merged.fq.gz",
        ref=config["references"]["fasta"],
    output:
        bam="results/wgs/align_bwa/{sample}.sorted.bam",
        bai="results/wgs/align_bwa/{sample}.sorted.bam.bai",
    conda:
        "../../envs/wgs_envs/align_bwa.yaml"
    threads: 32
    resources:
        mem_mb=48000,
        runtime=600,
        netio=1,
    params:
        rgpl="ILLUMINA",
        rglb="lib1",
        rgpu=lambda wc: f"{wc.sample}.1",
        extra=config.get("align", {}).get("bwa_extra", ""),
    log:
        "logs/wgs/bwa/{sample}.log",
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/wgs/align_bwa logs/wgs/bwa

        # 选择本地临时目录：优先 SLURM_TMPDIR，其次 /scratch/$USER/$SLURM_JOB_ID（若存在），否则 TMPDIR 或 /tmp
        SCR_CAND=()
        [ -n "${{SLURM_TMPDIR:-}}" ] && SCR_CAND+=("${{SLURM_TMPDIR}}")
        [ -n "${{SLURM_JOB_ID:-}}" ] && [ -d "/scratch/${{USER}}/${{SLURM_JOB_ID}}" ] && SCR_CAND+=("/scratch/${{USER}}/${{SLURM_JOB_ID}}")
        SCR_CAND+=("${{TMPDIR:-/tmp}}")
        for d in "${{SCR_CAND[@]}}"; do
          if mkdir -p "$d/smk_{wildcards.sample}" 2>/dev/null; then
            SCR="$d/smk_{wildcards.sample}"
            break
          fi
        done
        trap 'rm -rf "$SCR"' EXIT

        RG="@RG\tID:{wildcards.sample}\tSM:{wildcards.sample}\tLB:{params.rglb}\tPL:{params.rgpl}\tPU:{params.rgpu}"

        # 在本地盘完成排序与建索引，再原子回写到结果目录
        BAM_TMP="$SCR/{wildcards.sample}.sorted.tmp.bam"

        # 对 sort 指定 -T 本地临时前缀，避免在共享盘上创建海量临时块
        bwa mem -t {threads} -R "${{RG}}" {params.extra} {input.ref} {input.r1q} {input.r2q} 2> {log} \
          | samtools sort -@ {threads} -T "$SCR/sorttmp" -o "$BAM_TMP" - 2>> {log}

        samtools index -@ {threads} "$BAM_TMP" 2>> {log}

        # 回写到既定产物路径
        mv -f "$BAM_TMP" "{output.bam}"
        mv -f "$BAM_TMP.bai" "{output.bai}"

        # 基本健康检查（产物头部应包含 @RG）
        if ! samtools view -H "{output.bam}" | grep -q '^@RG'; then
            echo "[FATAL] No @RG found in {output.bam}" >> {log}
            exit 1
        fi
        """

