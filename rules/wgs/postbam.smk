# -*- coding: utf-8 -*-

# Post-BAM QC: samtools stats / flagstat

rule wgs_samtools_stats:
    input:
        bam="results/wgs/align_bwa/{sample}.sorted.bam",
    output:
        stats="results/wgs/bamstats/{sample}.stats",
    conda:
        "../../envs/wgs_envs/postbam.yaml"
    threads: 1
    resources:
        mem_mb=8000,
        runtime=600,
        netio=1,
    log:
        "logs/wgs/postbam/{sample}.samtools_stats.log",
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/wgs/bamstats logs/wgs/postbam

        # 选择本地临时目录（有则用，无则退）：SLURM_TMPDIR -> /scratch/$USER/$SLURM_JOB_ID -> TMPDIR -> /tmp
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

        # 自适应：若本地盘空间足够（文件大小的 1.5 倍余量），则拷到本地跑；否则直接用共享盘文件
        BAM_IN="{input.bam}"
        BAM_LOCAL="$BAM_IN"
        if command -v stat >/dev/null 2>&1; then
          SZ=$(stat -c %s "$BAM_IN" 2>/dev/null || echo 0)
        else
          SZ=0
        fi
        FREE=$(df -B1 "$SCR" 2>/dev/null | awk 'NR==2{{print $4}}')
        NEED=$(( SZ + SZ / 2 ))  # 1.5x 头寸
        if [ "$SZ" -gt 0 ] && [ -n "$FREE" ] && [ "$FREE" -gt "$NEED" ]; then
          cp -f --reflink=auto "$BAM_IN" "$SCR/in.bam" 2>> {log} || cp -f "$BAM_IN" "$SCR/in.bam" 2>> {log}
          BAM_LOCAL="$SCR/in.bam"
        fi

        # 运行 stats（参数与输出不变）
        samtools stats "$BAM_LOCAL" > {output.stats} 2> {log}
        """


# rules/wgs/postbam.smk
rule wgs_flagstat:
    input:
        bam = "results/wgs/align_bwa/{sample}.sorted.bam",
        bai = "results/wgs/align_bwa/{sample}.sorted.bam.bai"   # 强化依赖
    output:
        flag = "results/wgs/bamstats/{sample}.flagstat"
    conda:
        "../../envs/wgs_envs/postbam.yaml"
    log:
        "logs/wgs/postbam/{sample}.flagstat.log"
    threads: 1
    resources:
        mem_mb = 8000,
        runtime = 600,
        netio=1,
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/wgs/bamstats logs/wgs/postbam

        # 选择本地临时目录
        SCR_CAND=()
        [[ -n "${{SLURM_TMPDIR:-}}" ]] && SCR_CAND+=("${{SLURM_TMPDIR}}")
        [[ -n "${{SLURM_JOB_ID:-}}" ]] && [[ -d "/scratch/${{USER}}/${{SLURM_JOB_ID}}" ]] && SCR_CAND+=("/scratch/${{USER}}/${{SLURM_JOB_ID}}")
        SCR_CAND+=("${{TMPDIR:-/tmp}}")

        SCR=""
        for d in "${{SCR_CAND[@]}}"; do
          if mkdir -p "$d/smk_{wildcards.sample}" 2>/dev/null; then
            SCR="$d/smk_{wildcards.sample}"
            break
          fi
        done
        trap 'rm -rf "$SCR"' EXIT

        BAM_IN="{input.bam}"
        BAM_LOCAL="$BAM_IN"

        # 仅当本地空间足够时才复制
        if command -v stat >/dev/null 2>&1; then
          SZ=$(stat -c %s "$BAM_IN" 2>/dev/null || echo 0)
        else
          SZ=0
        fi
        if [[ -n "$SCR" ]]; then
          FREE=$(df -B1 "$SCR" 2>/dev/null | awk 'NR==2{{print $4}}')
        else
          FREE=0
        fi
        NEED=$(( SZ + SZ / 2 ))
        if [[ "$SZ" -gt 0 && -n "$FREE" && "$FREE" -gt "$NEED" ]]; then
          cp -f --reflink=auto "$BAM_IN" "$SCR/in.bam" 2>> {log} || cp -f "$BAM_IN" "$SCR/in.bam" 2>> {log}
          BAM_LOCAL="$SCR/in.bam"
        fi

        # 运行 flagstat
        samtools flagstat "$BAM_LOCAL" > {output.flag} 2>> {log}
        """

