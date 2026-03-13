# rules/rnaseq/novel_transcripts.smk
# Novel transcripts / genes 发现与注释流程
#
# 流程：
#   1. StringTie 单样本组装（reference-guided）
#   2. StringTie merge 合并所有样本的转录本
#   3. gffcompare 标注新转录本（与参考 GTF 比较）
#   4. 提取 novel transcripts/genes 并规范化
#   5. gffread 提取 novel transcripts 序列
#   6. TransDecoder 预测 novel 蛋白
#   7. eggNOG-mapper 注释 novel 蛋白
#   8. 合并 reference + novel 的 eggNOG 注释
#   9. 合并 reference + novel 的 GTF
#   10. 基于合并 GTF 重新计数

# 从 config 读取 novel 相关配置
NOVEL_CLASS_CODES = config.get("rnaseq", {}).get("novel_class_codes", ["u"])
# u: 完全落在基因间区的全新转录本
# x: 反义链上的转录本
# i: 内含子内的转录本
# 默认只提取 u，可通过 config 扩展

REF_GTF = config["references"]["gtf"]
REF_FASTA = config["references"]["fasta"]

# 确保 RNASEQ_IDS 已定义（从 ingest_samples.smk 导入）
try:
    RNASEQ_IDS
except NameError:
    # 如果未定义，尝试从 ingest_samples.smk 读取
    import csv
    RNASEQ_IDS = []
    try:
        with open("config/samples_rnaseq.csv") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("fastq1") and r.get("fastq2"):
                    RNASEQ_IDS.append(r["sample_id"].strip())
        RNASEQ_IDS = sorted(set(RNASEQ_IDS))
    except Exception:
        raise RuntimeError("无法读取 RNASEQ_IDS，请确保已 include ingest_samples.smk")


############################################
# 1. StringTie 单样本组装（reference-guided）
############################################
rule rnaseq_stringtie_assemble:
    """
    使用 StringTie 对每个样本的 BAM 进行 reference-guided 组装
    输出每个样本的 GTF 文件
    """
    input:
        bam="results/rnaseq/align_hisat2/{sample}.sorted.bam",
        gtf=REF_GTF,
    output:
        gtf="results/rnaseq/stringtie/{sample}.gtf",
    conda:
        "../../envs/rnaseq_envs/stringtie.yaml"
    threads: 16
    resources:
        mem_mb=16000,
        runtime=600,
    log:
        "logs/rnaseq/stringtie/{sample}.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p $(dirname {output.gtf})
        mkdir -p $(dirname {log})

        {{
            stringtie {input.bam} \
                -G {input.gtf} \
                -o {output.gtf} \
                -p {threads} \
                -l {wildcards.sample}
        }} &> {log}

        test -s {output.gtf} || (echo "Error: {output.gtf} is empty" >> {log} && exit 1)
        """


############################################
# 2. StringTie merge 合并所有样本的转录本
############################################
rule rnaseq_stringtie_merge:
    """
    使用 StringTie --merge 合并所有样本的 GTF
    生成统一的 merged.gtf
    """
    input:
        gtfs=expand("results/rnaseq/stringtie/{sample}.gtf", sample=RNASEQ_IDS),
        ref_gtf=REF_GTF,
    output:
        merged="results/rnaseq/stringtie/merged.gtf",
    conda:
        "../../envs/rnaseq_envs/stringtie.yaml"
    threads: 16
    resources:
        mem_mb=16000,
        runtime=1800,
    log:
        "logs/rnaseq/stringtie/merge.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p $(dirname {output.merged})
        mkdir -p $(dirname {log})

        # 创建 GTF 列表文件（在输出目录下）
        gtf_list="$(dirname {output.merged})/gtf_list.txt"
        echo "{input.gtfs}" | tr ' ' '\n' > "$gtf_list"

        {{
            stringtie --merge \
                -G {input.ref_gtf} \
                -o {output.merged} \
                -p {threads} \
                "$gtf_list"
        }} &> {log}

        test -s {output.merged} || (echo "Error: {output.merged} is empty" >> {log} && exit 1)
        """


############################################
# 3. gffcompare 标注新转录本
############################################
rule rnaseq_gffcompare:
    """
    使用 gffcompare 将 merged.gtf 与参考 GTF 比较
    标注每个转录本的 class_code（u=novel, x=antisense, i=intronic 等）
    """
    input:
        merged="results/rnaseq/stringtie/merged.gtf",
        ref_gtf=REF_GTF,
    output:
        annotated="results/rnaseq/stringtie/merged.annotated.gtf",
        tracking="results/rnaseq/stringtie/merged.tracking",
        stats="results/rnaseq/stringtie/merged.stats",
    conda:
        "../../envs/rnaseq_envs/stringtie.yaml"
    threads: 16
    resources:
        mem_mb=16000,
        runtime=600,
    log:
        "logs/rnaseq/stringtie/gffcompare.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p $(dirname {output.annotated})
        mkdir -p $(dirname {log})

        # gffcompare 的 -o 参数是输出前缀，需要去掉扩展名
        gffcompare_prefix="$(dirname {output.annotated})/merged"

        {{
            gffcompare \
                -r {input.ref_gtf} \
                -o "$gffcompare_prefix" \
                -G \
                {input.merged}
        }} &> {log}

        # gffcompare 输出文件命名规则：{{prefix}}.annotated.gtf, {{prefix}}.tracking, {{prefix}}.stats
        test -s {output.annotated} || (echo "Error: {output.annotated} is empty" >> {log} && exit 1)
        """


############################################
# 4. 提取 novel transcripts/genes 并规范化
############################################
rule rnaseq_novel_extract:
    """
    从 gffcompare 输出的 annotated GTF 中提取 novel transcripts/genes
    根据 class_code 筛选（默认 u，可通过 config 配置）
    规范化 gene_id 和 transcript_id 命名（gene:novel000001, transcript:novel000001.1）
    """
    input:
        annotated="results/rnaseq/stringtie/merged.annotated.gtf",
        code="scripts/rnaseq/extract_novel_gtf.py",
    output:
        novel_gtf="references/annotation/genome_novel.gtf",
    conda:
        "../../envs/hsi_env.yaml"  # 使用已有的 Python 环境
    threads: 8
    resources:
        mem_mb=8000,
        runtime=300,
    log:
        "logs/rnaseq/novel/extract_novel_gtf.log"
    params:
        class_codes=",".join(NOVEL_CLASS_CODES),
    shell:
        r"""
        set -euo pipefail
        mkdir -p $(dirname {output.novel_gtf})
        mkdir -p $(dirname {log})

        {{
            python {input.code} \
                --annotated-gtf {input.annotated} \
                --output {output.novel_gtf} \
                --class-codes {params.class_codes}
        }} &> {log}

        test -s {output.novel_gtf} || (echo "Error: {output.novel_gtf} is empty" >> {log} && exit 1)
        """


############################################
# 5. gffread 提取 novel transcripts 序列
############################################
rule rnaseq_novel_gffread_fasta:
    """
    使用 gffread 从 novel GTF 提取转录本序列
    """
    input:
        novel_gtf="references/annotation/genome_novel.gtf",
        genome_fa=REF_FASTA,
    output:
        transcripts_fa="results/rnaseq/novel/novel_transcripts.fa",
    conda:
        "../../envs/rnaseq_envs/stringtie.yaml"
    threads: 16
    resources:
        mem_mb=16000,
        runtime=300,
    log:
        "logs/rnaseq/novel/gffread_fasta.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p $(dirname {output.transcripts_fa})
        mkdir -p $(dirname {log})

        {{
            gffread {input.novel_gtf} \
                -g {input.genome_fa} \
                -w {output.transcripts_fa}
        }} &> {log}

        test -s {output.transcripts_fa} || (echo "Error: {output.transcripts_fa} is empty" >> {log} && exit 1)
        """


############################################
# 6. TransDecoder 预测 novel 蛋白
############################################
rule rnaseq_novel_transdecoder_longorfs:
    """
    TransDecoder 第一步：识别长开放阅读框（ORFs）
    方案：在项目根目录跑 LongOrfs，让它在根目录生成 BASE.transdecoder_dir，
         然后 mv 到 results/rnaseq/novel/... 下面。
    """
    input:
        transcripts_fa="results/rnaseq/novel/novel_transcripts.fa",
    output:
        orf_pep="results/rnaseq/novel/novel_transcripts.fa.transdecoder_dir/longest_orfs.pep",
        orf_gff="results/rnaseq/novel/novel_transcripts.fa.transdecoder_dir/longest_orfs.gff3",
    conda:
        "../../envs/rnaseq_envs/transdecoder.yaml"
    threads: 16
    resources:
        mem_mb=16000,
        runtime=1800,
    log:
        "logs/rnaseq/novel/transdecoder_longorfs.log"
    shell:
        r"""
        set -euo pipefail

        mkdir -p results/rnaseq/novel logs/rnaseq/novel

        FA="{input.transcripts_fa}"
        BASE=$(basename "$FA")                     # novel_transcripts.fa
        DEST_DIR=$(dirname "{output.orf_pep}")     # results/rnaseq/novel/novel_transcripts.fa.transdecoder_dir

        echo "[LongOrfs] CWD: $(pwd)" > {log}
        echo "[LongOrfs] Input: $FA" >> {log}
        echo "[LongOrfs] BASE:  $BASE" >> {log}
        echo "[LongOrfs] DEST_DIR: $DEST_DIR" >> {log}

        # 如果转录本 fasta 本身是空的，直接造空壳输出，避免 TransDecoder 报错
        if [ ! -s "$FA" ]; then
            echo "[LongOrfs] $FA is empty, create empty outputs and exit 0" >> {log}
            mkdir -p "$DEST_DIR"
            : > "{output.orf_pep}"
            : > "{output.orf_gff}"
            exit 0
        fi

        # 在项目根目录直接跑 TransDecoder.LongOrfs
        # 它会在当前目录生成 $BASE.transdecoder_dir
        TransDecoder.LongOrfs \
            -t "$FA" \
            -m 100 \
            >> {log} 2>&1

        # 如果 TransDecoder 在根目录生成了 novel_transcripts.fa.transdecoder_dir，就移动到目标位置
        if [ -d "$BASE.transdecoder_dir" ]; then
            rm -rf "$DEST_DIR"
            mv "$BASE.transdecoder_dir" "$DEST_DIR"
        fi

        # 确认最终目标文件存在且非空
        test -s "{output.orf_pep}" || (echo "[LongOrfs] Error: {output.orf_pep} is empty" >> {log} && exit 1)
        """



rule rnaseq_novel_transdecoder_predict:
    """
    TransDecoder 第二步：预测编码序列（CDS）
    也在 results/rnaseq/novel 目录运行，保证能看到同一个 .transdecoder_dir。
    """
    input:
        transcripts_fa="results/rnaseq/novel/novel_transcripts.fa",
        orf_pep="results/rnaseq/novel/novel_transcripts.fa.transdecoder_dir/longest_orfs.pep",
    output:
        pep="results/rnaseq/novel/novel_transcripts.pep",
        gff3="results/rnaseq/novel/novel_transcripts.gff3",
    conda:
        "../../envs/rnaseq_envs/transdecoder.yaml"
    threads: 16
    resources:
        mem_mb=16000,
        runtime=1800,
    log:
        "logs/rnaseq/novel/transdecoder_predict.log"
    shell:
        r"""
        set -euo pipefail

        ROOT_DIR="$(pwd)"

        TD_DIR="$ROOT_DIR/$(dirname "{input.transcripts_fa}")"
        TD_BASE="$(basename "{input.transcripts_fa}")"   # novel_transcripts.fa

        LOG_PATH="$ROOT_DIR/{log}"
        DEST_PEP="$ROOT_DIR/{output.pep}"
        DEST_GFF="$ROOT_DIR/{output.gff3}"

        mkdir -p "$ROOT_DIR/results/rnaseq/novel" "$ROOT_DIR/logs/rnaseq/novel"

        (
          echo "[Predict] ROOT_DIR: $ROOT_DIR"
          echo "[Predict] TD_DIR  : $TD_DIR"
          echo "[Predict] TD_BASE : $TD_BASE"

          cd "$TD_DIR"
          echo "[Predict] CWD after cd: $(pwd)"

          # 如果 longorfs 的 pep 是空的/不存在，直接写空结果
          if [ ! -s "$TD_BASE.transdecoder_dir/longest_orfs.pep" ]; then
              echo "[Predict] $TD_BASE.transdecoder_dir/longest_orfs.pep is empty or missing, write empty outputs and exit 0"
              : > "$DEST_PEP"
              : > "$DEST_GFF"
              exit 0
          fi

          # 正式跑 Predict
          TransDecoder.Predict \
              -t "$TD_BASE" \
              --retain_long_orfs_mode dynamic

          TD_PEP="$TD_BASE.transdecoder.pep"
          TD_GFF="$TD_BASE.transdecoder.gff3"

          # 处理 pep
          if [ -s "$TD_PEP" ]; then
              mv "$TD_PEP" "$DEST_PEP"
          else
              echo "[Predict] Error: $TD_PEP not found or empty"
              exit 1
          fi

          # 处理 gff3（没有也不致命，就建个空壳）
          if [ -f "$TD_GFF" ]; then
              mv "$TD_GFF" "$DEST_GFF"
          else
              echo "[Predict] Warning: $TD_GFF not found, create empty placeholder"
              : > "$DEST_GFF"
          fi
        ) > "$LOG_PATH" 2>&1

        test -s "$DEST_PEP" || (echo "[Predict] Error: $DEST_PEP is empty" >> "$LOG_PATH" && exit 1)
        """






############################################
# 7. eggNOG-mapper 注释 novel 蛋白
############################################
rule rnaseq_novel_eggnog_raw:
    """
    使用 eggNOG-mapper v2 对 novel 蛋白质序列进行功能注释
    """
    input:
        pep="results/rnaseq/novel/novel_transcripts.pep",
    output:
        annotations="results/rnaseq/eggnog/novel_emapper.emapper.annotations",
        seed_orthologs="results/rnaseq/eggnog/novel_emapper.emapper.seed_orthologs",
        hits="results/rnaseq/eggnog/novel_emapper.emapper.hits",
    params:
        prefix="results/rnaseq/eggnog/novel_emapper",
    conda:
        "../../envs/rnaseq_envs/eggnog.yaml"
    threads: 16
    resources:
        mem_mb=32000,
        runtime=7200,
    log:
        "logs/rnaseq/eggnog/novel_emapper_raw.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p $(dirname {output.annotations})
        mkdir -p $(dirname {log})

        {{
            emapper.py \
                -i {input.pep} \
                --itype proteins \
                -m diamond \
                -o {params.prefix} \
                --cpu {threads} \
                --data_dir /public/agis/wanfanghao_group/zoujianghua/eggnog_data \
                --dmnd_db /public/agis/wanfanghao_group/zoujianghua/eggnog_data/eggnog_proteins.dmnd
        }} &> {log}

        test -s {output.annotations} || (echo "Error: {output.annotations} is empty" >> {log} && exit 1)
        """


rule rnaseq_novel_eggnog_tidy:
    """
    整理 novel 蛋白的 eggNOG 注释结果，按 gene_id 聚合
    """
    input:
        annotations="results/rnaseq/eggnog/novel_emapper.emapper.annotations",
        novel_gtf="references/annotation/genome_novel.gtf",
        code="scripts/rnaseq/tidy_eggnog_annotations.py",
    output:
        tidy="results/rnaseq/eggnog/novel_eggnog_annotations.tsv",
    conda:
        "../../envs/hsi_env.yaml"
    threads: 4
    resources:
        mem_mb=8000,
        runtime=300,
    log:
        "logs/rnaseq/eggnog/novel_tidy_annotations.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p $(dirname {output.tidy})
        mkdir -p $(dirname {log})

        {{
            python {input.code} \
                --annotations {input.annotations} \
                --gtf {input.novel_gtf} \
                --output {output.tidy}
        }} &> {log}

        test -s {output.tidy} || (echo "Error: {output.tidy} is empty" >> {log} && exit 1)
        """


############################################
# 8. Novel 发现与注释规模汇总表
############################################
rule rnaseq_novel_summary:
    """
    汇总 novel transcripts/genes 发现与注释规模，生成一张便于论文正文/附录引用的表。
    """
    input:
        novel_gtf="references/annotation/genome_novel.gtf",
        novel_pep="results/rnaseq/novel/novel_transcripts.pep",
        novel_eggnog="results/rnaseq/eggnog/novel_eggnog_annotations.tsv",
        code="scripts/rnaseq/summarize_novel_discovery.py",
    output:
        summary="results/rnaseq/novel/novel_discovery_summary.tsv",
    conda:
        "../../envs/hsi_env.yaml"
    threads: 1
    resources:
        mem_mb=2000,
        runtime=30,
    log:
        "logs/rnaseq/novel/novel_discovery_summary.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/rnaseq/novel logs/rnaseq/novel
        python {input.code} \
          --novel-gtf {input.novel_gtf} \
          --novel-pep {input.novel_pep} \
          --novel-eggnog {input.novel_eggnog} \
          --output {output.summary} \
          > {log} 2>&1
        """


############################################
# 8. 合并 reference + novel 的 eggNOG 注释
############################################
rule rnaseq_eggnog_merge_novel:
    """
    合并 reference 和 novel 的 eggNOG 注释表
    检查并避免 gene_id 重复
    注意：此规则依赖 rnaseq_eggnog_tidy（reference 注释）和 rnaseq_novel_eggnog_tidy（novel 注释）
    """
    input:
        ref_annot="results/rnaseq/eggnog/eggnog_annotations.tsv",
        novel_annot="results/rnaseq/eggnog/novel_eggnog_annotations.tsv",
        code="scripts/rnaseq/merge_eggnog_annotations.py",
    output:
        merged="results/rnaseq/eggnog/eggnog_annotations_with_novel.tsv",
    conda:
        "../../envs/hsi_env.yaml"
    threads: 4
    resources:
        mem_mb=4000,
        runtime=60,
    log:
        "logs/rnaseq/eggnog/merge_novel.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p $(dirname {output.merged})
        mkdir -p $(dirname {log})

        {{
            python {input.code} \
                --ref-annot {input.ref_annot} \
                --novel-annot {input.novel_annot} \
                --output {output.merged}
        }} &> {log}

        test -s {output.merged} || (echo "Error: {output.merged} is empty" >> {log} && exit 1)
        """


############################################
# 9. 合并 reference + novel 的 GTF
############################################
rule ref_genes_with_novel_gtf:
    """
    将 reference GTF 和 novel GTF 合并
    生成 genes_with_novel.gtf，用于后续重新计数
    """
    input:
        ref_gtf=REF_GTF,
        novel_gtf="references/annotation/genome_novel.gtf",
        code="scripts/rnaseq/merge_gtf.py",
    output:
        merged_gtf="references/annotation/genes_with_novel.gtf",
    conda:
        "../../envs/hsi_env.yaml"
    threads: 4
    resources:
        mem_mb=4000,
        runtime=60,
    log:
        "logs/rnaseq/novel/merge_gtf.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p $(dirname {output.merged_gtf})
        mkdir -p $(dirname {log})

        {{
            python {input.code} \
                --ref-gtf {input.ref_gtf} \
                --novel-gtf {input.novel_gtf} \
                --output {output.merged_gtf}
        }} &> {log}

        test -s {output.merged_gtf} || (echo "Error: {output.merged_gtf} is empty" >> {log} && exit 1)
        """
