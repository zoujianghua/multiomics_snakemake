# rules/rnaseq/annot_eggnog.smk
# eggNOG-mapper v2 注释流程
#
# 流程：
#   1. ref_proteins_from_gtf: 从 GTF 生成蛋白质 FASTA
#   2. rnaseq_eggnog_raw: 运行 eggNOG-mapper 获得原始注释
#   3. rnaseq_eggnog_tidy: 整理并聚合为按 gene_id 汇总的注释表

############################################
# 1) eggNOG-mapper 原始注释
############################################
rule rnaseq_eggnog_raw:
    """
    使用 eggNOG-mapper v2 对蛋白质序列进行功能注释
    使用 DIAMOND 模式进行快速比对
    """
    input:
        proteins="references/proteins_from_gtf.fa",
    output:
        annotations="results/rnaseq/eggnog/mikania_emapper.emapper.annotations",
        seed_orthologs="results/rnaseq/eggnog/mikania_emapper.emapper.seed_orthologs",
        hits="results/rnaseq/eggnog/mikania_emapper.emapper.hits",
    params:
        prefix="results/rnaseq/eggnog/mikania_emapper",
    conda:
        "../../envs/rnaseq_envs/eggnog.yaml"
    threads: 16
    resources:
        mem_mb=32000,
        runtime=7200,  # 2小时，根据实际数据量调整
    log:
        "logs/rnaseq/eggnog/emapper_raw.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/rnaseq/eggnog logs/rnaseq/eggnog

        # 运行 eggNOG-mapper
        # -i: 输入文件
        # --itype proteins: 输入类型为蛋白质序列
        # -m diamond: 使用 DIAMOND 模式（快速）
        # -o: 输出前缀
        # --cpu: CPU 线程数
        # --data_dir: 如果使用本地数据库，可指定路径（默认使用在线数据库）
        emapper.py \
            -i {input.proteins} \
            --itype proteins \
            -m diamond \
            -o {params.prefix} \
            --cpu {threads} \
            --data_dir /public/agis/wanfanghao_group/zoujianghua/eggnog_data \
            --dmnd_db /public/agis/wanfanghao_group/zoujianghua/eggnog_data/eggnog_proteins.dmnd \
            > {log} 2>&1

        # 验证输出文件存在
        test -s {output.annotations} || (echo "Error: {output.annotations} is empty" && exit 1)
        """


############################################
# 2) 整理 eggNOG 注释为按 gene_id 汇总的 TSV
############################################
rule rnaseq_eggnog_tidy:
    """
    整理 eggNOG 原始注释文件，按 gene_id 聚合
    输出格式与 featureCounts 的 gene_id 保持一致
    """
    input:
        annotations="results/rnaseq/eggnog/mikania_emapper.emapper.annotations",
        gtf=config["references"]["gtf"],
        code="scripts/rnaseq/tidy_eggnog_annotations.py",
    output:
        tidy="results/rnaseq/eggnog/eggnog_annotations.tsv",
    conda:
        "../../envs/hsi_env.yaml"  # 使用已有的 Python 环境（包含 pandas）
    threads: 16
    resources:
        mem_mb=8000,
        runtime=300,
    log:
        "logs/rnaseq/eggnog/tidy_annotations.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/rnaseq/eggnog logs/rnaseq/eggnog

        python {input.code} \
            --annotations {input.annotations} \
            --gtf {input.gtf} \
            --output {output.tidy} \
            > {log} 2>&1

        # 验证输出文件存在且非空
        test -s {output.tidy} || (echo "Error: {output.tidy} is empty" && exit 1)
        """

