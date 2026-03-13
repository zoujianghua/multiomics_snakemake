# 输入文件在 config 里指定
REF_FASTA = config["references"]["fasta"]
REF_GFF3 = config.get("references", {}).get(
    "gff3", REF_FASTA.replace(".fa", ".gff3").replace(".fasta", ".gff3")
)
HISAT2_PREFIX = config["references"][
    "hisat2_index"
]  # 例如 /.../references/genome/hisat2/genome
GTF_OUT = config["references"]["gtf"]


# 1) samtools faidx
rule ref_faidx:
    input:
        fasta=REF_FASTA,
    output:
        fai=REF_FASTA + ".fai",
    conda:
        "../envs/wgs_envs/postbam.yaml"  # 内有 samtools
    log: "logs/ref/ref_faidx.log"
    shell:
        "samtools faidx {input.fasta}"


# 2) GATK dict
rule ref_dict:
    input:
        fasta=REF_FASTA,
    output:
        dict=REF_FASTA.rsplit(".", 1)[0] + ".dict",
    conda:
        "../envs/wgs_envs/gatk.yaml"
    log: "logs/ref/ref_dict.log"
    shell:
        r"gatk CreateSequenceDictionary -R {input.fasta} -O {output.dict}"


# 3) BWA index（在 fasta 所在目录生成）
rule ref_bwa_index:
    input:
        fasta=REF_FASTA,
    output:
        amb=REF_FASTA + ".amb",
        ann=REF_FASTA + ".ann",
        bwt=REF_FASTA + ".bwt",
        pac=REF_FASTA + ".pac",
        sa=REF_FASTA + ".sa",
    conda:
        "../envs/wgs_envs/align_bwa.yaml"
    threads: 8
    log: "logs/ref/ref_bwa_index.log"
    shell:
        "bwa index {input.fasta}"


# 4) HISAT2 index
rule ref_hisat2_index:
    input:
        fasta=REF_FASTA,
    output:
        expand("{prefix}.{i}.ht2", prefix=HISAT2_PREFIX, i=range(1, 9)),
    conda:
        "../envs/rnaseq_envs/align_hisat2.yaml"
    threads: 16
    log: "logs/ref/ref_hisat2_index.log"
    shell:
        r"""
        mkdir -p $(dirname {HISAT2_PREFIX})
        hisat2-build -p {threads} {input.fasta} {HISAT2_PREFIX}
        """


# 5) GFF3 -> GTF
rule ref_gff3_to_gtf:
    input:
        gff3=REF_GFF3,
    output:
        gtf=GTF_OUT,
    conda:
        "../envs/annotation.yaml"  # 很小的 gffread 环境
    log: "logs/ref/ref_gff3_to_gtf.log"
    shell:
        r"""
        mkdir -p $(dirname {output.gtf})
        gffread {input.gff3} -T -o {output.gtf}
        """


# 6) GTF -> Protein FASTA (for eggNOG-mapper)
rule ref_proteins_from_gtf:
    """
    从 GTF 和基因组 FASTA 生成蛋白质序列 FASTA
    用于后续 eggNOG-mapper 注释
    """
    input:
        gtf=GTF_OUT,
        fasta=REF_FASTA,
    output:
        proteins="references/proteins_from_gtf.fa",
    conda:
        "../envs/annotation.yaml"  # 使用已有的 gffread 环境
    threads: 16
    resources:
        mem_mb=16000,
        runtime=300,
    log: "logs/ref/ref_proteins_from_gtf.log"
    shell:
        r"""
        mkdir -p references logs/ref
        # 使用 gffread 从 GTF 提取蛋白质序列
        # -y: 输出蛋白质序列
        # -S: 使用 CDS 特征（默认）
        gffread {input.gtf} -g {input.fasta} -y {output.proteins} -S \
            > {log} 2>&1

        # 验证输出文件存在且非空
        test -s {output.proteins} || (echo "Error: {output.proteins} is empty" && exit 1)
        """


rule all_references:
    input:
        rules.ref_faidx.output,
        rules.ref_dict.output,
        rules.ref_bwa_index.output,
        rules.ref_hisat2_index.output,
        rules.ref_gff3_to_gtf.output,
        rules.ref_proteins_from_gtf.output,
