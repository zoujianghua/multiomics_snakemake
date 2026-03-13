# rules/rnaseq/align_hisat2.smk


rule rnaseq_align_hisat2:
    input:
        # ä¾èµçæ­£çç´¢å¼æä»¶ï¼8ä¸ªï¼ï¼è¿æ · Snakemake æä¼å»å»º / æ£æ¥å®ä»¬
        idx=expand("{p}.{i}.ht2", p=config["references"]["hisat2_index"], i=range(1, 9)),
        r1="results/rnaseq/clean/{sample}_R1.fastq.gz",
        r2="results/rnaseq/clean/{sample}_R2.fastq.gz",
    params:
        # æâåç¼âåç¬ä½ä¸ºåæ°ä¼ ç» hisat2 -x
        idx_prefix=lambda wc: config["references"]["hisat2_index"],
    output:
        bam="results/rnaseq/align_hisat2/{sample}.sorted.bam",
        bai="results/rnaseq/align_hisat2/{sample}.sorted.bam.bai",
    threads: 8
    resources:
        mem_mb=48000,
        runtime=180,
    conda:
        "../../envs/rnaseq_envs/align_hisat2.yaml"
    log:
        "logs/rnaseq/hisat2/{sample}.log",
    shell:
        r"""
        mkdir -p results/rnaseq/align_hisat2 logs/rnaseq/hisat2
        hisat2 -x {params.idx_prefix} -1 {input.r1} -2 {input.r2} -p {threads} --dta 2> {log} \
        | samtools sort -@ {threads} -o {output.bam}
        samtools index {output.bam}
        """
