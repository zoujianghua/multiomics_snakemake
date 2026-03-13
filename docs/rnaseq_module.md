# RNAseq（转录组学）模块文档

## 概述

RNAseq 模块处理 RNA-seq 双端测序数据，流程包括：质量控制（FastQC、fastp、MultiQC）、HISAT2 比对、featureCounts 计数、设计矩阵生成、DESeq2 差异表达分析、eggNOG-mapper 功能注释，以及可选的 novel transcripts/genes 发现与合并计数。**本 pipeline 在服务器上完成至计数与差异分析；下游的 pathway 富集、聚类、高级可视化等建议在本地 R 环境中进行**，便于交互式调试与图表修改。

## 模块结构

### 规则文件

- `rules/rnaseq/ingest_samples.smk`：样本导入
- `rules/rnaseq/qc_fastp.smk`：质量控制（FastQC、fastp、MultiQC）
- `rules/rnaseq/align_hisat2.smk`：HISAT2 比对
- `rules/rnaseq/quantify_featurecounts.smk`：featureCounts 计数
- `rules/rnaseq/de_deseq2.smk`：DESeq2 差异表达分析
- `rules/rnaseq/annot_eggnog.smk`：eggNOG-mapper 功能注释
- `rules/rnaseq/novel_transcripts.smk`：Novel transcripts/genes 发现与注释
- `rules/rnaseq/rnaseq_all.smk`：RNAseq 模块的汇总入口

### 输入文件

- `config/samples_rnaseq.csv`：RNAseq 样本表，包含样本 ID、fastq1、fastq2 路径及元数据（temp、phase、time、group）

### 输出目录

- `results/rnaseq/`：所有 RNAseq 相关结果
  - `clean/`：修剪后的 FASTQ 文件
  - `qc/`：质量控制结果（FastQC、fastp、MultiQC）
  - `align_hisat2/`：HISAT2 比对结果（BAM 文件）
  - `counts/`：计数矩阵
  - `deseq2/`：DESeq2 差异表达分析结果
  - `eggnog/`：eggNOG 功能注释结果
  - `stringtie/`：StringTie 组装结果
  - `novel/`：Novel transcripts 相关结果

---

## 1. 样本导入

### 1.1 样本表格式

**文件**：`config/samples_rnaseq.csv`

**必需列**：
- `sample_id`：样本 ID
- `fastq1`：R1 FASTQ 文件路径
- `fastq2`：R2 FASTQ 文件路径

**可选列**（元数据）：
- `temperature`：温度
- `phase`：阶段
- `time`：时间点
- `group`：分组信息

### 1.2 样本字典

**规则文件**：`rules/rnaseq/ingest_samples.smk`

**输出**：
- `RNASEQ` 字典：包含每个样本的 fastq1、fastq2 路径及元数据
- `RNASEQ_IDS` 列表：所有样本 ID 的排序列表

**使用**：其他规则通过 `RNASEQ[wc.sample]["r1"]`、`RNASEQ[wc.sample]["r2"]` 等访问样本路径；`all_rnaseq` 的输入包含 `config/rnaseq_design.tsv`，保证设计矩阵在计数前已生成。

---

## 2. 质量控制

### 2.1 FastQC（原始数据）

**规则**：`rnaseq_fastqc_raw`

**输入**：
- R1 FASTQ：`RNASEQ[sample]["r1"]`
- R2 FASTQ：`RNASEQ[sample]["r2"]`

**输出**：`results/rnaseq/qc/fastqc_raw/{sample}/`（目录型输出，`directory()`）

**功能**：对原始 R1/R2 FASTQ 运行 FastQC；规则中 `mkdir -p {output.outdir} logs/rnaseq/fastqc_raw`。

### 2.2 fastp（数据修剪）

**规则**：`rnaseq_fastp`

**输入**：
- R1 FASTQ：`RNASEQ[sample]["r1"]`
- R2 FASTQ：`RNASEQ[sample]["r2"]`

**输出**：
- `results/rnaseq/clean/{sample}_R1.fastq.gz`：修剪后的 R1
- `results/rnaseq/clean/{sample}_R2.fastq.gz`：修剪后的 R2
- `results/rnaseq/qc/fastp/{sample}.fastp.html`：fastp HTML 报告
- `results/rnaseq/qc/fastp/{sample}.fastp.json`：fastp JSON 报告

**参数**（与 `rules/rnaseq/qc_fastp.smk` 一致）：
- `-q 20`：碱基质量阈值
- `-u 30`：未识别碱基比例阈值
- `-n 5`：N 碱基数量阈值
- `-l 25`：最小读长
- `--detect_adapter_for_pe`：自动检测接头
- 规则会先 `mkdir -p results/rnaseq/clean results/rnaseq/qc/fastp logs/rnaseq/fastp`

### 2.3 FastQC（修剪后数据）

**规则**：`rnaseq_fastqc_clean`

**输入**：
- 修剪后的 R1/R2 FASTQ

**输出**：`results/rnaseq/qc/fastqc_clean/{sample}/`（目录型输出）。规则中 `mkdir -p {output.outdir} logs/rnaseq/fastqc_clean`。

### 2.4 MultiQC 汇总

**规则**：`rnaseq_multiqc`

**与重测序（WGS）解耦**：本规则**仅**依赖 RNA-seq 模块的输入（见下方），不依赖任何 `results/wgs` 或 `logs/wgs`。转录组与重测序的 MultiQC 互不依赖，可单独运行。

**输入**：
- 所有 FastQC 结果（raw/clean）
- 所有 fastp 报告
- HISAT2 日志
- featureCounts 汇总文件

**输出**：
- `results/rnaseq/qc/multiqc/multiqc_report.html`
- `results/rnaseq/qc/multiqc/multiqc_data/`（由 MultiQC 自动生成，内含 `multiqc_general_stats.txt` 等，供上游 QC 汇总表使用）

**功能**：仅扫描 `results/rnaseq` 与 `logs/rnaseq`，汇总上述 QC 结果到 HTML 与 multiqc_data。

若 `upstream_qc_summary.tsv` 报错“未找到 general stats”，说明 MultiQC 尚未正确跑完或 multiqc_data 未生成，请先执行：
`snakemake --profile profiles/slurm results/rnaseq/qc/multiqc/multiqc_report.html`，并查看 `logs/rnaseq/multiqc/multiqc.log`。

---

## 3. 比对

### 3.1 HISAT2 比对

**规则**：`rnaseq_align_hisat2`

**输入**：
- HISAT2 索引：`config["references"]["hisat2_index"]`（8 个 .ht2 文件）
- 修剪后的 R1/R2 FASTQ

**输出**：
- `results/rnaseq/align_hisat2/{sample}.sorted.bam`：排序后的 BAM 文件
- `results/rnaseq/align_hisat2/{sample}.sorted.bam.bai`：BAM 索引文件

**工具**：HISAT2 + samtools

**参数**：
- `--dta`：下游转录本组装模式
- `--rna-strandness`：链特异性（从 config 读取）

---

## 4. 计数

### 4.1 featureCounts 计数

**规则**：`rnaseq_featurecounts_hisat2`

**输入**：
- GTF 文件：`config["references"]["gtf"]`
- 所有样本的 BAM 文件

**输出**：
- `results/rnaseq/counts/featurecounts_hisat2.tsv`：计数矩阵
- `results/rnaseq/counts/featurecounts_hisat2.tsv.summary`：汇总统计

**参数**（与 `rules/rnaseq/quantify_featurecounts.smk` 一致）：
- `-t exon -g gene_id`：按 exon 计数并聚合到 gene_id
- `-p -B -C`：paired-end，仅统计正确配对的 fragments
- `-s {stranded}`：链特异性（来自 `config["rnaseq"]["stranded"]`）
- `-Q 10`：最小 mapping quality
- 规则会 `mkdir -p results/rnaseq/counts logs/rnaseq/featurecounts`

### 4.2 基于合并 GTF 的重新计数（Novel Pipeline）

**规则**：`rnaseq_featurecounts_hisat2_with_novel`

**输入**：
- 合并 GTF：`references/annotation/genes_with_novel.gtf`（reference + novel）
- 所有样本的 BAM 文件

**输出**：
- `results/rnaseq/counts/featurecounts_hisat2_with_novel.tsv`：包含 novel 基因的计数矩阵
- `results/rnaseq/counts/featurecounts_hisat2_with_novel.tsv.summary`：汇总统计

**说明**：这是原规则的平行版本，使用合并后的 GTF（包含 novel transcripts）

---

## 5. 差异表达分析

### 5.1 设计矩阵生成

**规则**：`rnaseq_make_design`

**输入**：`config/samples_rnaseq.csv`、`scripts/make_rnaseq_design.py`

**输出**：`config/rnaseq_design.tsv`（设计矩阵，供 DESeq2 使用）

**功能**：从样本表生成 DESeq2 所需的设计矩阵。调用方式：`python {input.code} {input.csv} {output.tsv}`（与 `rules/rnaseq/de_deseq2.smk` 一致）。

### 5.2 DESeq2 差异分析

**规则**：`rnaseq_deseq2`

**输入**：计数矩阵 `results/rnaseq/counts/featurecounts_hisat2.tsv`、设计矩阵 `config/rnaseq_design.tsv`、对比文件 `config/contrasts.tsv`、R 脚本 `scripts/rnaseq/rnaseq_deseq2_contrasts.R`。

**输出**：`results/rnaseq/deseq2/normalized_counts.tsv`（标准化计数）、`results/rnaseq/deseq2/DEG_results.tsv`（差异表达基因，含 log2FoldChange、padj 等）。

**功能**：按 `config/contrasts.tsv` 中定义的对比组批量运行 DESeq2，输出标准化计数与 DEG 结果。**下游分析**（如 KEGG/GO 富集、WGCNA、聚类与自定义图表）建议在本地 R 中基于上述 TSV 完成。

**参数**（与 `rules/rnaseq/de_deseq2.smk` 一致）：`--min_per_grp 2`、`--lfc_th 1`、`--fdr_th 0.05`；规则中会先执行 `mkdir -p results/rnaseq/deseq2 logs/rnaseq`。

---

## 6. 功能注释（eggNOG-mapper）

### 6.1 蛋白质序列生成

**规则**：`ref_proteins_from_gtf`（位于 `rules/references.smk`）

**输入**：
- GTF 文件：`references/annotation/genes.gtf`
- 基因组 FASTA：`references/Mikania_micrantha.RefGenome.Chromosome.fasta`

**输出**：
- `references/proteins_from_gtf.fa`

**工具**：gffread

**功能**：从 GTF 文件中提取 CDS 序列并翻译为蛋白质序列

### 6.2 eggNOG-mapper 注释

**规则**：`rnaseq_eggnog_raw`

**输入**：
- `references/proteins_from_gtf.fa`

**输出**：
- `results/rnaseq/eggnog/mikania_emapper.emapper.annotations`：主要注释文件
- `results/rnaseq/eggnog/mikania_emapper.emapper.seed_orthologs`
- `results/rnaseq/eggnog/mikania_emapper.emapper.hits`

**工具**：eggNOG-mapper v2 (emapper.py)

**参数**：
- `--itype proteins`：输入类型为蛋白质序列
- `-m diamond`：使用 DIAMOND 比对模式

### 6.3 注释结果整理

**规则**：`rnaseq_eggnog_tidy`

**输入**：
- eggNOG 注释文件：`results/rnaseq/eggnog/mikania_emapper.emapper.annotations`
- GTF 文件：`references/annotation/genes.gtf`

**输出**：
- `results/rnaseq/eggnog/eggnog_annotations.tsv`

**脚本**：`scripts/rnaseq/tidy_eggnog_annotations.py`

**功能**：
- 从 GTF 文件中提取 transcript_id -> gene_id 映射
- 将 eggNOG 注释从 transcript 级别聚合到 gene 级别
- 提取 KEGG KO、KEGG pathway、GO terms、功能描述等字段

**输出格式**：
- `gene_id`：基因 ID（格式：`gene:Mm01G000001`）
- `kegg_ko`：KEGG KO 编号（多个值用分号分隔）
- `kegg_pathway`：KEGG 通路（多个值用分号分隔）
- `go_terms`：GO 术语（多个值用分号分隔）
- `description`：功能描述

---

## 7. Novel Transcripts/Genes 发现与注释

### 7.1 流程概述

Novel pipeline 用于发现和注释薇甘菊转录组中的新转录本和新基因，包括：

1. StringTie 组装（单样本 + merge）
2. gffcompare 标注新转录本
3. 提取并规范化 novel GTF
4. TransDecoder 预测 novel 蛋白
5. eggNOG-mapper 注释 novel 蛋白
6. 合并 reference + novel 的 GTF 和 eggNOG 注释
7. 基于合并 GTF 重新计数

### 7.2 StringTie 单样本组装

**规则**：`rnaseq_stringtie_assemble`

**输入**：
- BAM 文件：`results/rnaseq/align_hisat2/{sample}.sorted.bam`
- 参考 GTF：`references/annotation/genes.gtf`

**输出**：
- `results/rnaseq/stringtie/{sample}.gtf`

**工具**：stringtie（reference-guided 模式）

**参数**：
- `-G`：指定参考 GTF
- `-l`：样本标签（使用 sample_id）
- `-p 4`：线程数

### 7.3 StringTie Merge

**规则**：`rnaseq_stringtie_merge`

**输入**：
- 所有样本的 GTF：`results/rnaseq/stringtie/{sample}.gtf`
- 参考 GTF：`references/annotation/genes.gtf`

**输出**：
- `results/rnaseq/stringtie/merged.gtf`

**工具**：stringtie --merge

**功能**：合并所有样本的转录本，生成统一的转录本集合

### 7.4 gffcompare 标注

**规则**：`rnaseq_gffcompare`

**输入**：
- merged GTF：`results/rnaseq/stringtie/merged.gtf`
- 参考 GTF：`references/annotation/genes.gtf`

**输出**：
- `results/rnaseq/stringtie/merged.annotated.gtf`：标注后的 GTF
- `results/rnaseq/stringtie/merged.tracking`：追踪文件
- `results/rnaseq/stringtie/merged.stats`：统计信息

**工具**：gffcompare

**功能**：将 merged GTF 与参考 GTF 比较，为每个转录本标注 class_code：
- `u`：完全落在基因间区的全新转录本（默认提取）
- `x`：反义链上的转录本
- `i`：内含子内的转录本

### 7.5 提取 Novel GTF

**规则**：`rnaseq_novel_extract`

**输入**：
- annotated GTF：`results/rnaseq/stringtie/merged.annotated.gtf`
- 脚本：`scripts/rnaseq/extract_novel_gtf.py`

**输出**：
- `references/annotation/genome_novel.gtf`

**脚本**：`scripts/rnaseq/extract_novel_gtf.py`

**功能**：
- 根据 class_code 筛选 novel 转录本（默认 `u`，可通过 config 配置）
- 规范化 gene_id 和 transcript_id 命名：
  - gene_id：`gene:novel000001`
  - transcript_id：`transcript:novel000001.1`
- 确保输出 GTF 包含 transcript、exon、CDS、gene 等所有行

**配置**：
- 在 `config/config.yaml` 中设置 `rnaseq.novel_class_codes`（默认 `["u"]`）

### 7.6 Novel Transcripts 序列提取

**规则**：`rnaseq_novel_gffread_fasta`

**输入**：
- novel GTF：`references/annotation/genome_novel.gtf`
- 基因组 FASTA：`references/Mikania_micrantha.RefGenome.Chromosome.fasta`

**输出**：
- `results/rnaseq/novel/novel_transcripts.fa`

**工具**：gffread

**功能**：从基因组序列中提取 novel transcripts 的 cDNA 序列

### 7.7 TransDecoder 蛋白预测

**规则链**：

1. **`rnaseq_novel_transdecoder_longorfs`**：识别长开放阅读框（ORFs）
   - 输入：`results/rnaseq/novel/novel_transcripts.fa`
   - 输出：`results/rnaseq/novel/novel_transcripts.fa.transdecoder_dir/longest_orfs.pep`
   - 工具：TransDecoder.LongOrfs
   - 参数：`-m 100`（最小 ORF 长度）

2. **`rnaseq_novel_transdecoder_predict`**：预测编码序列（CDS）
   - 输入：
     - transcripts FASTA：`results/rnaseq/novel/novel_transcripts.fa`
     - ORF peptides：`results/rnaseq/novel/novel_transcripts.fa.transdecoder_dir/longest_orfs.pep`
   - 输出：
     - `results/rnaseq/novel/novel_transcripts.pep`：预测的蛋白序列
     - `results/rnaseq/novel/novel_transcripts.gff3`：CDS 注释
   - 工具：TransDecoder.Predict

### 7.8 Novel 蛋白 eggNOG 注释

**规则链**：

1. **`rnaseq_novel_eggnog_raw`**：运行 eggNOG-mapper
   - 输入：`results/rnaseq/novel/novel_transcripts.pep`
   - 输出：
     - `results/rnaseq/eggnog/novel_emapper.emapper.annotations`
     - `results/rnaseq/eggnog/novel_emapper.emapper.seed_orthologs`
     - `results/rnaseq/eggnog/novel_emapper.emapper.hits`
   - 工具：eggNOG-mapper v2 (emapper.py)
   - 参数：与 reference 蛋白注释保持一致（DIAMOND 模式）

2. **`rnaseq_novel_eggnog_tidy`**：整理注释结果
   - 输入：
     - eggNOG 注释：`results/rnaseq/eggnog/novel_emapper.emapper.annotations`
     - novel GTF：`references/annotation/genome_novel.gtf`
   - 输出：`results/rnaseq/eggnog/novel_eggnog_annotations.tsv`
   - 脚本：`scripts/rnaseq/tidy_eggnog_annotations.py`
   - 功能：按 gene_id 聚合注释信息（KEGG、GO、描述等）

### 7.9 合并 Reference + Novel 的 eggNOG 注释

**规则**：`rnaseq_eggnog_merge_novel`

**输入**：
- reference 注释：`results/rnaseq/eggnog/eggnog_annotations.tsv`
- novel 注释：`results/rnaseq/eggnog/novel_eggnog_annotations.tsv`
- 脚本：`scripts/rnaseq/merge_eggnog_annotations.py`

**输出**：
- `results/rnaseq/eggnog/eggnog_annotations_with_novel.tsv`

**功能**：
- 合并两个注释表
- 检查并避免 gene_id 重复（如果 novel 的 gene_id 与 reference 重复，保留 reference 的注释并记录警告）

### 7.10 合并 Reference + Novel 的 GTF

**规则**：`ref_genes_with_novel_gtf`

**输入**：
- reference GTF：`references/annotation/genes.gtf`
- novel GTF：`references/annotation/genome_novel.gtf`
- 脚本：`scripts/rnaseq/merge_gtf.py`

**输出**：
- `references/annotation/genes_with_novel.gtf`

**功能**：
- 合并两个 GTF 文件
- 检查并避免 gene_id 重复（如果 novel 的 gene_id 与 reference 重复，跳过 novel 的条目）
- 确保输出 GTF 格式正确，可用于 featureCounts

---

## 8. 运行方法

### 运行完整 RNAseq 流程

```bash
# 运行所有 RNAseq 相关规则
snakemake -j 16 all_rnaseq --use-conda

# 或分步运行
snakemake -j 16 rnaseq_fastp --use-conda
snakemake -j 16 rnaseq_align_hisat2 --use-conda
snakemake -j 16 rnaseq_featurecounts_hisat2 --use-conda
snakemake -j 16 rnaseq_deseq2 --use-conda
```

### 运行 eggNOG 注释流程

```bash
# 运行完整的 eggNOG 注释流程
snakemake -j 16 rnaseq_eggnog_all --use-conda

# 或只运行注释整理步骤（如果已经完成 eggNOG-mapper 注释）
snakemake -j 4 rnaseq_eggnog_tidy --use-conda
```

### 运行 Novel Pipeline

```bash
# 运行完整的 novel pipeline（包括所有步骤）
snakemake -j 16 rnaseq_novel_all --use-conda

# 或分步运行
snakemake -j 16 rnaseq_stringtie_merge --use-conda
snakemake -j 4 rnaseq_gffcompare --use-conda
snakemake -j 4 rnaseq_novel_extract --use-conda
snakemake -j 8 rnaseq_novel_transdecoder_predict --use-conda
snakemake -j 16 rnaseq_novel_eggnog_tidy --use-conda
snakemake -j 4 ref_genes_with_novel_gtf rnaseq_eggnog_merge_novel --use-conda
snakemake -j 8 rnaseq_featurecounts_hisat2_with_novel --use-conda
```

---

## 9. 配置说明

### config/config.yaml

```yaml
rnaseq:
  novel_class_codes: ["u"]  # gffcompare class_code 筛选，可选 ["u", "x", "i"]
  
references:
  hisat2_index: "references/hisat2_index/genome"
  gtf: "references/annotation/genes.gtf"
  fasta: "references/Mikania_micrantha.RefGenome.Chromosome.fasta"
```

### config/contrasts.tsv

定义差异分析的对比组，格式：
```
contrast_name,group1,group2
temp_30_vs_25,30,25
temp_35_vs_25,35,25
```

---

## 10. 依赖环境

### Conda 环境

1. **qc.yaml**（用于 FastQC、fastp、MultiQC）
   - 位置：`envs/rnaseq_envs/qc.yaml`
   - 包含：fastqc, fastp, multiqc

2. **align_hisat2.yaml**（用于 HISAT2）
   - 位置：`envs/rnaseq_envs/align_hisat2.yaml`
   - 包含：hisat2, samtools

3. **count_featurecounts.yaml**（用于 featureCounts）
   - 位置：`envs/rnaseq_envs/count_featurecounts.yaml`
   - 包含：subread（featureCounts）

4. **deseq2.yaml**（用于 DESeq2）
   - 位置：`envs/rnaseq_envs/deseq2.yaml`
   - 包含：R, bioconductor-deseq2

5. **eggnog.yaml**（用于 eggNOG-mapper）
   - 位置：`envs/rnaseq_envs/eggnog.yaml`
   - 包含：python=3.10, eggnog-mapper, diamond, hmmer, blast

6. **stringtie.yaml**（用于 StringTie、gffcompare、gffread）
   - 位置：`envs/rnaseq_envs/stringtie.yaml`
   - 包含：stringtie, gffcompare, gffread

7. **transdecoder.yaml**（用于 TransDecoder）
   - 位置：`envs/rnaseq_envs/transdecoder.yaml`
   - 包含：transdecoder, gffread

8. **hsi_env.yaml**（用于 Python 脚本）
   - 位置：`envs/hsi_env.yaml`
   - 包含：pandas 等 Python 包

---

## 11. 输出文件总结

### 质量控制
- `results/rnaseq/qc/fastqc_raw/{sample}/`：原始数据 FastQC
- `results/rnaseq/qc/fastqc_clean/{sample}/`：修剪后 FastQC
- `results/rnaseq/qc/fastp/{sample}.fastp.{html,json}`：fastp 报告
- `results/rnaseq/qc/multiqc/multiqc_report.html`：MultiQC 汇总报告
- `results/rnaseq/qc/upstream_qc_summary.tsv`：整合 MultiQC 与 featureCounts 的上游 QC 汇总表

### 比对
- `results/rnaseq/align_hisat2/{sample}.sorted.bam`：排序后的 BAM 文件
- `results/rnaseq/align_hisat2/{sample}.sorted.bam.bai`：BAM 索引

### 计数
- `results/rnaseq/counts/featurecounts_hisat2.tsv`：计数矩阵（reference）
- `results/rnaseq/counts/featurecounts_hisat2_with_novel.tsv`：计数矩阵（reference + novel）

### 差异表达
- `results/rnaseq/deseq2/normalized_counts.tsv`：标准化计数
- `results/rnaseq/deseq2/DEG_results.tsv`：差异表达基因结果

### 功能注释
- `results/rnaseq/eggnog/eggnog_annotations.tsv`：reference 基因注释
- `results/rnaseq/eggnog/novel_eggnog_annotations.tsv`：novel 基因注释
- `results/rnaseq/eggnog/eggnog_annotations_with_novel.tsv`：合并注释（reference + novel）

### Novel Pipeline
- `results/rnaseq/stringtie/{sample}.gtf`：单样本 StringTie 组装
- `results/rnaseq/stringtie/merged.gtf`：合并后的转录本集合
- `results/rnaseq/stringtie/merged.annotated.gtf`：gffcompare 标注后的 GTF
- `references/annotation/genome_novel.gtf`：提取的 novel GTF
- `references/annotation/genes_with_novel.gtf`：合并后的 GTF（reference + novel）
- `results/rnaseq/novel/novel_transcripts.pep`：TransDecoder 预测的蛋白序列
- `results/rnaseq/novel/novel_discovery_summary.tsv`：novel 基因/转录本/ORF/注释规模汇总表（**与计算解耦**：该表由 `rnaseq_novel_summary` 直接统计上述结果文件生成；目标 `rnaseq_novel_all` 不再包含此文件，避免“只想要汇总表”时触发 TransDecoder/eggNOG 等重跑。若上游三文件已存在，仅运行 `snakemake results/rnaseq/novel/novel_discovery_summary.tsv` 或 `rnaseq_novel_summary_only` 即可，只执行统计一步。）

---

## 12. 材料与方法写作要点

- **QC**：原始与修剪后数据均做 FastQC；修剪使用 fastp（质量与长度阈值见 2.2）；MultiQC 仅在 RNA-seq 模块目录（`results/rnaseq`, `logs/rnaseq`）中运行，汇总 FastQC、fastp、HISAT2 与 featureCounts 结果；`upstream_qc_summary.tsv` 提供“每样本一行”的上游 QC 汇总表，便于论文中“质量控制”小节引用。
- **比对与计数**：HISAT2（reference-guided，`--dta`）+ featureCounts（`-t exon -g gene_id`）；链特异性由 config 的 `rnaseq.stranded` 控制。
- **差异分析**：设计矩阵由样本表生成；DESeq2 按 `config/contrasts.tsv` 批量做对比，输出标准化计数与 DEG；阈值见 5.2。
- **下游**：富集分析、WGCNA、聚类与图表建议在本地 R 中完成，以 normalized_counts、DEG_results 及注释表为输入；`novel_discovery_summary.tsv` 可用于报告 novel 基因/转录本数量、获得 ORF 的转录本数量以及获得功能注释的 novel 基因数量。

---

## 13. 注意事项

1. **链特异性**：从 `config/config.yaml` 读取 `rnaseq.stranded`，影响 HISAT2 与 featureCounts。

2. **Novel Pipeline 向后兼容**：
   - 所有新规则都是独立的，不会影响现有的 `rnaseq_all` 流程
   - 原有的 `rnaseq_featurecounts_hisat2` 规则保持不变
   - 用户可以选择是否运行 novel pipeline

3. **gene_id 重复检查**：
   - 在合并 GTF 和 eggNOG 注释时，会自动检查并避免 gene_id 重复
   - 如果发现重复，会记录警告并保留 reference 的条目

4. **运行时间**：
   - HISAT2 比对：每个样本约 30-60 分钟
   - featureCounts 计数：约 4 分钟
   - DESeq2 差异分析：约 5-10 分钟
   - eggNOG-mapper 注释：约 2 小时
   - StringTie 组装：每个样本约 10 分钟
   - TransDecoder：约 30 分钟（取决于 novel transcripts 数量）

5. **资源需求**：
   - HISAT2：内存 16-32 GB，CPU 8-16 线程
   - featureCounts：内存 32 GB，CPU 8 线程
   - DESeq2：内存 16 GB，CPU 4 线程
   - eggNOG-mapper：内存 32 GB，CPU 16 线程
   - StringTie：内存 8-16 GB，CPU 4-8 线程
   - TransDecoder：内存 16 GB，CPU 8 线程

---

## 14. 软件版本

以下版本来自各步骤使用的 Conda 环境 YAML 文件（`envs/rnaseq_envs/`、`envs/r.yaml`、`envs/hsi_env.yaml`）。论文材料与方法中可引用本表。

| 步骤 / 用途 | 软件 | 版本 | 环境文件 |
|-------------|------|------|----------|
| 质量控制 | fastp | 0.24.3 | `rnaseq_envs/qc.yaml` |
| 质量控制 | FastQC | 0.12.1 | `rnaseq_envs/qc.yaml` |
| 质量控制 | MultiQC | 1.26 | `rnaseq_envs/qc.yaml` |
| 比对 | HISAT2 | 2.2.1 | `rnaseq_envs/align_hisat2.yaml` |
| 比对 | SAMtools | 1.22.1 | `rnaseq_envs/align_hisat2.yaml` |
| 比对 | HTSlib | 1.22.1 | `rnaseq_envs/align_hisat2.yaml` |
| 计数 | Subread (featureCounts) | 2.1.1 | `rnaseq_envs/count_featurecounts.yaml` |
| 差异分析 | R | 4.3.2 | `rnaseq_envs/de_r.yaml` |
| 差异分析 | Bioconductor DESeq2 | 1.40.* | `rnaseq_envs/de_r.yaml` |
| 差异分析 | Bioconductor apeglm | 1.22.* | `rnaseq_envs/de_r.yaml` |
| 差异分析 | Bioconductor tximport | 1.28.* | `rnaseq_envs/de_r.yaml` |
| 差异分析 | ggplot2 / pheatmap / tidyverse 等 | 见 de_r.yaml | `rnaseq_envs/de_r.yaml` |
| 功能注释 | eggnog-mapper | 见 env 内版本 | `rnaseq_envs/eggnog.yaml` |
| 功能注释 | DIAMOND | 见 env 内版本 | `rnaseq_envs/eggnog.yaml` |
| 功能注释 | HMMER | 见 env 内版本 | `rnaseq_envs/eggnog.yaml` |
| 功能注释 | BLAST | 见 env 内版本 | `rnaseq_envs/eggnog.yaml` |
| Novel / 组装 | StringTie | 2.2.1 | `rnaseq_envs/stringtie.yaml` |
| Novel / 组装 | gffcompare | 0.12.6 | `rnaseq_envs/stringtie.yaml` |
| Novel / 组装 | gffread | 0.12.7 | `rnaseq_envs/stringtie.yaml` |
| Novel / 蛋白预测 | TransDecoder | 5.7.1 | `rnaseq_envs/transdecoder.yaml` |
| Novel / 蛋白预测 | gffread | 0.12.7 | `rnaseq_envs/transdecoder.yaml` |
| 设计矩阵 / Novel 脚本 | Python | 3.11 | `hsi_env.yaml`（与 HSI 脚本共用） |

**说明**：

- **DESeq2**：规则中通过 `conda run -n r` 调用 R；若本机环境名为 `r` 且由 `de_r.yaml` 创建，则上表 R/DESeq2 版本适用；否则以实际 `conda list -n r` 为准。
- **eggNOG**：`eggnog.yaml` 中未固定 eggnog-mapper、DIAMOND、HMMER、BLAST 的版本，实际版本可在该环境中执行 `conda list` 查看。
- **设计矩阵**：`rnaseq_make_design` 规则未在 Snakemake 中指定 conda，表中 Python 3.11 对应 novel/eggnog 等使用 `hsi_env.yaml` 的 Python 脚本；若设计矩阵脚本单独运行于其他环境，以该环境版本为准。

