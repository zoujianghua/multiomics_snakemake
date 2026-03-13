# WGS（全基因组测序）模块文档

## 概述

WGS 模块处理全基因组测序数据，包括质量控制、BWA 比对、BAM 后处理和 GATK 变异检测。支持完整的 WGS 分析流程。

## 模块结构

### 规则文件

- `rules/wgs/ingest_samples.smk`：样本导入
- `rules/wgs/qc_fastp.smk`：质量控制（FastQC、fastp、MultiQC）
- `rules/wgs/align_bwa.smk`：BWA 比对
- `rules/wgs/postbam.smk`：BAM 后处理（去重、质量评分重校准等）
- `rules/wgs/gatk.smk`：GATK 变异检测
- `rules/wgs/wgs_all.smk`：WGS 模块的汇总入口

### 输入文件

- `config/samples_wgs.csv`：WGS 样本表，包含样本 ID、fastq1、fastq2 路径及元数据（big_region、sub_region）

### 输出目录

- `results/wgs/`：所有 WGS 相关结果
  - `clean/`：修剪后的 FASTQ 文件
  - `qc/`：质量控制结果（FastQC、fastp、MultiQC）
  - `align_bwa/`：BWA 比对结果（BAM 文件）
  - `bamstats/`：BAM 统计信息
  - `gatk/`：GATK 变异检测结果

---

## 1. 样本导入

### 1.1 样本表格式

**文件**：`config/samples_wgs.csv`

**必需列**：
- `sample_id`：样本 ID
- `fastq1`：R1 FASTQ 文件路径（支持多个文件，用分号分隔；支持目录，以 `/` 结尾）
- `fastq2`：R2 FASTQ 文件路径（格式同 fastq1）

**可选列**（元数据）：
- `big_region`：大区域
- `sub_region`：子区域

**特殊功能**：
- 如果路径以 `/` 结尾，会自动展开该目录下的所有 `*.fq.gz` 文件
- 如果路径包含通配符，会自动 glob 匹配
- R1 和 R2 文件列表必须长度一致且一一对应

### 1.2 样本字典

**规则文件**：`rules/wgs/ingest_samples.smk`

**输出**：
- `WGS` 字典：包含每个样本的 r1、r2 文件列表及元数据
- `WGS_IDS` 列表：所有样本 ID 的排序列表

**使用**：其他规则通过 `WGS[wc.sample]["r1"]` 等方式访问样本信息

---

## 2. 质量控制

### 2.1 FastQC（原始数据，可选）

**规则**：`wgs_fastqc_raw`

**输入**：
- R1 FASTQ：`WGS[sample]["r1"]`（列表）
- R2 FASTQ：`WGS[sample]["r2"]`（列表）

**输出**：
- `results/wgs/qc/fastqc_raw/{sample}/`（目录）

**功能**：对原始 FASTQ 文件进行质量评估

**配置**：
- 可通过 `config["wgs"]["fastqc_raw"]` 控制是否运行（默认 True）

### 2.2 fastp（数据修剪）

**规则**：`wgs_fastp`

**输入**：
- R1 FASTQ：`WGS[sample]["r1"]`（列表）
- R2 FASTQ：`WGS[sample]["r2"]`（列表）

**输出**：
- `results/wgs/clean/{sample}_R1.merged.fq.gz`：合并并修剪后的 R1
- `results/wgs/clean/{sample}_R2.merged.fq.gz`：合并并修剪后的 R2
- `results/wgs/qc/fastp/{sample}.fastp.html`：fastp HTML 报告
- `results/wgs/qc/fastp/{sample}.fastp.json`：fastp JSON 报告

**功能**：
- 如果样本有多个 R1/R2 文件，先合并再修剪
- 使用 fastp 进行质量修剪和接头去除

**参数**：
- `-q 20`：质量阈值
- `-u 30`：未识别碱基比例阈值
- `-n 5`：N 碱基阈值
- `-l 25`：最小读长
- `--detect_adapter_for_pe`：自动检测接头

### 2.3 FastQC（修剪后数据）

**规则**：`wgs_fastqc_clean`

**输入**：
- 修剪后的 R1/R2 FASTQ

**输出**：
- `results/wgs/qc/fastqc_clean/{sample}/`（目录）

**配置**：
- 可通过 `config["wgs"]["fastqc_clean"]` 控制是否运行（默认 True）

### 2.4 MultiQC 汇总

**规则**：`wgs_multiqc`

**输入**：
- 所有 FastQC 结果（如果运行）
- 所有 fastp 报告

**输出**：
- `results/wgs/qc/multiqc/multiqc_report.html`

**功能**：汇总所有 QC 结果到一个 HTML 报告

---

## 3. 比对

### 3.1 BWA 比对

**规则**：`wgs_align_bwa`

**输入**：
- BWA 索引：`config["references"]["fasta"]` 对应的 5 个索引文件（.amb, .ann, .bwt, .pac, .sa）
- 修剪后的 R1/R2 FASTQ

**输出**：
- `results/wgs/align_bwa/{sample}.sorted.bam`：排序后的 BAM 文件
- `results/wgs/align_bwa/{sample}.sorted.bam.bai`：BAM 索引文件

**工具**：BWA MEM + samtools

**参数**：
- `-t {threads}`：线程数（默认 32）
- `-R`：Read group 信息（从 params 读取）
- `{params.extra}`：额外参数（从 config 读取）

**Read Group 信息**：
- ID：`{sample}`
- SM：`{sample}`
- LB：`lib1`（可从 config 配置）
- PL：`ILLUMINA`（可从 config 配置）
- PU：`{sample}.1`（可从 config 配置）

**优化**：
- 使用本地临时目录（优先 SLURM_TMPDIR，其次 /scratch/$USER/$SLURM_JOB_ID，否则 TMPDIR 或 /tmp）
- 在本地盘完成排序与建索引，再原子回写到结果目录
- 避免在共享盘上创建海量临时块

**健康检查**：
- 检查 BAM 文件头部是否包含 @RG 信息

---

## 4. BAM 后处理

### 4.1 BAM 统计

**规则**：`wgs_bamstats`

**输入**：
- BAM 文件：`results/wgs/align_bwa/{sample}.sorted.bam`

**输出**：
- `results/wgs/bamstats/{sample}.stats`：samtools stats 输出
- `results/wgs/bamstats/{sample}.flagstat`：samtools flagstat 输出

**工具**：samtools stats, samtools flagstat

**功能**：生成 BAM 文件的统计信息

### 4.2 Mark Duplicates（可选）

**规则**：`wgs_markdups`（如果实现）

**输入**：
- BAM 文件

**输出**：
- 去重后的 BAM 文件

**工具**：samtools markdup 或 GATK MarkDuplicates

**功能**：标记或移除 PCR 重复

### 4.3 Base Quality Score Recalibration（可选）

**规则**：`wgs_bqsr`（如果实现）

**输入**：
- BAM 文件
- 已知变异位点（VCF）

**输出**：
- 质量评分重校准后的 BAM 文件

**工具**：GATK BaseRecalibrator + ApplyBQSR

**功能**：根据已知变异位点重新校准碱基质量分数

---

## 5. GATK 变异检测

### 5.1 HaplotypeCaller（单样本 gVCF）

**规则**：`wgs_gatk_haplotype_caller`

**输入**：
- BAM 文件：`results/wgs/align_bwa/{sample}.sorted.bam`（或后处理后的 BAM）
- 参考基因组：`config["references"]["fasta"]`
- 参考索引：`.fai` 和 `.dict` 文件

**输出**：
- `results/wgs/gatk/gvcf/{sample}.g.vcf.gz`：gVCF 文件
- `results/wgs/gatk/gvcf/{sample}.g.vcf.gz.tbi`：gVCF 索引

**工具**：GATK HaplotypeCaller

**参数**：
- `-ERC GVCF`：输出 gVCF 格式
- `-R`：参考基因组
- `-I`：输入 BAM 文件
- `-O`：输出 gVCF 文件

**功能**：
- 对每个样本单独调用变异，输出 gVCF 格式
- gVCF 包含所有位点的信息（包括非变异位点），便于后续合并

### 5.2 CombineGVCFs（合并 gVCF）

**规则**：`wgs_gatk_combine_gvcfs`

**输入**：
- 所有样本的 gVCF 文件：`results/wgs/gatk/gvcf/{sample}.g.vcf.gz`

**输出**：
- `results/wgs/gatk/cohort.g.vcf.gz`：合并后的 gVCF
- `results/wgs/gatk/cohort.g.vcf.gz.tbi`：索引

**工具**：GATK CombineGVCFs

**功能**：
- 将所有样本的 gVCF 合并为一个 cohort gVCF
- 为后续 GenotypeGVCFs 做准备

### 5.3 GenotypeGVCFs（联合基因分型）

**规则**：`wgs_gatk_genotype_gvcfs`

**输入**：
- 合并后的 gVCF：`results/wgs/gatk/cohort.g.vcf.gz`
- 参考基因组：`config["references"]["fasta"]`

**输出**：
- `results/wgs/gatk/cohort.vcf.gz`：联合基因分型后的 VCF
- `results/wgs/gatk/cohort.vcf.gz.tbi`：索引

**工具**：GATK GenotypeGVCFs

**功能**：
- 对合并后的 gVCF 进行联合基因分型
- 输出包含所有样本基因型的 VCF 文件

### 5.4 Variant Quality Score Recalibration（VQSR，可选）

**规则**：`wgs_gatk_vqsr`（如果实现）

**输入**：
- VCF 文件：`results/wgs/gatk/cohort.vcf.gz`
- 已知变异资源（如 HapMap、1000G、dbSNP）

**输出**：
- 重校准后的 VCF 文件

**工具**：GATK VariantRecalibrator + ApplyVQSR

**功能**：
- 根据已知变异资源对变异进行质量评分重校准
- 过滤低质量变异

### 5.5 Variant Filtering（硬过滤，可选）

**规则**：`wgs_gatk_filter`（如果实现）

**输入**：
- VCF 文件：`results/wgs/gatk/cohort.vcf.gz`

**输出**：
- `results/wgs/gatk/cohort.pass.vcf.gz`：过滤后的 VCF（只保留 PASS 位点）
- `results/wgs/gatk/cohort.pass.vcf.gz.tbi`：索引

**工具**：GATK VariantFiltration 或 bcftools

**功能**：
- 根据硬过滤标准（如 QD、FS、MQ、SOR、ReadPosRankSum 等）过滤变异
- 只保留 PASS 位点

**过滤标准**（示例）：
- QD < 2.0
- FS > 60.0
- MQ < 40.0
- SOR > 3.0
- ReadPosRankSum < -8.0

---

## 6. 运行方法

### 运行完整 WGS 流程

```bash
# 运行所有 WGS 相关规则
snakemake -j 32 all_wgs --use-conda

# 或分步运行
snakemake -j 16 wgs_fastp --use-conda
snakemake -j 32 wgs_align_bwa --use-conda
snakemake -j 16 wgs_gatk_haplotype_caller --use-conda
snakemake -j 8 wgs_gatk_combine_gvcfs --use-conda
snakemake -j 8 wgs_gatk_genotype_gvcfs --use-conda
```

### 只运行质量控制

```bash
snakemake -j 16 wgs_fastp wgs_multiqc --use-conda
```

### 只运行比对

```bash
snakemake -j 32 wgs_align_bwa --use-conda
```

### 只运行 GATK 变异检测

```bash
snakemake -j 16 wgs_gatk_haplotype_caller --use-conda
snakemake -j 8 wgs_gatk_combine_gvcfs --use-conda
snakemake -j 8 wgs_gatk_genotype_gvcfs --use-conda
```

---

## 7. 配置说明

### config/config.yaml

```yaml
wgs:
  fastqc_raw: true   # 是否运行原始数据 FastQC
  fastqc_clean: true # 是否运行修剪后 FastQC

align:
  bwa_extra: ""      # BWA 额外参数

references:
  fasta: "references/Mikania_micrantha.RefGenome.Chromosome.fasta"
```

### config/samples_wgs.csv

**格式示例**：
```csv
sample_id,fastq1,fastq2,big_region,sub_region
sample1,/path/to/sample1_R1.fq.gz,/path/to/sample1_R2.fq.gz,region1,sub1
sample2,/path/to/sample2_R1.fq.gz,/path/to/sample2_R2.fq.gz,region1,sub2
```

**多文件支持**：
```csv
sample_id,fastq1,fastq2
sample1,file1_R1.fq.gz;file2_R1.fq.gz,file1_R2.fq.gz;file2_R2.fq.gz
```

**目录展开**：
```csv
sample_id,fastq1,fastq2
sample1,/path/to/fastq_dir/,/path/to/fastq_dir/
```

---

## 8. 依赖环境

### Conda 环境

1. **qc.yaml**（用于 FastQC、fastp、MultiQC）
   - 位置：`envs/wgs_envs/qc.yaml`（或 `envs/rnaseq_envs/qc.yaml`）
   - 包含：fastqc, fastp, multiqc

2. **align_bwa.yaml**（用于 BWA）
   - 位置：`envs/wgs_envs/align_bwa.yaml`
   - 包含：bwa, samtools

3. **gatk.yaml**（用于 GATK）
   - 位置：`envs/wgs_envs/gatk.yaml`
   - 包含：gatk4, samtools, bcftools

---

## 9. 输出文件总结

### 质量控制
- `results/wgs/qc/fastqc_raw/{sample}/`：原始数据 FastQC（可选）
- `results/wgs/qc/fastqc_clean/{sample}/`：修剪后 FastQC（可选）
- `results/wgs/qc/fastp/{sample}.fastp.{html,json}`：fastp 报告
- `results/wgs/qc/multiqc/multiqc_report.html`：MultiQC 汇总报告

### 比对
- `results/wgs/clean/{sample}_R1.merged.fq.gz`：合并并修剪后的 R1
- `results/wgs/clean/{sample}_R2.merged.fq.gz`：合并并修剪后的 R2
- `results/wgs/align_bwa/{sample}.sorted.bam`：排序后的 BAM 文件
- `results/wgs/align_bwa/{sample}.sorted.bam.bai`：BAM 索引

### BAM 统计
- `results/wgs/bamstats/{sample}.stats`：samtools stats 输出
- `results/wgs/bamstats/{sample}.flagstat`：samtools flagstat 输出

### GATK 变异检测
- `results/wgs/gatk/gvcf/{sample}.g.vcf.gz`：单样本 gVCF
- `results/wgs/gatk/gvcf/{sample}.g.vcf.gz.tbi`：gVCF 索引
- `results/wgs/gatk/cohort.g.vcf.gz`：合并后的 cohort gVCF
- `results/wgs/gatk/cohort.vcf.gz`：联合基因分型后的 VCF
- `results/wgs/gatk/cohort.pass.vcf.gz`：过滤后的 VCF（只保留 PASS 位点，如果实现）

---

## 10. 注意事项

1. **多文件支持**：
   - 如果样本有多个 R1/R2 文件，fastp 规则会自动合并
   - 确保 R1 和 R2 文件列表长度一致且一一对应

2. **临时目录优化**：
   - BWA 比对使用本地临时目录，避免在共享盘上创建大量临时文件
   - 优先使用 SLURM_TMPDIR，其次 /scratch/$USER/$SLURM_JOB_ID

3. **Read Group 信息**：
   - BWA 比对会自动添加 Read Group 信息
   - 确保 BAM 文件头部包含 @RG 信息，否则后续 GATK 步骤可能失败

4. **资源需求**：
   - BWA 比对：内存 48 GB，CPU 32 线程
   - GATK HaplotypeCaller：内存 32-64 GB，CPU 8-16 线程
   - GATK CombineGVCFs：内存 64 GB，CPU 8 线程
   - GATK GenotypeGVCFs：内存 64 GB，CPU 8 线程

5. **运行时间**：
   - BWA 比对：每个样本约 2-4 小时（取决于数据量）
   - GATK HaplotypeCaller：每个样本约 4-8 小时
   - GATK CombineGVCFs：约 2-4 小时（取决于样本数）
   - GATK GenotypeGVCFs：约 4-8 小时（取决于样本数和位点数）

6. **VCF 格式**：
   - 所有 VCF 文件都使用 bgzip 压缩并建立 tabix 索引
   - 便于后续分析和可视化

7. **变异过滤**：
   - 建议使用 VQSR（如果资源允许）或硬过滤
   - 只保留 PASS 位点用于后续分析

---

## 11. 后续分析建议

1. **变异注释**：
   - 使用 ANNOVAR、VEP 或类似工具注释变异
   - 添加功能影响、频率信息等

2. **群体遗传分析**：
   - 计算等位基因频率
   - 进行 Hardy-Weinberg 平衡检验
   - 进行连锁不平衡分析

3. **关联分析**：
   - 与表型数据进行关联分析
   - 使用 PLINK、GCTA 等工具

4. **选择分析**：
   - 计算 Fst、π、Tajima's D 等统计量
   - 识别受选择区域

