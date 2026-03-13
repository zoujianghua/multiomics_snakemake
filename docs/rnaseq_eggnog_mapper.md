# RNAseq eggNOG-mapper 注释流程文档

## 概述

本流程使用 eggNOG-mapper v2 对薇甘菊（Mikania micrantha）的蛋白质序列进行功能注释，生成按 gene_id 汇总的注释表，用于后续 KEGG/GO 富集分析。

## 流程说明

### 1. 蛋白质序列生成

**规则**：`ref_proteins_from_gtf`（位于 `rules/references.smk`）

- **输入**：
  - GTF 文件：`references/annotation/genes.gtf`
  - 基因组 FASTA：`references/Mikania_micrantha.RefGenome.Chromosome.fasta`
- **输出**：`references/proteins_from_gtf.fa`
- **工具**：gffread
- **说明**：从 GTF 文件中提取 CDS 序列并翻译为蛋白质序列

### 2. eggNOG-mapper 注释

**规则**：`rnaseq_eggnog_raw`（位于 `rules/rnaseq/annot_eggnog.smk`）

- **输入**：`references/proteins_from_gtf.fa`
- **输出**：
  - `results/rnaseq/eggnog/mikania_emapper.emapper.annotations`（主要注释文件）
  - `results/rnaseq/eggnog/mikania_emapper.emapper.seed_orthologs`
  - `results/rnaseq/eggnog/mikania_emapper.emapper.hits`
- **工具**：eggNOG-mapper v2 (emapper.py)
- **参数**：
  - 输入类型：蛋白质序列 (`--itype proteins`)
  - 比对模式：DIAMOND (`-m diamond`)
  - CPU 线程数：16（可通过 Snakemake 的 `-j` 参数调整）
- **运行时间**：约 2 小时（根据数据量调整）

### 3. 注释结果整理

**规则**：`rnaseq_eggnog_tidy`（位于 `rules/rnaseq/annot_eggnog.smk`）

- **输入**：
  - eggNOG 注释文件：`results/rnaseq/eggnog/mikania_emapper.emapper.annotations`
  - GTF 文件：`references/annotation/genes.gtf`（用于 transcript -> gene 映射）
- **输出**：`results/rnaseq/eggnog/eggnog_annotations.tsv`
- **脚本**：`scripts/rnaseq/tidy_eggnog_annotations.py`
- **功能**：
  - 从 GTF 文件中提取 transcript_id -> gene_id 映射
  - 将 eggNOG 注释从 transcript 级别聚合到 gene 级别
  - 提取并整理以下字段：
    - `gene_id`：基因 ID（与 featureCounts 结果中的 gene_id 格式一致，如 `gene:Mm01G000001`）
    - `kegg_ko`：KEGG KO 编号（多个值用分号分隔）
    - `kegg_pathway`：KEGG 通路（多个值用分号分隔）
    - `go_terms`：GO 术语（多个值用分号分隔）
    - `description`：功能描述

## 运行方法

### 在 HPC 上运行完整流程

```bash
# 运行 eggNOG 注释流程（包括蛋白质序列生成、注释和整理）
snakemake -j 16 rnaseq_eggnog_all --use-conda

# 或者只运行注释整理步骤（如果已经完成 eggNOG-mapper 注释）
snakemake -j 4 rnaseq_eggnog_tidy --use-conda
```

### 检查运行状态

```bash
# 查看规则依赖关系（dry-run）
snakemake -j 16 rnaseq_eggnog_all --use-conda --dry-run

# 查看规则依赖图
snakemake -j 16 rnaseq_eggnog_all --use-conda --dag | dot -Tpng > eggnog_dag.png
```

## 输出文件

### 主要输出文件

**`results/rnaseq/eggnog/eggnog_annotations.tsv`**

这是最终整理好的注释表，格式如下：

| gene_id | kegg_ko | kegg_pathway | go_terms | description |
|---------|---------|--------------|----------|-------------|
| gene:Mm01G000001 | K12345 | ko01234;ko05678 | GO:0001234;GO:0005678 | predicted protein |
| gene:Mm01G000002 | K23456 | ko02345 | GO:0002345 | hypothetical protein |

**重要说明**：
- `gene_id` 列与 `results/rnaseq/counts/featurecounts_hisat2.tsv` 中的 `Geneid` 列格式完全一致
- 缺失值用空字符串表示
- 多个值用分号（`;`）分隔
- 文件编码为 UTF-8，无 BOM

### 中间文件

- `references/proteins_from_gtf.fa`：从 GTF 生成的蛋白质序列 FASTA
- `results/rnaseq/eggnog/mikania_emapper.emapper.annotations`：eggNOG-mapper 原始注释文件
- `results/rnaseq/eggnog/mikania_emapper.emapper.seed_orthologs`：种子直系同源物
- `results/rnaseq/eggnog/mikania_emapper.emapper.hits`：比对结果

## 下载到本地

运行完成后，从 HPC 下载以下文件到本地 Windows 项目：

```bash
# 在本地 Windows 上使用 scp 或类似工具
scp user@hpc:/public/home/zoujianghua/multiomics_project/results/rnaseq/eggnog/eggnog_annotations.tsv \
    D:/projects/multiomics/data/annotations/
```

**文件路径**：
- HPC：`/public/home/zoujianghua/multiomics_project/results/rnaseq/eggnog/eggnog_annotations.tsv`
- 本地：`D:/projects/multiomics/data/annotations/eggnog_annotations.tsv`

## 依赖环境

### Conda 环境

1. **annotation.yaml**（用于 gffread）
   - 位置：`envs/annotation.yaml`
   - 包含：gffread

2. **eggnog.yaml**（用于 eggNOG-mapper）
   - 位置：`envs/rnaseq_envs/eggnog.yaml`
   - 包含：python=3.10, eggnog-mapper, diamond, hmmer, blast

3. **hsi_env.yaml**（用于整理脚本）
   - 位置：`envs/hsi_env.yaml`
   - 包含：pandas 等 Python 包

## 注意事项

1. **gene_id 格式一致性**：
   - 输出的 `gene_id` 必须与 featureCounts 结果中的 `Geneid` 列格式完全一致
   - 当前格式为：`gene:Mm01G000001`（带 `gene:` 前缀）

2. **transcript -> gene 映射**：
   - gffread 生成的蛋白质序列 ID 是 transcript_id
   - 整理脚本会从 GTF 文件中提取 transcript_id -> gene_id 映射
   - 如果一个基因有多个转录本，注释会被聚合到 gene 级别

3. **运行时间**：
   - eggNOG-mapper 注释步骤可能需要较长时间（约 2 小时，取决于数据量）
   - 建议在后台运行或使用作业调度系统（如 SLURM）

4. **资源需求**：
   - 内存：32 GB（eggNOG-mapper 步骤）
   - CPU：16 线程（可调整）
   - 磁盘空间：约 1-2 GB（用于中间文件和结果）

5. **错误处理**：
   - 如果某个步骤失败，检查对应的日志文件：
     - `logs/ref/ref_proteins_from_gtf.log`
     - `logs/rnaseq/eggnog/emapper_raw.log`
     - `logs/rnaseq/eggnog/tidy_annotations.log`

## 故障排除

### 问题 1：无法映射 transcript 到 gene

**症状**：整理脚本报错 "No transcripts could be mapped to gene_id"

**解决**：
- 检查 GTF 文件格式是否正确
- 确认 GTF 文件中包含 `transcript_id` 和 `gene_id` 属性
- 检查 eggNOG 注释文件中的 query 列格式

### 问题 2：eggNOG-mapper 运行失败

**症状**：emapper.py 报错或超时

**解决**：
- 检查网络连接（如果使用在线数据库）
- 增加运行时间限制（修改 `runtime` 资源）
- 检查 conda 环境是否正确安装

### 问题 3：输出文件为空

**症状**：输出文件存在但为空

**解决**：
- 检查输入文件是否正确
- 查看日志文件了解详细错误信息
- 确认所有依赖规则已成功运行

## 后续使用

生成的 `eggnog_annotations.tsv` 文件可以用于：

1. **KEGG 富集分析**：
   - 使用 `kegg_ko` 或 `kegg_pathway` 列构建基因集
   - 与差异表达基因结果进行富集分析

2. **GO 富集分析**：
   - 使用 `go_terms` 列构建 GO 基因集
   - 进行 GO 功能富集分析

3. **功能注释**：
   - 使用 `description` 列查看基因功能描述
   - 结合差异表达结果进行功能解释

## 相关文件

- 规则文件：`rules/rnaseq/annot_eggnog.smk`
- 整理脚本：`scripts/rnaseq/tidy_eggnog_annotations.py`
- Conda 环境：`envs/rnaseq_envs/eggnog.yaml`
- 参考基因组规则：`rules/references.smk`

