# RNAseq Novel Transcripts/Genes 发现与注释流程文档

## 概述

本流程用于发现和注释薇甘菊（Mikania micrantha）转录组中的新转录本和新基因，包括：

1. **StringTie 组装**：基于已有 BAM 文件进行 reference-guided 转录本组装
2. **gffcompare 标注**：与参考 GTF 比较，识别 novel transcripts
3. **Novel GTF 提取**：提取并规范化 novel transcripts/genes
4. **蛋白预测**：使用 TransDecoder 预测 novel 蛋白序列
5. **功能注释**：使用 eggNOG-mapper 注释 novel 蛋白
6. **合并与重新计数**：合并 reference + novel 的 GTF 和注释，并重新计数

## 流程说明

### 1. StringTie 单样本组装

**规则**：`rnaseq_stringtie_assemble`（位于 `rules/rnaseq/novel_transcripts.smk`）

- **输入**：
  - BAM 文件：`results/rnaseq/align_hisat2/{sample}.sorted.bam`
  - 参考 GTF：`references/annotation/genes.gtf`
- **输出**：`results/rnaseq/stringtie/{sample}.gtf`
- **工具**：stringtie（reference-guided 模式）
- **参数**：
  - `-G`：指定参考 GTF
  - `-l`：样本标签（使用 sample_id）
  - `-p`：线程数（4）

### 2. StringTie Merge 合并转录本

**规则**：`rnaseq_stringtie_merge`（位于 `rules/rnaseq/novel_transcripts.smk`）

- **输入**：
  - 所有样本的 GTF：`results/rnaseq/stringtie/{sample}.gtf`
  - 参考 GTF：`references/annotation/genes.gtf`
- **输出**：`results/rnaseq/stringtie/merged.gtf`
- **工具**：stringtie --merge
- **功能**：合并所有样本的转录本，生成统一的转录本集合

### 3. gffcompare 标注新转录本

**规则**：`rnaseq_gffcompare`（位于 `rules/rnaseq/novel_transcripts.smk`）

- **输入**：
  - merged GTF：`results/rnaseq/stringtie/merged.gtf`
  - 参考 GTF：`references/annotation/genes.gtf`
- **输出**：
  - `results/rnaseq/stringtie/merged.annotated.gtf`（标注后的 GTF）
  - `results/rnaseq/stringtie/merged.tracking`（追踪文件）
  - `results/rnaseq/stringtie/merged.stats`（统计信息）
- **工具**：gffcompare
- **功能**：将 merged GTF 与参考 GTF 比较，为每个转录本标注 class_code：
  - `u`：完全落在基因间区的全新转录本（默认提取）
  - `x`：反义链上的转录本
  - `i`：内含子内的转录本
  - 其他：参考 gffcompare 文档

### 4. 提取 Novel Transcripts/Genes

**规则**：`rnaseq_novel_extract`（位于 `rules/rnaseq/novel_transcripts.smk`）

- **输入**：
  - annotated GTF：`results/rnaseq/stringtie/merged.annotated.gtf`
  - 脚本：`scripts/rnaseq/extract_novel_gtf.py`
- **输出**：`references/annotation/genome_novel.gtf`
- **脚本**：`scripts/rnaseq/extract_novel_gtf.py`
- **功能**：
  - 根据 class_code 筛选 novel 转录本（默认 `u`，可通过 config 配置）
  - 规范化 gene_id 和 transcript_id 命名：
    - gene_id：`gene:novel000001`
    - transcript_id：`transcript:novel000001.1`
  - 确保输出 GTF 格式符合 featureCounts 要求

**配置**：
- 在 `config/config.yaml` 中设置 `rnaseq.novel_class_codes`（默认 `["u"]`）
- 可扩展为 `["u", "x", "i"]` 等

### 5. gffread 提取 Novel Transcripts 序列

**规则**：`rnaseq_novel_gffread_fasta`（位于 `rules/rnaseq/novel_transcripts.smk`）

- **输入**：
  - novel GTF：`references/annotation/genome_novel.gtf`
  - 基因组 FASTA：`references/Mikania_micrantha.RefGenome.Chromosome.fasta`
- **输出**：`results/rnaseq/novel/novel_transcripts.fa`
- **工具**：gffread
- **功能**：从基因组序列中提取 novel transcripts 的 cDNA 序列

### 6. TransDecoder 预测 Novel 蛋白

**规则链**：

- **rnaseq_novel_transdecoder_longorfs**：识别长开放阅读框（ORFs）
  - 输入：`results/rnaseq/novel/novel_transcripts.fa`
  - 输出：`results/rnaseq/novel/novel_transcripts.fa.transdecoder_dir/longest_orfs.pep`
  - 工具：TransDecoder.LongOrfs
  - 参数：`-m 100`（最小 ORF 长度）

- **rnaseq_novel_transdecoder_predict**：预测编码序列（CDS）
  - 输入：
    - transcripts FASTA：`results/rnaseq/novel/novel_transcripts.fa`
    - ORF peptides：`results/rnaseq/novel/novel_transcripts.fa.transdecoder_dir/longest_orfs.pep`
  - 输出：
    - `results/rnaseq/novel/novel_transcripts.pep`（预测的蛋白序列）
    - `results/rnaseq/novel/novel_transcripts.gff3`（CDS 注释）
  - 工具：TransDecoder.Predict

### 7. eggNOG-mapper 注释 Novel 蛋白

**规则链**：

- **rnaseq_novel_eggnog_raw**：运行 eggNOG-mapper
  - 输入：`results/rnaseq/novel/novel_transcripts.pep`
  - 输出：
    - `results/rnaseq/eggnog/novel_emapper.emapper.annotations`
    - `results/rnaseq/eggnog/novel_emapper.emapper.seed_orthologs`
    - `results/rnaseq/eggnog/novel_emapper.emapper.hits`
  - 工具：eggNOG-mapper v2 (emapper.py)
  - 参数：与 reference 蛋白注释保持一致（DIAMOND 模式）

- **rnaseq_novel_eggnog_tidy**：整理注释结果
  - 输入：
    - eggNOG 注释：`results/rnaseq/eggnog/novel_emapper.emapper.annotations`
    - novel GTF：`references/annotation/genome_novel.gtf`
  - 输出：`results/rnaseq/eggnog/novel_eggnog_annotations.tsv`
  - 脚本：`scripts/rnaseq/tidy_eggnog_annotations.py`
  - 功能：按 gene_id 聚合注释信息（KEGG、GO、描述等）

### 8. 合并 Reference + Novel 的 eggNOG 注释

**规则**：`rnaseq_eggnog_merge_novel`（位于 `rules/rnaseq/novel_transcripts.smk`）

- **输入**：
  - reference 注释：`results/rnaseq/eggnog/eggnog_annotations.tsv`
  - novel 注释：`results/rnaseq/eggnog/novel_eggnog_annotations.tsv`
  - 脚本：`scripts/rnaseq/merge_eggnog_annotations.py`
- **输出**：`results/rnaseq/eggnog/eggnog_annotations_with_novel.tsv`
- **功能**：
  - 合并两个注释表
  - 检查并避免 gene_id 重复（如果 novel 的 gene_id 与 reference 重复，保留 reference 的注释并记录警告）

### 9. 合并 Reference + Novel 的 GTF

**规则**：`ref_genes_with_novel_gtf`（位于 `rules/rnaseq/novel_transcripts.smk`）

- **输入**：
  - reference GTF：`references/annotation/genes.gtf`
  - novel GTF：`references/annotation/genome_novel.gtf`
  - 脚本：`scripts/rnaseq/merge_gtf.py`
- **输出**：`references/annotation/genes_with_novel.gtf`
- **功能**：
  - 合并两个 GTF 文件
  - 检查并避免 gene_id 重复（如果 novel 的 gene_id 与 reference 重复，跳过 novel 的条目）
  - 确保输出 GTF 格式正确，可用于 featureCounts

### 10. 基于合并 GTF 重新计数

**规则**：`rnaseq_featurecounts_hisat2_with_novel`（位于 `rules/rnaseq/quantify_featurecounts.smk`）

- **输入**：
  - 合并 GTF：`references/annotation/genes_with_novel.gtf`
  - 所有样本的 BAM 文件
- **输出**：
  - `results/rnaseq/counts/featurecounts_hisat2_with_novel.tsv`（计数矩阵）
  - `results/rnaseq/counts/featurecounts_hisat2_with_novel.tsv.summary`（汇总统计）
- **工具**：featureCounts
- **参数**：与原规则完全一致（`-t exon -g gene_id -p -B -C -s {stranded} -Q 10`）
- **说明**：这是原 `rnaseq_featurecounts_hisat2` 规则的平行版本，唯一区别是使用合并后的 GTF

## 运行方法

### 在 HPC 上运行完整 Novel Pipeline

```bash
# 运行完整的 novel pipeline（包括所有步骤）
snakemake -j 16 rnaseq_novel_all --use-conda

# 或者先 dry-run 检查依赖关系
snakemake -j 16 rnaseq_novel_all --use-conda --dry-run
```

### 分步运行

如果只想运行部分步骤：

```bash
# 只运行 StringTie 组装和 merge
snakemake -j 16 rnaseq_stringtie_merge --use-conda

# 只运行 gffcompare
snakemake -j 4 rnaseq_gffcompare --use-conda

# 只提取 novel GTF
snakemake -j 4 rnaseq_novel_extract --use-conda

# 只运行 TransDecoder
snakemake -j 8 rnaseq_novel_transdecoder_predict --use-conda

# 只运行 novel eggNOG 注释
snakemake -j 16 rnaseq_novel_eggnog_tidy --use-conda

# 只合并 GTF 和注释
snakemake -j 4 ref_genes_with_novel_gtf rnaseq_eggnog_merge_novel --use-conda

# 只重新计数
snakemake -j 8 rnaseq_featurecounts_hisat2_with_novel --use-conda
```

## 输出文件

### 主要输出文件

1. **Novel GTF**：
   - `references/annotation/genome_novel.gtf`：提取的 novel transcripts/genes
   - `references/annotation/genes_with_novel.gtf`：合并后的 GTF（reference + novel）

2. **Novel 蛋白序列**：
   - `results/rnaseq/novel/novel_transcripts.pep`：TransDecoder 预测的蛋白序列

3. **Novel eggNOG 注释**：
   - `results/rnaseq/eggnog/novel_eggnog_annotations.tsv`：novel 蛋白的注释表
   - `results/rnaseq/eggnog/eggnog_annotations_with_novel.tsv`：合并后的注释表（reference + novel）

4. **重新计数结果**：
   - `results/rnaseq/counts/featurecounts_hisat2_with_novel.tsv`：基于合并 GTF 的计数矩阵
   - `results/rnaseq/counts/featurecounts_hisat2_with_novel.tsv.summary`：汇总统计

### 中间文件

- `results/rnaseq/stringtie/{sample}.gtf`：每个样本的 StringTie 组装结果
- `results/rnaseq/stringtie/merged.gtf`：合并后的转录本集合
- `results/rnaseq/stringtie/merged.annotated.gtf`：gffcompare 标注后的 GTF
- `results/rnaseq/novel/novel_transcripts.fa`：novel transcripts 的 cDNA 序列
- `results/rnaseq/novel/novel_transcripts.fa.transdecoder_dir/`：TransDecoder 中间文件

## 配置说明

### config/config.yaml

在 `rnaseq` 部分添加了以下配置：

```yaml
rnaseq:
    # ... 其他配置 ...
    novel_class_codes: ["u"]  # gffcompare class_code 筛选
```

**class_code 说明**：
- `u`：完全落在基因间区的全新转录本（默认）
- `x`：反义链上的转录本
- `i`：内含子内的转录本
- 其他：参考 gffcompare 文档

**示例**：如果要同时提取 `u`、`x`、`i` 三种类型：
```yaml
rnaseq:
    novel_class_codes: ["u", "x", "i"]
```

## 依赖环境

### Conda 环境

1. **stringtie.yaml**（用于 StringTie、gffcompare、gffread）
   - 位置：`envs/rnaseq_envs/stringtie.yaml`
   - 包含：stringtie, gffcompare, gffread

2. **transdecoder.yaml**（用于 TransDecoder）
   - 位置：`envs/rnaseq_envs/transdecoder.yaml`
   - 包含：transdecoder, gffread

3. **eggnog.yaml**（用于 eggNOG-mapper）
   - 位置：`envs/rnaseq_envs/eggnog.yaml`
   - 包含：python=3.10, eggnog-mapper, diamond, hmmer, blast

4. **hsi_env.yaml**（用于 Python 脚本）
   - 位置：`envs/hsi_env.yaml`
   - 包含：pandas 等 Python 包

5. **count_featurecounts.yaml**（用于 featureCounts）
   - 位置：`envs/rnaseq_envs/count_featurecounts.yaml`
   - 包含：subread（featureCounts）

## 注意事项

1. **向后兼容**：
   - 所有新规则都是独立的，不会影响现有的 `rnaseq_all` 流程
   - 原有的 `rnaseq_featurecounts_hisat2` 规则保持不变
   - 用户可以选择是否运行 novel pipeline

2. **gene_id 重复检查**：
   - 在合并 GTF 和 eggNOG 注释时，会自动检查并避免 gene_id 重复
   - 如果发现重复，会记录警告并保留 reference 的条目

3. **运行时间**：
   - StringTie 单样本组装：每个样本约 10 分钟
   - StringTie merge：约 30 分钟
   - gffcompare：约 10 分钟
   - TransDecoder：约 30 分钟（取决于 novel transcripts 数量）
   - eggNOG-mapper：约 2 小时（取决于 novel 蛋白数量）
   - featureCounts 重新计数：约 4 分钟

4. **资源需求**：
   - StringTie：内存 8-16 GB，CPU 4-8 线程
   - TransDecoder：内存 16 GB，CPU 8 线程
   - eggNOG-mapper：内存 32 GB，CPU 16 线程
   - featureCounts：内存 32 GB，CPU 8 线程

5. **文件命名规范**：
   - Novel gene_id：`gene:novel000001`（6 位数字，前导零）
   - Novel transcript_id：`transcript:novel000001.1`（gene_id + transcript 编号）

## 故障排除

### 问题 1：StringTie 组装失败

**症状**：某个样本的 GTF 文件为空

**解决**：
- 检查对应的 BAM 文件是否存在且非空
- 检查参考 GTF 文件格式是否正确
- 查看日志文件：`logs/rnaseq/stringtie/{sample}.log`

### 问题 2：gffcompare 没有找到 novel transcripts

**症状**：merged.annotated.gtf 中没有 class_code="u" 的条目

**解决**：
- 检查 merged.gtf 是否包含转录本
- 检查参考 GTF 是否覆盖了所有区域
- 考虑扩展 `novel_class_codes` 配置（例如添加 `x`、`i`）

### 问题 3：TransDecoder 输出文件找不到

**症状**：`novel_transcripts.pep` 不存在

**解决**：
- 检查 `novel_transcripts.fa` 是否存在且非空
- 检查 TransDecoder 工作目录：`results/rnaseq/novel/novel_transcripts.fa.transdecoder_dir/`
- 查看日志文件：`logs/rnaseq/novel/transdecoder_predict.log`

### 问题 4：gene_id 重复警告

**症状**：合并时出现大量 gene_id 重复警告

**解决**：
- 这是正常情况，说明某些 novel transcripts 可能与参考基因重叠
- 系统会自动保留 reference 的条目，跳过 novel 的重复条目
- 如果需要保留 novel 的条目，需要手动修改合并脚本

## 后续使用

### 下载结果到本地

运行完成后，从 HPC 下载以下文件到本地 Windows 项目：

```bash
# Novel GTF
scp user@hpc:/public/home/zoujianghua/multiomics_project/references/annotation/genes_with_novel.gtf \
    D:/projects/multiomics/data/annotations/

# 合并后的 eggNOG 注释
scp user@hpc:/public/home/zoujianghua/multiomics_project/results/rnaseq/eggnog/eggnog_annotations_with_novel.tsv \
    D:/projects/multiomics/data/annotations/

# 重新计数结果
scp user@hpc:/public/home/zoujianghua/multiomics_project/results/rnaseq/counts/featurecounts_hisat2_with_novel.tsv \
    D:/projects/multiomics/data/rnaseq/counts/
```

### 在本地 R 中使用

1. **计数矩阵**：
   - 使用 `featurecounts_hisat2_with_novel.tsv` 替代原来的计数矩阵
   - 包含 reference + novel 的所有基因

2. **功能注释**：
   - 使用 `eggnog_annotations_with_novel.tsv` 替代原来的注释表
   - 包含 reference + novel 的所有基因注释

3. **差异表达分析**：
   - 在本地 R 中使用 DESeq2 进行差异表达分析
   - 使用合并后的计数矩阵和注释表

## 相关文件

- 规则文件：`rules/rnaseq/novel_transcripts.smk`
- 计数规则：`rules/rnaseq/quantify_featurecounts.smk`
- 辅助脚本：
  - `scripts/rnaseq/extract_novel_gtf.py`
  - `scripts/rnaseq/merge_eggnog_annotations.py`
  - `scripts/rnaseq/merge_gtf.py`
  - `scripts/rnaseq/tidy_eggnog_annotations.py`（复用）
- Conda 环境：
  - `envs/rnaseq_envs/stringtie.yaml`
  - `envs/rnaseq_envs/transdecoder.yaml`
  - `envs/rnaseq_envs/eggnog.yaml`
- 配置文件：`config/config.yaml`

