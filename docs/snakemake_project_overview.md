# Snakemake 多组学项目结构概览

## 项目整体结构

本项目是一个基于 Snakemake 的多组学数据分析流水线，包含以下主要模块：

- **HSI（高光谱成像）模块**：处理高光谱图像数据，提取特征并进行机器学习分类
- **rnaseq（转录组学）模块**：处理 RNA-seq 数据，包括 QC、比对、计数和差异表达分析
- **wgs（全基因组测序）模块**：处理 WGS 数据（比对、GATK 等）
- **references（参考基因组）模块**：生成各种参考基因组索引和注释文件

（代谢组模块已从本项目中移除，未使用。）

## 详细文档

各模块的详细文档请参考：

- **[HSI 模块文档](hsi_module.md)**：高光谱成像数据处理完整流程
- **[HSI 生理状态标签定义指南](hsi_physiological_states_guide.md)**：如何利用转录组和代谢组数据定义生理状态标签（本地下游分析指导）
- **[HSI 数据下游分析指南](hsi_downstream_analysis_guide.md)**：`image_features.tsv` 生成流程说明和本地 R 分析指导
- **[RNAseq 模块文档](rnaseq_module.md)**：转录组学分析完整流程
- **[WGS 模块文档](wgs_module.md)**：全基因组测序分析完整流程
- **[RNAseq Novel Pipeline 文档](rnaseq_novel_pipeline.md)**：Novel transcripts/genes 发现与注释流程
- **[RNAseq eggNOG-mapper 文档](rnaseq_eggnog_mapper.md)**：功能注释流程

---

## 模块详细说明

### 1. HSI 模块

#### 功能概述
高光谱成像数据处理，从原始图像提取光谱特征，进行图像级、叶片级和 patch 级的机器学习分类。

#### 主要规则文件
- `rules/hsi.smk`：图像预处理、清洗、聚合和可视化
- `rules/hsi_ml.smk`：图像级别的机器学习分类
- `rules/leaf.smk`：叶片级别的机器学习分类
- `rules/patch.smk`：patch 级别的机器学习分类（包括深度学习模型）
- `rules/hsi_all.smk`：HSI 模块的汇总入口

#### 主要流程

**1.1 图像预处理流程**（`rules/hsi.smk`）
- **输入**：`config/samples_hsi.csv`
- **规则链**：
  - `hsi_preprocess`：从原始 HSI 数据提取图像特征
    - 输入：`config/samples_hsi.csv`
    - 输出：`results/hsi/raw_image_features.tsv`
  - `hsi_clean_and_indices`：清洗异常样本并计算植被指数
    - 输入：`results/hsi/raw_image_features.tsv`
    - 输出：`results/hsi/clean_image_features.tsv` → `results/hsi/image_features.tsv`
  - `hsi_aggregate`：聚合图像特征到 session 级别
    - 输入：`results/hsi/image_features.tsv`
    - 输出：`results/hsi/session_features.tsv`、`results/hsi/delta_hsi.tsv`、`results/hsi/resilience_metrics.tsv`
  - `hsi_quicklook`：生成可视化图表
    - 输出：`results/hsi/plot_ndvi_timeseries.png` 等

**1.2 叶片级预处理**（`rules/hsi.smk`）
- `hsi_preprocess_leaf`：提取叶片特征
  - 输入：`config/samples_hsi.csv`、`results/hsi/cube`
  - 输出：`results/hsi/raw_leaf_features.tsv`
- `hsi_leaf_clean_and_indices`：清洗并计算指数
  - 输出：`results/hsi/leaf_features.tsv`

**1.3 图像级机器学习**（`rules/hsi_ml.smk`）
- **输入**：`results/hsi/image_features.tsv`
- **模型**：RF、SVM、KNN、LR、XGB、PLS、LDA、1D-CNN
- **输出目录**：`results/hsi/ml/{model}/`
- **汇总规则**：`hsi_ml_merge` → `results/hsi/ml/all_models_summary.csv`

**1.4 叶片级机器学习**（`rules/leaf.smk`）
- **输入**：`results/hsi/leaf_features.tsv`
- **模型**：RF、SVM、KNN、LR、XGB、PLS、LDA、1D-CNN
- **输出目录**：`results/hsi/leaf/{model}/`
- **汇总规则**：`leaf_merge` → `results/hsi/leaf/all_models_summary.csv`

**1.5 Patch 级机器学习**（`rules/patch.smk`）
- **流程**：
  - `hsi_patch_index_base`：生成通用几何索引（不含 target/split）
    - 输出：`results/hsi/ml/patch_index_base.tsv`
  - `hsi_patch_cubes`：生成 patch 集合文件（一 cube 一个文件，减少 I/O 开销）
    - 输出：`results/hsi/patch_cubes/` 目录 + `results/hsi/ml/patch_index_base_cubes.tsv`
  - `hsi_patch_split`：按 patch_target 分层划分数据集
    - 输出：`results/hsi/ml/split_{patch_target}_seed42.tsv`
  - `hsi_patch_make_index`：构建任务特定 patch 索引
    - 输出：`results/hsi/ml/patch_index_{patch_target}_seed42.tsv`
  - `hsi_patch_features_raw`：提取原始 patch 特征（传统 ML 用）
    - 输出：`results/hsi/raw_patch_features_{patch_target}.tsv`
  - `hsi_patch_clean_and_indices`：清洗并计算指数
    - 输出：`results/hsi/patch_features_{patch_target}.tsv`
  - `hsi_patch_features_omics`：生成多组学关联分析用特征表
    - 输出：`results/hsi/patch_features_omics.tsv`
- **模型**：
  - 传统 ML：RF、SVM、KNN、LR、XGB、PLS、LDA
  - 深度学习：
    - **1D-CNN**（光谱序列）：带残差结构的轻量网络，支持 `forward_features()` 提取 embedding
      - 结构：初始 Conv1d (1→32) + 3 个 ResidualBlock1D (32→64→128→128) + GlobalAveragePooling1D + 特征层 (128→embedding_dim) + 分类头
      - 输入：光谱特征序列（从 TSV 读取）
      - 输出：`best_model.pt` + `metrics.tsv`
    - **2D-CNN**（空间+光谱）：小型 ResNet 风格，支持 embedding 导出
      - 结构：1×1 Conv 光谱降维（B 波段→32 通道）+ 3 个 ResidualBlock2D (32→64→128→128) + GlobalAveragePooling2D + 特征层 (128→embedding_dim) + 分类头
      - 输入：Patch 张量 [B, C, H, W]，C 为光谱波段数
      - 输出：`best_model.pt` + `metrics.tsv` + `patch_embeddings_2d.tsv`
    - **3D-CNN**（空间+光谱）：轻量 3D ResNet 风格，支持 embedding 导出
      - 结构：1×1×k 3D Conv 光谱降维 + 3 个 ResidualBlock3D (16→32→64→64) + GlobalAveragePooling3D + 特征层 (64→embedding_dim) + 分类头
      - 输入：Patch 张量 [B, 1, D, H, W]，D 为光谱维度
      - 输出：`best_model.pt` + `metrics.tsv` + `patch_embeddings_3d.tsv`
- **深度学习改进**（2024 年更新）：
  - **DataLoader 稳定性**：
    - 支持 `--num-workers` 参数（默认 0），避免多进程 OOM 问题
    - 支持 `--no-pin-memory` 参数（默认使用 `pin_memory=True`）
    - 所有 DataLoader（train/val/test）统一使用这些参数，不再硬编码
    - 3D CNN 默认 batch_size 调整为 8（更保守，避免显存不足）
  - **模型结构升级**：
    - 所有 CNN 模型采用残差网络结构（ResidualBlock），提升表示能力
    - 1D-CNN：通道数逐步增加（32→64→128），带 shortcut 连接
    - 2D-CNN：先用 1×1 Conv 做光谱通道降维，再堆叠残差块
    - 3D-CNN：先用 1×1×k 3D Conv 做光谱维降维，再堆叠 3D 残差块
  - **Embedding 导出功能**：
    - 所有 CNN 模型都实现了 `forward_features()` 方法，返回倒数第二层的 embedding
    - 2D/3D CNN 训练结束后自动导出 patch 级 embedding：
      - 2D CNN：`results/hsi/patch/{patch_target}/2dcnn/patch_embeddings_2d.tsv`
      - 3D CNN：`results/hsi/patch/{patch_target}/3dcnn/patch_embeddings_3d.tsv`
    - Embedding TSV 格式：
      - 列：`patch_id`, `source_sample_id`, `split`, `target`, `emb_0`, `emb_1`, ..., `emb_{D-1}`
      - 包含所有 patch（train + test），按原始顺序排列
      - 便于在本地 R 多组学项目中进行关联分析
  - **数据集改进**：
    - `HSIPatchDataset` 支持 `split="train"`, `"test"`, `"all"` 三种模式
    - 维护 `meta_df`，包含 patch 元数据信息，确保与 DataLoader 顺序一致
- **输出目录**：`results/hsi/patch/{patch_target}/{model}/`
- **输出文件**：
  - 传统 ML：`best_model.pt`（如适用）+ `metrics.tsv` + 其他评估文件
  - 深度学习：`best_model.pt` + `metrics.tsv` + `history.tsv` + `patch_embeddings_{2d|3d}.tsv`（2D/3D CNN）
- **汇总规则**：`patch_merge` → `results/hsi/patch/{patch_target}/all_models_summary.csv`

**1.6 生理状态分类**（`rules/physiological.smk`）
- **用途**：基于多组学定义的 `physiological_state` 标签，重复 patch 分类流程，与 phase 分类效果对比
- **约束**：不修改 `image_features.tsv`，避免触发整个项目重跑
- **流程**：
  - `physio_image_meta`：image_features + mapping → `image_features_with_physio.tsv`（临时）
  - `physio_patch_features`：patch_features_{patch_target} + mapping → `patch_features_with_physiological.tsv`
  - `physio_split`：按 physiological_state 在 image 级别分层划分
  - `physio_patch_index`：复用 base_cubes + image_meta + split → `patch_index_physiological_state_seed42.tsv`
  - 模型训练：复用 patch.smk 中所有模型脚本（RF/SVM/.../2D/3D CNN）
- **输出**：`results/hsi/patch/physiological_state/all_models_summary.csv`
- **汇总规则**：`physio_patch_merge`

---

### 2. RNAseq 模块

#### 功能概述
RNA-seq 数据处理流水线，包括质量控制、比对、计数和差异表达分析。

#### 主要规则文件
- `rules/rnaseq/ingest_samples.smk`：样本导入
- `rules/rnaseq/qc_fastp.smk`：质量控制（FastQC、fastp、MultiQC）
- `rules/rnaseq/align_hisat2.smk`：HISAT2 比对
- `rules/rnaseq/quantify_featurecounts.smk`：featureCounts 计数
- `rules/rnaseq/de_deseq2.smk`：DESeq2 差异表达分析
- `rules/rnaseq/rnaseq_all.smk`：RNAseq 模块的汇总入口

#### 主要流程

**2.1 样本导入**（`rules/rnaseq/ingest_samples.smk`）
- **输入**：`config/samples_rnaseq.csv`
- **输出**：定义 `RNASEQ` 字典和 `RNASEQ_IDS` 列表，包含样本的 fastq1、fastq2 路径及元数据（temp、phase、time、group）

**2.2 质量控制**（`rules/rnaseq/qc_fastp.smk`）
- **规则链**：
  - `rnaseq_fastqc_raw`：原始 FASTQ 的 FastQC
    - 输出：`results/rnaseq/qc/fastqc_raw/{sample}/`
  - `rnaseq_fastp`：修剪和质控
    - 输入：原始 FASTQ（从 `RNASEQ` 字典读取）
    - 输出：`results/rnaseq/clean/{sample}_R1.fastq.gz`、`results/rnaseq/clean/{sample}_R2.fastq.gz`、fastp 报告
  - `rnaseq_fastqc_clean`：修剪后 FASTQ 的 FastQC
    - 输出：`results/rnaseq/qc/fastqc_clean/{sample}/`
  - `rnaseq_multiqc`：汇总所有 QC 结果
    - 输出：`results/rnaseq/qc/multiqc/multiqc_report.html`

**2.3 比对**（`rules/rnaseq/align_hisat2.smk`）
- **规则**：`rnaseq_align_hisat2`
- **输入**：
  - HISAT2 索引：`config["references"]["hisat2_index"]`（8 个 .ht2 文件）
  - 修剪后的 FASTQ：`results/rnaseq/clean/{sample}_R1.fastq.gz`、`results/rnaseq/clean/{sample}_R2.fastq.gz`
- **输出**：`results/rnaseq/align_hisat2/{sample}.sorted.bam` 及索引文件

**2.4 计数**（`rules/rnaseq/quantify_featurecounts.smk`）
- **规则**：`rnaseq_featurecounts_hisat2`
- **输入**：
  - GTF 文件：`config["references"]["gtf"]`
  - 所有样本的 BAM 文件
- **输出**：
  - `results/rnaseq/counts/featurecounts_hisat2.tsv`（计数矩阵）
  - `results/rnaseq/counts/featurecounts_hisat2.tsv.summary`（汇总统计）

**2.5 差异表达分析**（`rules/rnaseq/de_deseq2.smk`）
- **规则链**：
  - `rnaseq_make_design`：生成设计矩阵
    - 输入：`config/samples_rnaseq.csv`、`scripts/make_rnaseq_design.py`
    - 输出：`config/rnaseq_design.tsv`
  - `rnaseq_deseq2`：DESeq2 差异分析
    - 输入：
      - 计数矩阵：`results/rnaseq/counts/featurecounts_hisat2.tsv`
      - 设计矩阵：`config/rnaseq_design.tsv`
      - 对比文件：`config/contrasts.tsv`
      - R 脚本：`scripts/rnaseq/rnaseq_deseq2_contrasts.R`
    - 输出：
      - `results/rnaseq/deseq2/normalized_counts.tsv`（标准化计数）
      - `results/rnaseq/deseq2/DEG_results.tsv`（差异表达基因结果）

**2.6 已实现功能总结**
- ✅ 样本导入和元数据管理
- ✅ 原始数据 QC（FastQC）
- ✅ 数据修剪（fastp）
- ✅ 修剪后 QC（FastQC）
- ✅ QC 汇总（MultiQC）
- ✅ HISAT2 比对
- ✅ featureCounts 计数
- ✅ DESeq2 标准差异分析（按 phase、time 分组，按 temp 进行对比）

**2.7 计划中的功能**（在 `rules/rnaseq/rnaseq_all.smk` 中引用但可能尚未实现）
- `qc_plots.smk`：QC 图表（PCA、样本距离热图等）
  - 计划输出：`results/rnaseq/qc/libsize_bar.png`、`results/rnaseq/qc/pca_vst.png`、`results/rnaseq/qc/sample_distance_heatmap.png`
- `timeseries_msigpro.smk`：时间序列分析（maSigPro）
  - 计划输出：`results/rnaseq/masigpro_fit.rds`、`results/rnaseq/masigpro_sig_summary.tsv`
- `fgsea.smk`：GSEA 富集分析
  - 计划输出：`results/rnaseq/fgsea/.done`

---

### 3. WGS 模块

#### 功能概述
全基因组测序数据处理，包括比对、GATK 变异检测等。

#### 主要规则文件
- `rules/wgs/ingest_samples.smk`：样本导入
- `rules/wgs/qc_fastp.smk`：质量控制（与 rnaseq 类似）
- `rules/wgs/align_bwa.smk`：BWA 比对
- `rules/wgs/postbam.smk`：BAM 后处理（去重、质量评分重校准等）
- `rules/wgs/gatk.smk`：GATK 变异检测
- `rules/wgs/wgs_all.smk`：WGS 模块的汇总入口

#### 主要流程
- 样本导入 → QC（fastp）→ BWA 比对 → BAM 后处理 → GATK 变异检测

---

### 4. References 模块

#### 功能概述
生成各种参考基因组索引和注释文件，供其他模块使用。

#### 主要规则文件
- `rules/references.smk`

#### 主要规则
- `ref_faidx`：samtools faidx（生成 .fai 索引）
- `ref_dict`：GATK CreateSequenceDictionary（生成 .dict）
- `ref_bwa_index`：BWA index（生成 .amb、.ann、.bwt、.pac、.sa）
- `ref_hisat2_index`：HISAT2 index（生成 8 个 .ht2 文件）
- `ref_gff3_to_gtf`：GFF3 转 GTF（使用 gffread）

---

## 后续扩展计划

### RNAseq 模块扩展接口

#### 1. 功能注释（eggNOG）
- **计划位置**：`rules/rnaseq/annotate_eggnog.smk`
- **输入**：`results/rnaseq/deseq2/DEG_results.tsv`（或所有基因列表）
- **输出**：`results/rnaseq/annotation/eggnog_annotations.tsv`
- **脚本接口**：`scripts/rnaseq/run_eggnog.py` 或直接调用 `emapper.py`
- **依赖**：eggNOG-mapper 环境

#### 2. KEGG/GO 富集分析
- **计划位置**：`rules/rnaseq/enrichment_kegg_go.smk`
- **输入**：
  - 差异基因列表：`results/rnaseq/deseq2/DEG_results.tsv`
  - 基因注释：`results/rnaseq/annotation/eggnog_annotations.tsv`（或 KEGG/GO 映射表）
- **输出**：
  - `results/rnaseq/enrichment/kegg_enrichment.tsv`
  - `results/rnaseq/enrichment/go_enrichment.tsv`
  - 可视化图表（如 `results/rnaseq/enrichment/kegg_dotplot.pdf`）
- **脚本接口**：`scripts/rnaseq/enrichment_kegg_go.R`（使用 clusterProfiler 或类似工具）
- **依赖**：R 环境（clusterProfiler、org.*.eg.db 等）

#### 3. 时间模式分析
- **计划位置**：`rules/rnaseq/timeseries_msigpro.smk`（已在 `rnaseq_all.smk` 中引用）
- **输入**：
  - 标准化计数：`results/rnaseq/deseq2/normalized_counts.tsv`
  - 设计矩阵：`config/rnaseq_design.tsv`（需包含 time 列）
- **输出**：
  - `results/rnaseq/masigpro_fit.rds`（maSigPro 拟合对象）
  - `results/rnaseq/masigpro_sig_summary.tsv`（显著时间模式基因）
- **脚本接口**：`scripts/rnaseq/timeseries_msigpro.R`

#### 4. QC 图表扩展
- **计划位置**：`rules/rnaseq/qc_plots.smk`（已在 `rnaseq_all.smk` 中引用）
- **输入**：
  - 标准化计数：`results/rnaseq/deseq2/normalized_counts.tsv`
  - 设计矩阵：`config/rnaseq_design.tsv`
- **输出**：
  - `results/rnaseq/qc/libsize_bar.png`（文库大小条形图）
  - `results/rnaseq/qc/pca_vst.png`（VST 转换后的 PCA）
  - `results/rnaseq/qc/sample_distance_heatmap.png`（样本距离热图）
- **脚本接口**：`scripts/rnaseq/qc_plots.R`

#### 5. GSEA 富集分析
- **计划位置**：`rules/rnaseq/fgsea.smk`（已在 `rnaseq_all.smk` 中引用）
- **输入**：
  - 差异分析结果：`results/rnaseq/deseq2/DEG_results.tsv`（包含 log2FoldChange 和 padj）
  - 基因集文件（如 MSigDB 格式）
- **输出**：`results/rnaseq/fgsea/.done`（标识文件，实际结果在 fgsea 目录下）
- **脚本接口**：`scripts/rnaseq/fgsea.R`（使用 fgsea 包）

---

## 文件路径约定

### 输入文件
- 样本表：`config/samples_{module}.csv`
- 设计矩阵：`config/{module}_design.tsv`（如 `config/rnaseq_design.tsv`）
- 对比文件：`config/contrasts.tsv`（所有模块共享）
- 参考基因组：从 `config["references"]` 读取

### 输出文件
- 结果目录：`results/{module}/`
- 日志目录：`logs/{module}/`
- 临时文件：通常不追踪，或放在 `results/{module}/tmp/`

### 脚本文件
- Python 脚本：`scripts/{module}/`
- R 脚本：`scripts/{module}/` 或 `scripts/`（如 `scripts/rnaseq_deseq2.R`）

---

## 注意事项

1. **未实现的规则**：`rules/rnaseq/rnaseq_all.smk` 中引用了 `qc_plots.smk`、`timeseries_msigpro.smk`、`fgsea.smk`，但这些文件可能尚未创建，需要在后续开发中实现。

2. **环境依赖**：各模块使用不同的 conda 环境，定义在 `envs/` 目录下，通过 `conda:` 指令指定。

3. **资源配置**：各规则通过 `resources:` 指定内存、运行时间、GPU 等资源需求。

4. **模块独立性**：各模块相对独立，主要通过共享 `config/contrasts.tsv` 和统一的输出目录结构来保持一致性。

---

## 总结

本项目目前已经实现了：
- ✅ HSI 模块的完整流程（预处理、ML、DL）
  - ✅ 图像级、叶片级、patch 级特征提取和机器学习
  - ✅ Patch 级深度学习（1D/2D/3D CNN）采用残差网络结构
    - ✅ 1D-CNN：光谱序列模型，带残差结构，支持 `forward_features()` 提取 embedding
    - ✅ 2D-CNN：空间+光谱模型，小型 ResNet 风格，自动导出 `patch_embeddings_2d.tsv`
    - ✅ 3D-CNN：空间+光谱模型，轻量 3D ResNet 风格，自动导出 `patch_embeddings_3d.tsv`
  - ✅ DataLoader 稳定性改进：支持 `--num-workers` 和 `--no-pin-memory` 参数，避免 worker 被 OOM 杀掉
  - ✅ Patch 级 embedding 导出功能，包含完整的元数据（patch_id, source_sample_id, split, target），便于多组学关联分析
  - ✅ `HSIPatchDataset` 支持 `split="all"` 模式，用于导出所有 patch 的 embedding
- ✅ RNAseq 模块的基础流程（QC、比对、计数、DESeq2）
  - ✅ Novel transcripts/genes 发现与注释流程（StringTie + eggNOG）
  - ✅ eggNOG-mapper 功能注释（reference + novel）
  - ✅ 基于合并 GTF（reference + novel）的重新计数
- ✅ Metabo 模块的基础流程（导入、PCA、差异分析）
- ✅ WGS 模块的基础流程
- ✅ References 模块的索引生成

**待扩展**：
- ⏳ RNAseq 模块的 KEGG/GO 富集分析
- ⏳ RNAseq 模块的时间序列分析（maSigPro）
- ⏳ RNAseq 模块的 QC 图表
- ⏳ RNAseq 模块的 GSEA 分析
- ⏳ 模块间整合分析（HSI patch embeddings + 代谢组/转录组）
  - 当前已具备基础：HSI patch embeddings 已导出，包含完整的元数据信息

