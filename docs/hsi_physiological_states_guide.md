# HSI 项目：基于多组学数据定义生理状态标签指南

## 项目背景

### 当前问题

1. **分类准确率低**：按 `phase`（23 类）分类时，由于一些时间点接近的样本生理状态相似，导致识别准确率低
2. **分类过于粗糙**：按 `phase_core`（5 类）分类时，虽然类别数少，但丢失了时间维度的信息
3. **需要生物学意义的标签**：希望利用转录组和代谢组数据定义更符合生物学意义的生理状态标签

### 解决方案

通过整合转录组和代谢组数据，将原始的 `phase` 标签细化为 `physiological_state` 标签，使得：
- 生理状态相似的样本归为一类（即使它们属于不同的 phase）
- 生理状态不同的样本分开（即使它们属于同一个 phase）
- 最终分类数在 5-23 之间，平衡了准确率和生物学意义

---

## 数据准备

### 1. 确认数据文件

确保你已经有了以下文件（从本地下游分析获得）：

#### 转录组数据
- **标准化 counts 矩阵**：`normalized_counts.tsv` 或类似文件
  - 行：基因 ID
  - 列：样本 ID（例如 `CK25_T2h_R1`, `HT35_T3d_R2` 等）
- **差异分析结果**：各对比组的 DEG 结果
- **设计矩阵**：`rnaseq_design.tsv`，包含 `sample`, `phase`, `phase_core`, `temp`, `time` 等列
- **WGCNA 结果**（如果有）：模块表达矩阵或模块特征基因

#### 代谢组数据
- **标准化强度矩阵**：`meta_intensity_pos.xlsx` 和 `meta_intensity_neg.xlsx` 或类似文件
  - 行：代谢物 ID
  - 列：样本 ID
- **差异分析结果**：各对比组的差异代谢物结果
- **设计矩阵**：`metabo_design_pos.tsv` 和 `metabo_design_neg.tsv`

#### HSI 数据
- **图像特征表**：`results/hsi/image_features.tsv`
  - 包含 `sample_id`, `phase`, `phase_core`, `temp`, `time` 等列

---

## 方法 1：基于差异特征整合定义生理状态（推荐）

### 思路

1. 从转录组和代谢组的差异分析结果中提取关键特征（差异基因、差异代谢物）
2. 整合这些特征，构建每个样本的特征向量
3. 使用聚类方法（K-means 或层次聚类）将样本分组
4. 在每个 `phase` 内部，根据聚类结果进一步细分，生成 `physiological_state`

### 步骤 1：提取差异特征

#### R 代码示例

```r
# ==================== 加载包 ====================
library(tidyverse)
library(readxl)

# ==================== 读取数据 ====================
# 转录组：标准化 counts
rnaseq_norm <- read_tsv("normalized_counts.tsv") %>%
  column_to_rownames(var = "gene_id")

# 转录组：设计矩阵
rnaseq_design <- read_tsv("rnaseq_design.tsv")

# 转录组：差异分析结果（假设有多个对比组的结果）
# 例如：stress_35_vs_control, stress_10_vs_control, recovery_vs_stress 等
deg_files <- list.files("rnaseq_diff_results/", pattern = "DEG_results.*\\.tsv", full.names = TRUE)

# 代谢组：标准化强度（POS 模式）
metabo_pos <- read_excel("meta_intensity_pos.xlsx") %>%
  column_to_rownames(var = "metabolite_id")

# 代谢组：设计矩阵
metabo_design_pos <- read_tsv("metabo_design_pos.tsv")

# ==================== 提取差异基因 ====================
# 合并所有差异分析结果，提取显著差异基因
all_degs <- tibble()
for (deg_file in deg_files) {
  deg_df <- read_tsv(deg_file) %>%
    filter(padj < 0.05, abs(log2FoldChange) > 1) %>%  # 调整阈值
    select(gene_id, log2FoldChange, padj)
  all_degs <- bind_rows(all_degs, deg_df)
}

# 去重，保留每个基因的最大 log2FC（绝对值）
key_genes <- all_degs %>%
  group_by(gene_id) %>%
  summarise(max_abs_lfc = max(abs(log2FoldChange)), .groups = "drop") %>%
  arrange(desc(max_abs_lfc)) %>%
  head(500)  # 选择 top 500 差异基因

# 提取这些基因的表达矩阵
rnaseq_features <- rnaseq_norm %>%
  rownames_to_column("gene_id") %>%
  filter(gene_id %in% key_genes$gene_id) %>%
  column_to_rownames("gene_id")

# ==================== 提取差异代谢物 ====================
# 假设你有差异代谢物列表（从代谢组差异分析获得）
# 如果没有，可以基于方差或 PCA 载荷选择
metabo_var <- apply(metabo_pos, 1, var, na.rm = TRUE)
key_metabolites <- names(sort(metabo_var, decreasing = TRUE)[1:200])  # top 200 高方差代谢物

metabo_features <- metabo_pos[key_metabolites, ]

# ==================== 整合特征 ====================
# 确保样本 ID 一致
common_samples <- intersect(colnames(rnaseq_features), colnames(metabo_features))

rnaseq_features_subset <- rnaseq_features[, common_samples]
metabo_features_subset <- metabo_features[, common_samples]

# 标准化（Z-score）
rnaseq_scaled <- t(scale(t(rnaseq_features_subset)))
metabo_scaled <- t(scale(t(metabo_features_subset)))

# 合并特征矩阵
combined_features <- rbind(
  rnaseq_scaled,
  metabo_scaled
)

# 转置：行为样本，列为特征
combined_features <- t(combined_features)

# 保存中间结果
write_tsv(
  as.data.frame(combined_features) %>% rownames_to_column("sample_id"),
  "combined_omics_features.tsv"
)
```

### 步骤 2：降维和聚类

#### R 代码示例

```r
# ==================== 降维（PCA） ====================
# 如果特征数太多，先做 PCA 降维
pca_result <- prcomp(combined_features, center = TRUE, scale. = TRUE)

# 选择解释方差的前 N 个主成分（累计解释 > 80%）
pca_var <- summary(pca_result)$importance[2, ]
cum_var <- cumsum(pca_var)
n_components <- min(which(cum_var >= 0.8), 50)  # 最多 50 个主成分

pca_features <- pca_result$x[, 1:n_components]

# ==================== 确定聚类数 ====================
# 方法 1：基于 phase 数量估计
phase_info <- rnaseq_design %>%
  select(sample, phase, phase_core) %>%
  filter(sample %in% rownames(combined_features))

n_phases <- length(unique(phase_info$phase))
n_phase_cores <- length(unique(phase_info$phase_core))

# 允许每个 phase 内部有 1-3 个生理状态
n_clusters <- min(n_phases * 2, nrow(combined_features) / 5)
n_clusters <- max(5, min(n_clusters, 20))  # 限制在 5-20 之间

cat("建议聚类数:", n_clusters, "\n")

# 方法 2：使用 elbow method 或 silhouette 方法
library(cluster)
library(factoextra)

# Elbow method
fviz_nbclust(pca_features, kmeans, method = "wss", k.max = 20)

# Silhouette method
fviz_nbclust(pca_features, kmeans, method = "silhouette", k.max = 20)

# ==================== K-means 聚类 ====================
set.seed(42)
kmeans_result <- kmeans(pca_features, centers = n_clusters, nstart = 25, iter.max = 100)

# 提取聚类标签
cluster_labels <- kmeans_result$cluster

# ==================== 生成生理状态标签 ====================
# 将聚类结果与 phase 信息结合
physio_states <- tibble(
  sample_id = rownames(combined_features),
  cluster_id = cluster_labels
) %>%
  left_join(
    phase_info %>% select(sample, phase, phase_core),
    by = c("sample_id" = "sample")
  )

# 策略 1：在每个 phase_core 内部，用 cluster_id 区分
physio_states <- physio_states %>%
  group_by(phase_core, cluster_id) %>%
  mutate(
    physio_state_id = cur_group_id(),
    physiological_state = paste0(phase_core, "_physio_", cluster_id)
  ) %>%
  ungroup()

# 策略 2：如果希望更简洁的标签，可以基于 cluster 的主要 phase 特征命名
# 例如：如果 cluster_1 主要包含 stress_35_T2h 和 stress_35_T6h，可以命名为 "stress_35_early"
physio_states <- physio_states %>%
  group_by(cluster_id) %>%
  mutate(
    dominant_phase_core = names(sort(table(phase_core), decreasing = TRUE))[1],
    dominant_time = names(sort(table(time), decreasing = TRUE))[1]
  ) %>%
  ungroup() %>%
  mutate(
    physiological_state = paste0(dominant_phase_core, "_", cluster_id)
  )

# 保存结果
write_tsv(
  physio_states %>% select(sample_id, physiological_state, cluster_id, phase, phase_core),
  "physiological_states.tsv"
)
```

### 步骤 3：可视化验证

#### R 代码示例

```r
# ==================== PCA 可视化 ====================
library(ggplot2)

pca_df <- as.data.frame(pca_result$x[, 1:2]) %>%
  rownames_to_column("sample_id") %>%
  left_join(physio_states, by = "sample_id")

p1 <- ggplot(pca_df, aes(x = PC1, y = PC2, color = physiological_state, shape = phase_core)) +
  geom_point(size = 3, alpha = 0.7) +
  theme_bw() +
  labs(
    title = "PCA: Physiological States",
    x = paste0("PC1 (", round(pca_var[1] * 100, 1), "%)"),
    y = paste0("PC2 (", round(pca_var[2] * 100, 1), "%)")
  ) +
  theme(legend.position = "right")

ggsave("pca_physiological_states.pdf", p1, width = 10, height = 8)

# ==================== 聚类热图 ====================
library(pheatmap)

# 选择每个 cluster 的代表性样本
sample_order <- physio_states %>%
  arrange(physiological_state, cluster_id) %>%
  pull(sample_id)

heatmap_data <- combined_features[sample_order, ]

# 添加注释
annotation_col <- physio_states %>%
  select(sample_id, physiological_state, phase_core, cluster_id) %>%
  column_to_rownames("sample_id") %>%
  .[sample_order, ]

pheatmap(
  t(heatmap_data),
  annotation_col = annotation_col,
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  show_colnames = FALSE,
  main = "Heatmap: Combined Omics Features by Physiological State",
  filename = "heatmap_physiological_states.pdf",
  width = 12,
  height = 8
)

# ==================== Phase vs Physiological State 交叉表 ====================
cross_table <- physio_states %>%
  count(phase, physiological_state) %>%
  pivot_wider(names_from = physiological_state, values_from = n, values_fill = 0)

write_tsv(cross_table, "phase_physio_cross_table.tsv")

# 可视化
p2 <- ggplot(physio_states, aes(x = phase, fill = physiological_state)) +
  geom_bar(position = "stack") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Phase vs Physiological State Distribution")

ggsave("phase_physio_distribution.pdf", p2, width = 12, height = 6)
```

---

## 方法 2：基于 WGCNA 模块定义生理状态

如果你已经完成了 WGCNA 分析，可以利用模块表达来定义生理状态。

### 思路

1. 使用 WGCNA 模块的表达值（module eigengenes）作为特征
2. 结合代谢组的关键代谢物或代谢通路
3. 聚类定义生理状态

### R 代码示例

```r
# ==================== 读取 WGCNA 结果 ====================
# 假设你有模块表达矩阵
module_eigengenes <- read_tsv("wgcn a_module_eigengenes.tsv") %>%
  column_to_rownames("sample_id")

# 选择与胁迫相关的关键模块（例如：与 stress 相关的模块）
# 这需要根据你的 WGCNA 结果确定
key_modules <- c("MEturquoise", "MEblue", "MEbrown")  # 示例，需要根据实际情况调整

module_features <- module_eigengenes[, key_modules]

# ==================== 结合代谢组数据 ====================
# 选择关键代谢通路或代谢物
# 例如：从 KEGG 富集分析中选择关键通路的代表代谢物
key_metabolites <- c("metabolite_1", "metabolite_2", ...)  # 需要根据实际情况
metabo_features <- metabo_pos[key_metabolites, ]

# 整合
common_samples <- intersect(rownames(module_features), colnames(metabo_features))
combined_features <- cbind(
  module_features[common_samples, ],
  t(metabo_features[, common_samples])
)

# 标准化
combined_features_scaled <- scale(combined_features)

# ==================== 聚类 ====================
# 后续步骤与方法 1 相同
set.seed(42)
kmeans_result <- kmeans(combined_features_scaled, centers = n_clusters, nstart = 25)

# ... 生成生理状态标签（同方法 1）
```

---

## 方法 3：基于时间序列模式定义生理状态

如果你的实验设计包含时间序列，可以利用时间动态模式来定义生理状态。

### 思路

1. 对每个样本，提取其在不同时间点的表达/代谢模式
2. 使用时间序列聚类方法（例如：基于动态时间规整 DTW）
3. 将具有相似时间模式的样本归为一类

### R 代码示例

```r
# ==================== 构建时间序列特征 ====================
# 假设你有多个时间点的数据
time_points <- c("2h", "6h", "1d", "3d", "7d")

# 对每个样本，提取其在各时间点的平均表达/代谢水平
# 这里以转录组为例
rnaseq_design_with_time <- rnaseq_design %>%
  mutate(time_point = factor(time, levels = time_points))

# 按 phase_core 和时间点聚合
time_series_features <- rnaseq_norm %>%
  rownames_to_column("gene_id") %>%
  pivot_longer(-gene_id, names_to = "sample_id", values_to = "expression") %>%
  left_join(
    rnaseq_design_with_time %>% select(sample, phase_core, time_point),
    by = c("sample_id" = "sample")
  ) %>%
  group_by(gene_id, phase_core, time_point) %>%
  summarise(mean_expression = mean(expression, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = time_point, values_from = mean_expression) %>%
  unite("phase_time", phase_core, sep = "_") %>%
  column_to_rownames("phase_time")

# 选择高方差基因
gene_var <- apply(time_series_features, 1, var, na.rm = TRUE)
key_genes <- names(sort(gene_var, decreasing = TRUE)[1:200])
time_series_features_subset <- time_series_features[key_genes, ]

# ==================== 时间序列聚类 ====================
library(dtwclust)

# 转置：行为时间点，列为 phase_time 组合
time_series_matrix <- t(time_series_features_subset)

# 使用 DTW 距离进行层次聚类
dtw_dist <- proxy::dist(time_series_matrix, method = "DTW")
hc_result <- hclust(dtw_dist, method = "ward.D2")

# 切割树得到聚类
n_clusters <- 10  # 根据实际情况调整
cluster_labels <- cutree(hc_result, k = n_clusters)

# 生成生理状态标签
physio_states <- tibble(
  phase_time = names(cluster_labels),
  cluster_id = cluster_labels
) %>%
  separate(phase_time, into = c("phase_core", "time"), sep = "_", extra = "merge") %>%
  mutate(
    physiological_state = paste0(phase_core, "_physio_", cluster_id)
  )
```

---

## 整合到 HSI 项目

### 步骤 1：将生理状态标签添加到 image_features.tsv

#### Python 代码示例

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将生理状态标签添加到 image_features.tsv
"""

import pandas as pd
from pathlib import Path

# 读取生理状态标签表
physio_states = pd.read_csv("physiological_states.tsv", sep="\t")

# 读取 HSI image_features
image_features = pd.read_csv("results/hsi/image_features.tsv", sep="\t")

# 合并
image_features_updated = image_features.merge(
    physio_states[["sample_id", "physiological_state"]],
    on="sample_id",
    how="left"
)

# 检查匹配情况
matched = image_features_updated["physiological_state"].notna().sum()
total = len(image_features_updated)
print(f"匹配成功: {matched}/{total} ({matched/total*100:.1f}%)")

# 保存
image_features_updated.to_csv(
    "results/hsi/image_features_with_physio.tsv",
    sep="\t",
    index=False
)
print("已保存: results/hsi/image_features_with_physio.tsv")
```

### 步骤 2：更新 config.yaml

```yaml
hsi:
  # ... 其他配置 ...
  patch_target: "physiological_state"  # 改为使用生理状态标签
```

### 步骤 3：重新运行 HSI patch 分类流程

```bash
# 使用新的 image_features 和 physiological_state 标签
snakemake -j 8 results/hsi/patch_features_physiological_state.tsv
snakemake -j 8 results/hsi/patch/physiological_state/2dcnn_physiological_state/best_model.pt
```

---

## 验证和优化

### 1. 检查生理状态分布

```r
# 检查每个 physiological_state 的样本数
physio_states %>%
  count(physiological_state) %>%
  arrange(desc(n))

# 检查 phase 到 physiological_state 的映射
physio_states %>%
  count(phase, physiological_state) %>%
  pivot_wider(names_from = physiological_state, values_from = n, values_fill = 0)
```

### 2. 评估分类性能

运行 HSI 分类后，比较使用 `phase` 和 `physiological_state` 的分类准确率：

```python
# 比较不同标签的分类性能
import pandas as pd

# 读取分类结果
results_phase = pd.read_csv("results/hsi/patch/phase/all_models_summary.csv")
results_physio = pd.read_csv("results/hsi/patch/physiological_state/all_models_summary.csv")

print("Phase 分类结果:")
print(results_phase[["model", "test_accuracy", "test_f1_weighted"]])

print("\nPhysiological State 分类结果:")
print(results_physio[["model", "test_accuracy", "test_f1_weighted"]])
```

### 3. 调整聚类参数

如果分类效果不理想，可以调整：

- **聚类数**：增加或减少 `n_clusters`
- **特征选择**：调整差异基因/代谢物的数量或选择标准
- **降维方法**：尝试 UMAP 或 t-SNE 替代 PCA
- **聚类方法**：尝试层次聚类或 DBSCAN

---

## 常见问题

### Q1: 如何选择合适的聚类数？

**A**: 
- 基于生物学意义：每个 `phase_core` 内部可能有 1-3 个不同的生理状态
- 基于样本数：确保每个聚类至少有 3-5 个样本
- 使用 elbow method 或 silhouette method 辅助选择
- 最终根据 HSI 分类性能确定

### Q2: 如果某些样本没有匹配到生理状态怎么办？

**A**: 
- 检查样本 ID 是否一致（大小写、格式等）
- 对于未匹配的样本，可以：
  - 使用其 `phase` 作为 `physiological_state`
  - 或使用 `phase_core` 作为 `physiological_state`

### Q3: 如何解释生理状态标签的生物学意义？

**A**: 
- 查看每个生理状态的主要 `phase` 组成
- 分析每个生理状态的差异基因/代谢物
- 进行功能富集分析（KEGG/GO），了解每个生理状态的生物学功能

### Q4: 转录组和代谢组数据权重如何平衡？

**A**: 
- 方法 1：等权重（直接合并特征矩阵）
- 方法 2：根据数据质量调整权重（例如：如果转录组数据质量更好，可以增加转录组特征的权重）
- 方法 3：分别降维后再合并（例如：转录组 PCA + 代谢组 PCA）

---

## 总结

通过整合转录组和代谢组数据定义生理状态标签，可以：

1. **提高分类准确性**：将生理状态相似的样本归为一类
2. **增强生物学意义**：标签基于真实的生物学状态，而非仅实验设计
3. **平衡类别数**：在 5-23 之间找到合适的平衡点

关键步骤：
1. 提取差异特征（差异基因、差异代谢物）
2. 整合特征并降维
3. 聚类定义生理状态
4. 验证和优化
5. 整合到 HSI 项目

---

## 参考文件

- HSI 图像特征表：`results/hsi/image_features.tsv`
- 转录组设计矩阵：`config/rnaseq_design.tsv`
- 代谢组设计矩阵：`config/metabo_design_pos.tsv`, `config/metabo_design_neg.tsv`
- 配置文件：`config/config.yaml`

---

**最后更新**：2025-01-XX

