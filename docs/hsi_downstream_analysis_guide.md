# HSI 数据下游分析指南

## 概述

本文档说明 `image_features.tsv` 的生成流程，并提供在本地使用 R 进行 HSI 数据下游分析的详细指导。虽然项目已包含 `quicklook.py` 进行基础可视化，但在本地使用 R 可以更方便地进行可视化的调试和定制化分析。

---

## 1. `image_features.tsv` 生成流程

### 1.1 数据流程概览

```
原始 HSI 数据 (cube_npz)
    ↓
hsi_preprocess (scripts/hsi/preprocess.py)
    ↓
raw_image_features.tsv
    ↓
hsi_clean_and_indices
    ├─ clean_image_features.py (清洗异常样本)
    └─ add_indices.py (计算植被指数)
    ↓
image_features.tsv (最终输出)
```

### 1.2 详细步骤说明

#### 步骤 1：图像预处理 (`hsi_preprocess`)

**脚本**：`scripts/hsi/preprocess.py`

**功能**：
- 从原始 HSI cube 文件（`.npz`）读取高光谱数据
- 反射率标定（使用白板/黑板参考）
- 叶片分割（使用 NDVI/NDRE 阈值或 K-means 聚类）
- ROI（感兴趣区域）提取
- 计算 ROI 内的平均光谱
- 光谱平滑（Savitzky-Golay 滤波）

**输出**：`results/hsi/raw_image_features.tsv`

**主要列**：
- `sample_id`：样本 ID
- `phase`, `phase_core`, `temp`, `time`：实验设计信息
- `roi_area`：ROI 面积（像素数）
- `R_*`：光谱反射率列（例如 `R_450`, `R_550`, `R_670` 等，波长范围约 400-1000 nm）
- `R800_med`：800 nm 处的反射率中位数（用于质量控制）
- `spec_npz`：保存的光谱文件路径
- `mask_png`：分割掩膜图像路径

#### 步骤 2：数据清洗 (`clean_image_features.py`)

**功能**：
- 过滤异常样本（基于 `R800_med` 阈值和 Z-score）
- 移除质量不合格的图像

**参数**：
- `--min-r800`：最小 R800 阈值（默认 0.04）
- `--max-z`：最大 Z-score 阈值（默认 3.0）

**输出**：`results/hsi/clean_image_features.tsv`

#### 步骤 3：计算植被指数 (`add_indices.py`)

**脚本**：`scripts/hsi/add_indices.py`

**功能**：计算 400-1000 nm 范围内的经典植被指数

**计算的指数类别**：

1. **结构/叶绿素指数**：
   - `ndvi`：归一化植被指数
   - `gndvi`：绿光归一化植被指数
   - `rdvi`：重归一化植被指数
   - `dvi`：差值植被指数
   - `sr`：简单比值指数
   - `msr`：修正简单比值指数
   - `savi`：土壤调节植被指数
   - `osavi`：优化土壤调节植被指数
   - `msavi`, `msavi2`：修正土壤调节植被指数
   - `wdrvi`：宽动态范围植被指数

2. **红边/氮素相关指数**：
   - `ndre1`, `ndre2`, `ndre3`：归一化红边指数
   - `gi`：绿度指数
   - `gci`：绿度叶绿素指数
   - `cigreen`：叶绿素指数（绿光）
   - `cirededge`：叶绿素指数（红边）
   - `rep`：红边位置（四点法）
   - `rep_d1`：红边位置（一阶导数法）

3. **色素/黄化指数**：
   - `pri`：光化学反射指数
   - `ari1`, `ari2`：花青素反射指数
   - `npci`：归一化色素叶绿素指数
   - `sipi`：结构不敏感色素指数
   - `psri`：植物衰老反射指数
   - `cri1`, `cri2`：类胡萝卜素反射指数

4. **综合指数**：
   - `tcari`：转换叶绿素吸收反射指数
   - `mcari`：修正叶绿素吸收反射指数
   - `tcari_osavi`：TCARI/OSAVI 比值
   - `mcari2`, `mtvi2`：修正指数变体

5. **水分/结构指数**：
   - `wi` / `wbi`：水分波段指数（R900/R970）

**输出**：`results/hsi/image_features.tsv`（最终文件）

---

## 2. `image_features.tsv` 数据结构

### 2.1 列结构

`image_features.tsv` 包含以下类型的列：

#### 元数据列
- `sample_id`：样本 ID（例如：`CK25_T2h_R1`）
- `phase`：精细 phase 标签（例如：`control_25_T2h`, `stress_35_T3d`）
- `phase_core`：核心 phase 标签（例如：`control_25`, `stress_35`, `recovery_from_35`）
- `temp`：温度（例如：`25`, `35`, `10`）
- `time`：时间点（例如：`2h`, `6h`, `1d`, `3d`, `7d`）
- `time_h`：时间（小时，数值）
- `replicate`：重复号（例如：`R1`, `R2`, `R3`）
- `session_id`：会话 ID（用于聚合分析）
- `roi_area`：ROI 面积（像素数）
- `R800_med`：800 nm 反射率中位数（质量控制）

#### 光谱列
- `R_<nm>`：各波长的反射率（例如：`R_450`, `R_550`, `R_670`, `R_800` 等）
- 波长范围：约 400-1000 nm
- 列数：取决于原始数据的波段数（通常 100-300 个波段）

#### 植被指数列
- 上述所有计算的植被指数（`ndvi`, `gndvi`, `pri`, `ari`, `rep`, `rep_d1` 等）

### 2.2 数据读取示例

```r
library(tidyverse)

# 读取 image_features.tsv
hsi_data <- read_tsv("results/hsi/image_features.tsv")

# 查看数据结构
glimpse(hsi_data)

# 查看列名
colnames(hsi_data)

# 识别光谱列
spec_cols <- colnames(hsi_data)[grepl("^R_\\d+", colnames(hsi_data))]
cat("光谱列数量:", length(spec_cols), "\n")

# 识别植被指数列
vi_cols <- c("ndvi", "gndvi", "pri", "ari", "rep", "rep_d1", 
             "ndre1", "ndre2", "ndre3", "tcari", "mcari", "wi")
vi_cols <- vi_cols[vi_cols %in% colnames(hsi_data)]
cat("植被指数列:", paste(vi_cols, collapse = ", "), "\n")
```

---

## 3. 本地下游分析指南

### 3.1 数据准备和检查

```r
# ==================== 加载包 ====================
library(tidyverse)
library(ggplot2)
library(viridis)
library(RColorBrewer)
library(pheatmap)
library(corrplot)

# ==================== 读取数据 ====================
hsi_data <- read_tsv("results/hsi/image_features.tsv")

# 检查数据质量
cat("总样本数:", nrow(hsi_data), "\n")
cat("缺失值统计:\n")
print(colSums(is.na(hsi_data)))

# 检查 phase 分布
cat("\nPhase 分布:\n")
print(table(hsi_data$phase))

cat("\nPhase_core 分布:\n")
print(table(hsi_data$phase_core))

# 检查时间点分布
cat("\n时间点分布:\n")
print(table(hsi_data$time))
```

### 3.2 基础可视化

#### 3.2.1 植被指数时序图

```r
# NDVI 时序图
p1 <- ggplot(hsi_data, aes(x = time_h, y = ndvi, color = phase_core)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_smooth(method = "loess", se = TRUE, alpha = 0.2) +
  scale_color_manual(
    values = c(
      "control_25" = "#808080",
      "stress_10" = "#d62728",
      "stress_35" = "#1f77b4",
      "recovery_from_10" = "#ff7f0e",
      "recovery_from_35" = "#2ca02c"
    )
  ) +
  labs(
    x = "Time (hours)",
    y = "NDVI",
    color = "Phase",
    title = "NDVI Time Series"
  ) +
  theme_bw() +
  theme(
    legend.position = "right",
    plot.title = element_text(face = "bold", hjust = 0.5)
  )

ggsave("ndvi_timeseries.pdf", p1, width = 10, height = 6, dpi = 300)

# 多个植被指数对比
vi_to_plot <- c("ndvi", "gndvi", "pri", "ari")
hsi_long <- hsi_data %>%
  select(sample_id, phase_core, time_h, all_of(vi_to_plot)) %>%
  pivot_longer(cols = all_of(vi_to_plot), names_to = "index", values_to = "value")

p2 <- ggplot(hsi_long, aes(x = time_h, y = value, color = phase_core)) +
  geom_point(alpha = 0.5, size = 1.5) +
  geom_smooth(method = "loess", se = TRUE, alpha = 0.2) +
  facet_wrap(~ index, scales = "free_y", ncol = 2) +
  scale_color_manual(
    values = c(
      "control_25" = "#808080",
      "stress_10" = "#d62728",
      "stress_35" = "#1f77b4",
      "recovery_from_10" = "#ff7f0e",
      "recovery_from_35" = "#2ca02c"
    )
  ) +
  labs(
    x = "Time (hours)",
    y = "Index Value",
    color = "Phase",
    title = "Vegetation Indices Time Series"
  ) +
  theme_bw()

ggsave("vi_timeseries_multi.pdf", p2, width = 12, height = 8, dpi = 300)
```

#### 3.2.2 箱线图/小提琴图

```r
# 按 phase_core 分组的 NDVI 分布
p3 <- ggplot(hsi_data, aes(x = phase_core, y = ndvi, fill = phase_core)) +
  geom_violin(alpha = 0.7, trim = FALSE) +
  geom_boxplot(width = 0.1, fill = "white", outlier.size = 0.5) +
  scale_fill_manual(
    values = c(
      "control_25" = "#808080",
      "stress_10" = "#d62728",
      "stress_35" = "#1f77b4",
      "recovery_from_10" = "#ff7f0e",
      "recovery_from_35" = "#2ca02c"
    )
  ) +
  labs(
    x = "Phase",
    y = "NDVI",
    title = "NDVI Distribution by Phase"
  ) +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

ggsave("ndvi_boxplot.pdf", p3, width = 8, height = 6, dpi = 300)

# 多个指数的箱线图
vi_long <- hsi_data %>%
  select(phase_core, all_of(vi_to_plot)) %>%
  pivot_longer(cols = all_of(vi_to_plot), names_to = "index", values_to = "value")

p4 <- ggplot(vi_long, aes(x = phase_core, y = value, fill = phase_core)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.5) +
  facet_wrap(~ index, scales = "free_y", ncol = 2) +
  scale_fill_manual(
    values = c(
      "control_25" = "#808080",
      "stress_10" = "#d62728",
      "stress_35" = "#1f77b4",
      "recovery_from_10" = "#ff7f0e",
      "recovery_from_35" = "#2ca02c"
    )
  ) +
  labs(
    x = "Phase",
    y = "Index Value",
    title = "Vegetation Indices Distribution by Phase"
  ) +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

ggsave("vi_boxplot_multi.pdf", p4, width = 12, height = 8, dpi = 300)
```

#### 3.2.3 光谱曲线可视化

```r
# 提取光谱列
spec_cols <- colnames(hsi_data)[grepl("^R_\\d+", colnames(hsi_data))]
wavelengths <- as.numeric(gsub("R_", "", spec_cols))

# 按 phase_core 聚合光谱
spec_agg <- hsi_data %>%
  select(sample_id, phase_core, all_of(spec_cols)) %>%
  group_by(phase_core) %>%
  summarise(across(all_of(spec_cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  pivot_longer(cols = all_of(spec_cols), names_to = "wavelength", values_to = "reflectance") %>%
  mutate(wavelength = as.numeric(gsub("R_", "", wavelength)))

# 绘制平均光谱曲线
p5 <- ggplot(spec_agg, aes(x = wavelength, y = reflectance, color = phase_core)) +
  geom_line(size = 1.2, alpha = 0.8) +
  scale_color_manual(
    values = c(
      "control_25" = "#808080",
      "stress_10" = "#d62728",
      "stress_35" = "#1f77b4",
      "recovery_from_10" = "#ff7f0e",
      "recovery_from_35" = "#2ca02c"
    )
  ) +
  labs(
    x = "Wavelength (nm)",
    y = "Reflectance",
    color = "Phase",
    title = "Average Spectral Curves by Phase"
  ) +
  theme_bw() +
  theme(legend.position = "right")

ggsave("spectral_curves.pdf", p5, width = 10, height = 6, dpi = 300)

# 带置信区间的光谱曲线（使用所有样本）
spec_all <- hsi_data %>%
  select(sample_id, phase_core, all_of(spec_cols)) %>%
  pivot_longer(cols = all_of(spec_cols), names_to = "wavelength", values_to = "reflectance") %>%
  mutate(wavelength = as.numeric(gsub("R_", "", wavelength)))

spec_summary <- spec_all %>%
  group_by(phase_core, wavelength) %>%
  summarise(
    mean_reflectance = mean(reflectance, na.rm = TRUE),
    se_reflectance = sd(reflectance, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

p6 <- ggplot(spec_summary, aes(x = wavelength, y = mean_reflectance, color = phase_core, fill = phase_core)) +
  geom_ribbon(aes(ymin = mean_reflectance - se_reflectance, 
                  ymax = mean_reflectance + se_reflectance),
              alpha = 0.2, color = NA) +
  geom_line(size = 1.2, alpha = 0.8) +
  scale_color_manual(
    values = c(
      "control_25" = "#808080",
      "stress_10" = "#d62728",
      "stress_35" = "#1f77b4",
      "recovery_from_10" = "#ff7f0e",
      "recovery_from_35" = "#2ca02c"
    )
  ) +
  scale_fill_manual(
    values = c(
      "control_25" = "#808080",
      "stress_10" = "#d62728",
      "stress_35" = "#1f77b4",
      "recovery_from_10" = "#ff7f0e",
      "recovery_from_35" = "#2ca02c"
    )
  ) +
  labs(
    x = "Wavelength (nm)",
    y = "Reflectance (mean ± SE)",
    color = "Phase",
    fill = "Phase",
    title = "Spectral Curves with Confidence Intervals"
  ) +
  theme_bw() +
  theme(legend.position = "right")

ggsave("spectral_curves_with_ci.pdf", p6, width = 10, height = 6, dpi = 300)
```

### 3.3 统计分析

#### 3.3.1 差异分析

```r
# 比较不同 phase_core 之间的植被指数差异
# 例如：stress_35 vs control_25

# 准备数据
hsi_clean <- hsi_data %>%
  filter(!is.na(ndvi), !is.na(phase_core)) %>%
  mutate(
    phase_core = factor(phase_core, 
                        levels = c("control_25", "stress_10", "stress_35", 
                                  "recovery_from_10", "recovery_from_35"))
  )

# t 检验：stress_35 vs control_25
stress_35_ndvi <- hsi_clean %>% 
  filter(phase_core == "stress_35") %>% 
  pull(ndvi)

control_25_ndvi <- hsi_clean %>% 
  filter(phase_core == "control_25") %>% 
  pull(ndvi)

t_test_result <- t.test(stress_35_ndvi, control_25_ndvi)
print(t_test_result)

# 多组比较：ANOVA
aov_result <- aov(ndvi ~ phase_core, data = hsi_clean)
summary(aov_result)

# 事后检验：Tukey HSD
tukey_result <- TukeyHSD(aov_result)
print(tukey_result)

# 可视化 Tukey 结果
plot(tukey_result)
```

#### 3.3.2 相关性分析

```r
# 植被指数之间的相关性
vi_matrix <- hsi_data %>%
  select(all_of(vi_cols)) %>%
  cor(use = "pairwise.complete.obs")

# 相关性热图
pdf("vi_correlation_heatmap.pdf", width = 10, height = 8)
corrplot(vi_matrix, method = "color", type = "upper", 
         order = "hclust", tl.cex = 0.8, tl.col = "black")
dev.off()

# 或者使用 pheatmap
pheatmap(vi_matrix, 
         color = colorRampPalette(c("blue", "white", "red"))(100),
         cluster_rows = TRUE, cluster_cols = TRUE,
         filename = "vi_correlation_heatmap_pheatmap.pdf",
         width = 10, height = 8)
```

### 3.4 主成分分析（PCA）

```r
# 使用光谱数据进行 PCA
spec_data <- hsi_data %>%
  select(all_of(spec_cols)) %>%
  as.matrix()

# 移除缺失值过多的样本
complete_cases <- complete.cases(spec_data)
spec_data_clean <- spec_data[complete_cases, ]

# 标准化
spec_data_scaled <- scale(spec_data_clean)

# PCA
pca_result <- prcomp(spec_data_scaled, center = FALSE, scale. = FALSE)

# 提取主成分
pca_scores <- as.data.frame(pca_result$x[, 1:5])
pca_scores$sample_id <- hsi_data$sample_id[complete_cases]
pca_scores$phase_core <- hsi_data$phase_core[complete_cases]
pca_scores$temp <- hsi_data$temp[complete_cases]
pca_scores$time <- hsi_data$time[complete_cases]

# 解释的方差
pca_var <- summary(pca_result)$importance[2, ]
cat("PC1 解释方差:", round(pca_var[1] * 100, 2), "%\n")
cat("PC2 解释方差:", round(pca_var[2] * 100, 2), "%\n")

# PCA 散点图
p7 <- ggplot(pca_scores, aes(x = PC1, y = PC2, color = phase_core, shape = temp)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(
    values = c(
      "control_25" = "#808080",
      "stress_10" = "#d62728",
      "stress_35" = "#1f77b4",
      "recovery_from_10" = "#ff7f0e",
      "recovery_from_35" = "#2ca02c"
    )
  ) +
  labs(
    x = paste0("PC1 (", round(pca_var[1] * 100, 1), "%)"),
    y = paste0("PC2 (", round(pca_var[2] * 100, 1), "%)"),
    color = "Phase",
    shape = "Temperature",
    title = "PCA: Spectral Data"
  ) +
  theme_bw() +
  theme(legend.position = "right")

ggsave("pca_spectral.pdf", p7, width = 10, height = 8, dpi = 300)
```

### 3.5 与多组学数据关联分析

```r
# ==================== 读取多组学数据 ====================
# 转录组数据
rnaseq_counts <- read_tsv("results/rnaseq/deseq2/normalized_counts.tsv") %>%
  column_to_rownames("gene_id")

# 代谢组数据
metabo_intensity <- read_excel("data/metabolomics_raw/meta_intensity_pos.xlsx") %>%
  column_to_rownames("metabolite_id")

# ==================== 匹配样本 ====================
common_samples <- intersect(
  hsi_data$sample_id,
  intersect(colnames(rnaseq_counts), colnames(metabo_intensity))
)

cat("共同样本数:", length(common_samples), "\n")

# ==================== HSI 特征 vs 转录组表达 ====================
# 选择关键 HSI 特征
hsi_features <- hsi_data %>%
  filter(sample_id %in% common_samples) %>%
  select(sample_id, ndvi, gndvi, pri, ari, rep) %>%
  column_to_rownames("sample_id")

# 选择关键基因（例如：差异表达基因）
# 这里假设你已经有了差异基因列表
# deg_genes <- read_tsv("results/rnaseq/deseq2/DEG_results.tsv") %>%
#   filter(padj < 0.05, abs(log2FoldChange) > 1) %>%
#   pull(gene_id) %>%
#   head(50)

# 简化示例：选择高方差基因
gene_var <- apply(rnaseq_counts[, common_samples], 1, var, na.rm = TRUE)
top_genes <- names(sort(gene_var, decreasing = TRUE)[1:50])
rnaseq_subset <- rnaseq_counts[top_genes, common_samples]

# 相关性分析
# rnaseq_subset: 基因 × 样本
# hsi_features: 样本 × 特征
# 需要转置 rnaseq_subset 为 样本 × 基因，然后计算相关性
cor_matrix <- cor(
  t(rnaseq_subset),  # 转置为 样本 × 基因
  hsi_features,      # 已经是 样本 × 特征
  use = "pairwise.complete.obs"
)
# 结果：基因 × 特征 的相关性矩阵

# 可视化相关性热图
pheatmap(cor_matrix,
         color = colorRampPalette(c("blue", "white", "red"))(100),
         cluster_rows = TRUE, cluster_cols = TRUE,
         filename = "hsi_rnaseq_correlation.pdf",
         width = 8, height = 12)

# ==================== HSI 特征 vs 代谢组强度 ====================
# 选择关键代谢物（高方差）
metabo_var <- apply(metabo_intensity[, common_samples], 1, var, na.rm = TRUE)
top_metabolites <- names(sort(metabo_var, decreasing = TRUE)[1:50])
metabo_subset <- metabo_intensity[top_metabolites, common_samples]

# 相关性分析
# metabo_subset: 代谢物 × 样本
# hsi_features: 样本 × 特征
cor_matrix_metabo <- cor(
  t(metabo_subset),  # 转置为 样本 × 代谢物
  hsi_features,      # 已经是 样本 × 特征
  use = "pairwise.complete.obs"
)
# 结果：代谢物 × 特征 的相关性矩阵

# 可视化
pheatmap(cor_matrix_metabo,
         color = colorRampPalette(c("blue", "white", "red"))(100),
         cluster_rows = TRUE, cluster_cols = TRUE,
         filename = "hsi_metabo_correlation.pdf",
         width = 8, height = 12)
```

### 3.6 时间序列分析

```r
# 计算有效时间（恢复阶段 +168h）
hsi_data <- hsi_data %>%
  mutate(
    t_eff_h = case_when(
      str_detect(phase, "recovery_from_") ~ time_h + 168,
      TRUE ~ time_h
    )
  )

# 按 phase_core 和时间聚合
hsi_agg <- hsi_data %>%
  group_by(phase_core, t_eff_h) %>%
  summarise(
    ndvi_mean = mean(ndvi, na.rm = TRUE),
    ndvi_se = sd(ndvi, na.rm = TRUE) / sqrt(n()),
    gndvi_mean = mean(gndvi, na.rm = TRUE),
    gndvi_se = sd(gndvi, na.rm = TRUE) / sqrt(n()),
    n = n(),
    .groups = "drop"
  )

# 时序图（带误差棒）
p8 <- ggplot(hsi_agg, aes(x = t_eff_h, y = ndvi_mean, color = phase_core)) +
  geom_point(size = 2) +
  geom_line(size = 1) +
  geom_errorbar(aes(ymin = ndvi_mean - ndvi_se, ymax = ndvi_mean + ndvi_se),
                width = 2, alpha = 0.5) +
  scale_color_manual(
    values = c(
      "control_25" = "#808080",
      "stress_10" = "#d62728",
      "stress_35" = "#1f77b4",
      "recovery_from_10" = "#ff7f0e",
      "recovery_from_35" = "#2ca02c"
    )
  ) +
  labs(
    x = "Effective Time (hours)",
    y = "NDVI (mean ± SE)",
    color = "Phase",
    title = "NDVI Time Series (Aggregated)"
  ) +
  theme_bw() +
  theme(legend.position = "right")

ggsave("ndvi_timeseries_aggregated.pdf", p8, width = 10, height = 6, dpi = 300)
```

### 3.7 热图可视化

```r
# 按 phase 和时间点聚合，生成热图数据
hsi_heatmap <- hsi_data %>%
  group_by(phase_core, time) %>%
  summarise(
    ndvi = mean(ndvi, na.rm = TRUE),
    gndvi = mean(gndvi, na.rm = TRUE),
    pri = mean(pri, na.rm = TRUE),
    ari = mean(ari, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = c(ndvi, gndvi, pri, ari), 
               names_to = "index", values_to = "value") %>%
  mutate(
    phase_core = factor(phase_core, 
                        levels = c("control_25", "stress_10", "stress_35", 
                                  "recovery_from_10", "recovery_from_35")),
    time = factor(time, levels = c("2h", "6h", "1d", "3d", "7d"))
  )

# 热图
p9 <- ggplot(hsi_heatmap, aes(x = time, y = phase_core, fill = value)) +
  geom_tile(color = "white", size = 0.5) +
  scale_fill_viridis_c(name = "Index\nValue", option = "plasma") +
  facet_wrap(~ index, scales = "free", ncol = 2) +
  labs(
    x = "Time",
    y = "Phase",
    title = "Vegetation Indices Heatmap"
  ) +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold")
  )

ggsave("vi_heatmap.pdf", p9, width = 12, height = 8, dpi = 300)
```

---

## 4. 与 quicklook.py 的对比

### quicklook.py 的功能
- 自动生成基础可视化（时序图、热图等）
- 论文级别的图表格式（300+ dpi）
- 适合快速总览和论文发表

### 本地 R 分析的优势
- **灵活调试**：可以快速修改参数、调整配色、尝试不同的可视化方法
- **交互式探索**：可以在 RStudio 中实时查看和调整图表
- **统计分析**：可以方便地进行统计检验、相关性分析等
- **定制化**：可以根据具体需求定制图表样式和内容

### 建议工作流程
1. **快速总览**：先运行 `quicklook.py` 获得基础可视化
2. **深入分析**：在本地使用 R 进行详细分析和定制化可视化
3. **论文发表**：使用 R 生成最终的高质量图表

---

## 5. 常见问题

### Q1: 如何处理缺失值？

**A**: 
```r
# 检查缺失值
colSums(is.na(hsi_data))

# 移除缺失值过多的样本
hsi_clean <- hsi_data %>%
  filter(rowSums(is.na(select(., all_of(vi_cols)))) < length(vi_cols) * 0.5)

# 或者使用插值
library(zoo)
hsi_data$ndvi <- na.approx(hsi_data$ndvi)
```

### Q2: 如何选择关键的光谱波段？

**A**: 
```r
# 方法 1：基于方差
spec_var <- apply(hsi_data[, spec_cols], 2, var, na.rm = TRUE)
key_bands <- names(sort(spec_var, decreasing = TRUE)[1:20])

# 方法 2：基于与目标变量的相关性
cor_with_ndvi <- cor(hsi_data$ndvi, hsi_data[, spec_cols], use = "pairwise.complete.obs")
key_bands <- names(sort(abs(cor_with_ndvi), decreasing = TRUE)[1:20])
```

### Q3: 如何标准化数据？

**A**: 
```r
# Z-score 标准化
hsi_scaled <- hsi_data %>%
  mutate(across(all_of(vi_cols), ~ scale(.x)[, 1]))

# Min-Max 标准化
hsi_minmax <- hsi_data %>%
  mutate(across(all_of(vi_cols), ~ (.x - min(.x, na.rm = TRUE)) / 
                (max(.x, na.rm = TRUE) - min(.x, na.rm = TRUE))))
```

---

## 6. 参考文件

- HSI 图像特征表：`results/hsi/image_features.tsv`
- Session 聚合特征：`results/hsi/session_features.tsv`
- Delta 指标：`results/hsi/delta_hsi.tsv`
- 恢复力指标：`results/hsi/resilience_metrics.tsv`
- 转录组数据：`results/rnaseq/deseq2/normalized_counts.tsv`
- 代谢组数据：`data/metabolomics_raw/meta_intensity_pos.xlsx`

---

**最后更新**：2025-01-XX

