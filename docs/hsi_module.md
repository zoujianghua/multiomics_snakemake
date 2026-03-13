# HSI（高光谱成像）模块文档

## 概述

HSI 模块处理高光谱图像数据，从原始 HSI 立方体提取特征，进行图像级、叶片级和 patch 级的机器学习分类。支持传统机器学习（RF、SVM、KNN、LR、XGB、PLS、LDA）与深度学习（1D/2D/3D CNN）。流程由 Snakemake 编排，脚本与规则对应关系见下文，便于论文材料与方法写作时引用具体步骤与参数。

**材料与方法可概括为**：① 图像预处理与反射率标定（preprocess.py）；② 异常样本清洗与植被指数计算（clean_image_features.py + add_indices.py）；③ 按 session 聚合与恢复力指标（aggregate.py）；④ 图像/叶片/patch 三级划分与特征表；⑤ patch 几何索引与 patch_cubes 构建；⑥ 传统 ML 与 2D/3D CNN 训练（含混合精度与梯度累积）；⑦ 生理状态标签下的对比流程（physiological.smk）。**所有划分与模型训练使用固定种子 42**，便于复现。

## 模块结构

### 规则文件

- `rules/hsi.smk`：图像预处理、清洗、聚合和可视化
- `rules/hsi_ml.smk`：图像级别的机器学习分类
- `rules/leaf.smk`：叶片级别的机器学习分类
- `rules/patch.smk`：patch 级别的机器学习分类（包括深度学习模型）
- `rules/physiological.smk`：基于生理状态标签的 patch 分类（依赖 patch.smk）
- `rules/hsi_all.smk`：HSI 模块的汇总入口

### 输入文件

- `config/samples_hsi.csv`：HSI 样本表，包含样本 ID、cube_npz 路径等信息

### 输出目录

- `results/hsi/`：所有 HSI 相关结果
  - `raw_image_features.tsv`：原始图像特征
  - `clean_image_features.tsv`：清洗后的图像特征
  - `image_features.tsv`：最终图像特征（包含植被指数）
  - `session_features.tsv`：session 级别聚合特征
  - `delta_hsi.tsv`：差异特征
  - `resilience_metrics.tsv`：恢复力指标
  - `ml/`：图像级机器学习结果
  - `leaf/`：叶片级机器学习结果
  - `patch/`：patch 级机器学习结果
  - `patch_cubes/`：patch 集合文件（用于深度学习）

---

## 1. 图像预处理流程

### 1.1 特征提取（`hsi_preprocess`）

**规则**：`hsi_preprocess`

**输入**：`config/samples_hsi.csv`、`scripts/hsi/preprocess.py`

**输出**：`results/hsi/raw_image_features.tsv`

**功能**：从原始 HSI 立方体（ENVI 格式）进行反射率标定、叶片分割与 ROI 光谱提取；输出每样本一行，含元数据、精细 phase、光谱列（R_&lt;nm&gt;）及 R800_med 等质量指标。光谱经 Savitzky–Golay 平滑（脚本内），分割基于 NDVI/NDRE 等阈值。

**主要参数**（与 `rules/hsi.smk` 一致）：
- `--sg-window 9 --sg-poly 2`：Savitzky–Golay 滤波
- `--min-ndvi 0.10 --min-ndre 0.02`：植被指数阈值
- `--min-area 2000`：最小区域面积
- `--workers {threads}`：并行线程数
- 反射率标定可使用固定白/暗参考 HDR 路径（规则中通过 `--fixed-white-hdr` / `--fixed-dark-hdr` 传入）

### 1.2 清洗和植被指数计算（`hsi_clean_and_indices`）

**规则**：`hsi_clean_and_indices`

**输入**：
- `results/hsi/raw_image_features.tsv`
- `scripts/hsi/clean_image_features.py`
- `scripts/hsi/add_indices.py`

**输出**：
- `results/hsi/clean_image_features.tsv`（清洗后）
- `results/hsi/image_features.tsv`（最终，包含植被指数）

**功能**：
1. **清洗异常样本**（`clean_image_features.py`）：
   - 基于 R800 反射率阈值过滤异常样本
   - 基于 Z-score 过滤离群值

2. **计算植被指数**（`add_indices.py`）：
   - 在 400–1000 nm 范围内计算多种植被指数：NDVI、GNDVI、NDRE1/2/3、PRI、REP、EVI、SAVI、TCARI/OSAVI、水分指数等（见脚本注释）
   - 输出列名为小写指数名，写入 `image_features.tsv`

### 1.3 聚合到 Session 级别（`hsi_aggregate`）

**规则**：`hsi_aggregate`

**输入**：
- `results/hsi/image_features.tsv`
- `scripts/hsi/aggregate.py`

**输出**：
- `results/hsi/session_features.tsv`：session 级别聚合特征
- `results/hsi/delta_hsi.tsv`：差异特征（相对于对照）
- `results/hsi/resilience_metrics.tsv`：恢复力指标

**功能**：
- 按 session 聚合图像特征（使用 mean）
- 计算相对于对照温度（默认 25°C）的差异
- 计算恢复力相关指标

### 1.4 可视化（`hsi_quicklook`）

**规则**：`hsi_quicklook`

**输入**：
- `results/hsi/image_features.tsv`
- `results/hsi/session_features.tsv`
- `scripts/hsi/quicklook.py`

**输出**：
- `results/hsi/plot_ndvi_timeseries.png`：NDVI 时间序列图
- `results/hsi/plot_drep_timeseries.png`：DREP 时间序列图
- `results/hsi/plot_dndvi_heatmap.png`：ΔNDVI 热图

---

## 2. 叶片级预处理

### 2.1 叶片特征提取（`hsi_preprocess_leaf`）

**规则**：`hsi_preprocess_leaf`

**输入**：
- `config/samples_hsi.csv`
- `results/hsi/cube/`（cube 文件目录）
- `scripts/hsi/preprocess_leaf.py`

**输出**：
- `results/hsi/raw_leaf_features.tsv`

**功能**：
- 从 HSI cube 中提取叶片级别特征
- 基于 NDVI/NDRE 阈值识别叶片区域
- 提取叶片的光谱特征

### 2.2 叶片特征清洗和指数计算（`hsi_leaf_clean_and_indices`）

**规则**：`hsi_leaf_clean_and_indices`

**输入**：
- `results/hsi/raw_leaf_features.tsv`
- `scripts/hsi/clean_image_features.py`（复用）
- `scripts/hsi/add_indices.py`（复用）

**输出**：
- `results/hsi/leaf_features.tsv`

**功能**：与图像级清洗和指数计算类似，但针对叶片级别数据

---

## 3. 图像级机器学习

### 3.1 模型列表

**规则文件**：`rules/hsi_ml.smk`

**输入**：`results/hsi/image_features.tsv`

**支持的模型**：
- **传统 ML**：RF（随机森林）、SVM（支持向量机）、KNN（K近邻）、LR（逻辑回归）、XGB（XGBoost）、PLS（偏最小二乘）、LDA（线性判别分析）
- **深度学习**：1D-CNN（光谱序列模型）

**输出目录**：`results/hsi/ml/{model}/`

**输出文件**：
- `best_model.pt`（如适用）
- `metrics.tsv`：模型评估指标
- `classification_report.tsv`：分类报告
- `confusion_matrix_norm.png`：混淆矩阵
- `pca_scatter.png`：PCA 散点图
- `roc_pr_*.png`：ROC/PR 曲线（如适用）

### 3.2 模型汇总（`hsi_ml_merge`）

**规则**：`hsi_ml_merge`

**输入**：所有模型的 `metrics.tsv` 文件

**输出**：
- `results/hsi/ml/all_models_summary.csv`：所有模型的性能汇总

---

## 4. 叶片级机器学习

### 4.1 模型列表

**规则文件**：`rules/leaf.smk`

**输入**：`results/hsi/leaf_features.tsv`

**支持的模型**：与图像级相同（RF、SVM、KNN、LR、XGB、PLS、LDA、1D-CNN）

**输出目录**：`results/hsi/leaf/{model}/`

### 4.2 模型汇总（`leaf_merge`）

**规则**：`leaf_merge`

**输出**：
- `results/hsi/leaf/all_models_summary.csv`

---

## 5. Patch 级机器学习

### 5.1 流程概述

**规则文件**：`rules/patch.smk`

Patch 级机器学习流程包括以下步骤：

1. **通用几何索引生成**（`hsi_patch_index_base`）
2. **Patch 集合文件生成**（`hsi_patch_cubes`）
3. **任务特定划分**（`hsi_patch_split`）
4. **任务特定索引构建**（`hsi_patch_make_index`）
5. **特征提取**（传统 ML 用）
6. **模型训练**（传统 ML + 深度学习）

### 5.2 通用几何索引生成

**规则**：`hsi_patch_index_base`

**输入**：
- `results/hsi/image_features.tsv`
- `scripts/hsi/build_patch_index_base.py`

**输出**：`results/hsi/ml/patch_index_base.tsv`

**功能**：从 `image_features.tsv` 读取 sample_id 与 cube_npz 路径，在每幅图像上按滑窗生成候选 patch 坐标；不包含 target/split，为纯几何索引，供后续任务复用。

**参数**（由 `config["hsi"]` 或规则传入，默认）：
- `--patch-size`：patch 边长（默认 32，`config["hsi"]["patch_size"]`）
- `--stride`：滑窗步长（默认 16，`config["hsi"]["patch_stride"]`；可改为 32 减少重叠以缓解过拟合）
- `--min-mask-frac 1.0`：最小掩码比例
- `--max-patches-per-image`：每张图像最大 patch 数（默认 200，`config["hsi"]["max_patches_per_image"]`；可改为 50～100 减少同图重复）

### 5.3 Patch 集合文件生成

**规则**：`hsi_patch_cubes`

**输入**：
- `results/hsi/ml/patch_index_base.tsv`
- `scripts/hsi/build_patch_cubes.py`

**输出**：
- `results/hsi/patch_cubes/` 目录（每个 cube 一个 patch 集合文件）
- `results/hsi/ml/patch_index_base_cubes.tsv`（更新后的索引，包含 `cube_patch_npz` 和 `patch_idx`）

**功能**：
- 将 base 索引中的所有 patch 按 cube_npz 分组
- 生成「一 cube 一个 patch 集合文件」（`{sample_id}_patches.npz`）
- 大幅减少 I/O 开销，提升深度学习训练速度

**数据格式**：
- 每个 `.npz` 文件包含 `patches` 数组：`[N_patches, B, H, W]`
- dtype：`float32`

### 5.4 任务特定划分

**规则**：`hsi_patch_split`

**输入**：
- `results/hsi/image_features.tsv`
- `scripts/hsi/make_split_for_target.py`

**输出**：
- `results/hsi/ml/split_{patch_target}_seed{patch_seed}.tsv`

**功能**：
- 按指定 target 列（如 `phase_core`、`metabo_state`、`rnaseq_state`）生成分层 train/test 划分
- 支持多种 target，通过 `config["hsi"]["patch_target"]` 配置

**参数**：
- `--target-col {patch_target}`：目标列名
- `--seed {patch_seed}`：随机种子（默认 42）
- `--test-size 0.2`：测试集比例

### 5.5 任务特定索引构建

**规则**：`hsi_patch_make_index`

**输入**：
- `results/hsi/ml/patch_index_base_cubes_{param_suffix}.tsv`（`param_suffix` 由 `patch_stride`、`max_patches_per_image` 编码，如 `s16_maxp200`）
- `results/hsi/ml/split_{patch_target}_seed{patch_seed}.tsv`
- `scripts/hsi/make_patch_index_for_target.py`

**输出**：
- `results/hsi/ml/patch_index_{patch_target}_seed{patch_seed}_{param_suffix}.tsv`

**功能**：
- 将几何索引与 split 信息合并
- 为每个 patch 分配 target 标签和 split（train/test）

### 5.6 Patch 特征提取（传统 ML 用）

**规则链**：

1. **`hsi_patch_features_raw`**：提取原始 patch 特征
   - 输入：`patch_index_{patch_target}_seed{patch_seed}_{param_suffix}.tsv`
   - 输出：`results/hsi/raw_patch_features_{run_id}.tsv`（`run_id = {patch_target}_{param_suffix}`）

2. **`hsi_patch_clean_and_indices`**：清洗并计算指数
   - 输入：`raw_patch_features_{run_id}.tsv`
   - 输出：`results/hsi/patch_features_{run_id}.tsv`

3. **`hsi_patch_features_omics`**：生成多组学关联分析用特征表
   - 输出：`results/hsi/patch_features_omics_{run_id}.tsv`

### 5.7 传统机器学习模型

**支持的模型**：RF、SVM、KNN、LR、XGB、PLS、LDA

**输入**：`results/hsi/patch_features_{run_id}.tsv`

**输出目录**：`results/hsi/patch/{run_id}/{model}/`（如 `phase_core_s16_maxp200/rf/`；不同 stride/max_patches 可并存）

**SVM 说明**：使用 `RandomizedSearchCV`（默认 `--n-iter 48`），参数空间为 `kernel`（linear/rbf）、`C`（loguniform 1e-3–1e3）、`gamma`（loguniform 1e-4–10），在保证搜索覆盖的前提下缩短 patch 级训练时间；最佳模型保存为 `svm_best_model.joblib`。

### 5.8 深度学习模型

#### 5.8.1 1D-CNN（光谱序列模型）

**规则**：`patch_1dcnn`

**脚本**：`scripts/hsi/hsi_1dcnn.py`

**输入**：`results/hsi/patch_features_{run_id}.tsv`（从 TSV 读取光谱特征）

**模型结构**：
- 初始 Conv1d (1→32) + BatchNorm1d + ReLU
- 3 个 ResidualBlock1D：
  - Block 1: 32→64, stride=2
  - Block 2: 64→128, stride=2
  - Block 3: 128→128, stride=1
- GlobalAveragePooling1D
- 特征层：Linear(128 → embedding_dim)
- 分类头：Dropout(0.3) + Linear(embedding_dim → n_classes)

**输出**：`results/hsi/patch/1dcnn_image_{run_id}/best_model.pt`、`metrics.tsv` 等。

**特点**：输入为 TSV 中的光谱特征序列；模型含残差块与 `forward_features()`，可提取 embedding；训练/验证接口与 2D/3D 一致。

#### 5.8.2 2D-CNN（空间+光谱模型）

**规则**：`patch_2dcnn`

**脚本**：`scripts/hsi/hsi_patch_cnn2d.py`

**输入**：`results/hsi/ml/patch_index_{patch_target}_seed{patch_seed}_{param_suffix}.tsv`（使用 `HSIPatchDataset` 从 `patch_cubes_{param_suffix}` 读取）

**模型结构**：
- 1×1 Conv 光谱降维（B 波段 → 32 通道）+ BatchNorm2d + ReLU
- 3 个 ResidualBlock2D：
  - Block 1: 32→64, stride=1
  - Block 2: 64→128, stride=2
  - Block 3: 128→128, stride=1
- GlobalAveragePooling2D
- 特征层：Linear(128 → embedding_dim)
- 分类头：**Dropout(0.3)** + Linear(embedding_dim → n_classes)（缓解过拟合）
- 默认 **weight_decay=5e-4**（L2 正则）

**输出**：`results/hsi/patch/2dcnn_{run_id}/best_model.pt`、`metrics.tsv`、**`history.tsv`**、**`train_val_curves.png`**、**`patch_embeddings_2d.tsv`**（同目录下）。

**训练与显存**：
- `--amp`：混合精度（FP16），显著降低显存；Snakemake 规则中已默认开启
- `--gradient-accumulation-steps N`：梯度累积，可与较小 batch 配合使用
- `--num-workers`（默认 0）、`--no-pin-memory`：DataLoader 参数，默认不启多进程以防 OOM
- 每轮结束后执行 `torch.cuda.empty_cache()` 减轻显存碎片
- **默认 batch_size**：64（规则可覆盖）；生理状态规则中常用更小 batch（如 4）

#### 5.8.3 3D-CNN（空间+光谱模型）

**规则**：`patch_3dcnn`

**脚本**：`scripts/hsi/hsi_patch_cnn3d.py`

**输入**：与 2D-CNN 相同

**模型结构**：
- 1×1×k 3D Conv 光谱降维 + BatchNorm3d + ReLU
- 3 个 ResidualBlock3D：
  - Block 1: 16→32, stride=1
  - Block 2: 32→64, stride=2
  - Block 3: 64→64, stride=1
- GlobalAveragePooling3D
- 特征层：Linear(64 → embedding_dim)
- 分类头：**Dropout(0.3)** + Linear(embedding_dim → n_classes)（缓解过拟合）
- 默认 **weight_decay=5e-4**（L2 正则）

**输出**：`results/hsi/patch/3dcnn_{run_id}/best_model.pt`、`metrics.tsv`、**`history.tsv`**、**`train_val_curves.png`**、**`patch_embeddings_3d.tsv`**（同目录下）。

**训练与显存**：与 2D-CNN 相同（`--amp`、`--gradient-accumulation-steps`、`--num-workers`、`--no-pin-memory`）；**默认 batch_size**：8；规则中可传入 `--amp`。

**数据增强**：训练集默认启用轻量增强（水平/垂直翻转，概率 0.5）；可选 `--augment-noise-std 0.01` 添加光谱高斯噪声。禁用增强：`--no-augment`。

#### 5.8.4 Patch 级过拟合与优化

**常见原因**：① 2D/3D 模型未使用 Dropout，分类头易过拟合；② 无数据增强，模型易记忆固定空间/光谱模式；③ 滑窗 stride 小（如 16）、每图 patch 多（如 200），同一图像内 patch 高度重叠，有效独立样本数少；④ 仅 L2 正则（weight_decay）偏弱；⑤ 标签为图像级下放，若表型在空间上不均匀，patch 级标签带噪，准确率上限受限。

**已采取的改进**：
- **模型**：在 2D/3D 分类头前增加 **Dropout(0.3)**；默认 **weight_decay=5e-4**（可由规则或命令行覆盖）。
- **数据**：训练集默认启用**水平/垂直翻转**；可选 `--augment-noise-std` 添加光谱噪声；`config["hsi"]` 下 **patch_stride**、**max_patches_per_image** 可调（如 stride 改为 32 减少重叠，max_patches 改为 50～100 减少同图重复），需重跑 `hsi_patch_index_base` 与 `hsi_patch_cubes`。不同参数结果写入不同路径（见下文「参数化输出路径」），可并存对比。
- **诊断**：每轮 **train_loss / val_loss / train_f1_weighted / val_f1_weighted** 写入 **history.tsv**，并绘制 **train_val_curves.png**（loss 与 F1 随 epoch 变化）。若 train 持续降、val 先降后升，即为过拟合；可据此判断正则与增强是否足够。

**如何判断是架构还是 patch/标签问题**：对比**图像级**（1D-CNN 或传统 ML 在 image_features.tsv 上）与 **patch 级**（2D/3D CNN）在相同 target 下的 test 准确率与 F1。若在加强正则与增强后，patch 级仍明显低于图像级，则更可能是任务/标签/尺度（patch 分割与图像级标签下放）的问题；若 patch 级接近或超过图像级，则架构与 patch 设计基本合理，embedding 用于多组学更可信。

### 5.9 Patch Embedding 导出

**功能**：
- 2D/3D CNN 训练结束后自动导出 patch 级 embedding
- 包含完整的元数据信息

**输出格式**（TSV）：
- 列：`patch_id`, `source_sample_id`, `split`, `target`, `emb_0`, `emb_1`, ..., `emb_{D-1}`
- 包含所有 patch（train + test），按原始顺序排列
- 便于在本地 R 多组学项目中进行关联分析

**文件路径**：
- 2D CNN：`results/hsi/patch/2dcnn_{run_id}/patch_embeddings_2d.tsv`
- 3D CNN：`results/hsi/patch/3dcnn_{run_id}/patch_embeddings_3d.tsv`

### 5.10 参数化输出路径（多组参数并存）

**目的**：不同 `patch_stride` / `max_patches_per_image` 跑出的结果不覆盖、可并存。

**约定**：
- `param_suffix = s{patch_stride}_maxp{max_patches_per_image}`（如 `s16_maxp200`、`s32_maxp100`）
- `run_id = {patch_target}_{param_suffix}`（如 `phase_core_s16_maxp200`）

**路径示例**：
- 几何索引：`patch_index_base_{param_suffix}.tsv`、`patch_index_base_cubes_{param_suffix}.tsv`、`patch_cubes_{param_suffix}/`
- 任务索引：`patch_index_{patch_target}_seed{patch_seed}_{param_suffix}.tsv`
- 特征与模型：`patch_features_{run_id}.tsv`、`results/hsi/patch/{run_id}/`、`2dcnn_{run_id}/`、`3dcnn_{run_id}/` 等

修改 config 中 stride 或 max_patches 后重跑，会生成新目录，与旧结果并存；`hsi_all_models_index` 会扫描所有 `patch/*/all_models_summary.csv` 与 `2dcnn_*`/`3dcnn_*` 并汇总到一张表，表中 `target` 列会包含 run_id（如 `phase_core_s16_maxp200`）。

### 5.11 Patch 模型汇总

**规则**：`patch_merge`

**输入**：所有 patch 模型的 `metrics.tsv` 文件

**输出**：
- `results/hsi/patch/{run_id}/all_models_summary.csv`

### 5.12 生理状态分类（physiological.smk）

**规则文件**：`rules/physiological.smk`

基于多组学（代谢组+转录组）定义的 `physiological_state` 标签，复用 patch.smk 中已生成的数据，进行与 phase 分类的对比实验。

**流程**：
1. **physio_image_meta**：`image_features.tsv` + `config/phase_physiological_state_mapping.tsv` → `results/hsi/ml/image_features_with_physio.tsv`（临时，不覆盖原文件）
2. **physio_patch_features**：`patch_features_{run_id}.tsv` + mapping → `results/hsi/patch_features_with_physiological.tsv`
3. **physio_split**：按 `physiological_state` 在 image 级别分层划分（避免数据泄露）
4. **physio_patch_index**：复用 `patch_index_base_cubes_{param_suffix}.tsv`，生成 `patch_index_physiological_state_seed42_{param_suffix}.tsv`
5. **模型训练**：与传统 patch 流程相同，使用现有脚本（RF/SVM/KNN/LR/XGB/PLS/LDA + 1D/2D/3D CNN）

**输入**：
- `config/phase_physiological_state_mapping.tsv`：phase → physiological_state 映射表
- `results/hsi/patch_features_{run_id}.tsv`：来自 patch.smk
- `results/hsi/ml/patch_index_base_cubes_{param_suffix}.tsv`：来自 patch.smk

**输出**：
- `results/hsi/patch_features_with_physiological.tsv`：带生理状态标签的 patch 特征表
- `results/hsi/patch/{physio_run_id}/{model}/`：各模型结果（`physio_run_id = physiological_state_{param_suffix}`）
- `results/hsi/patch/{physio_run_id}/all_models_summary.csv`：汇总结果

**注意事项**：
- 不修改 `image_features.tsv`，避免触发整个项目重跑
- 充分利用 patch.smk 中已完成的 patch_cubes、patch_features 等数据
- 生理状态标签可能导致类别不平衡，`make_split` 已做分层采样

---

## 6. 数据集类（HSIPatchDataset）

**文件**：`scripts/hsi/hsi_patch_dataset.py`

**功能**：
- 支持从 `patch_cubes` 文件读取 patch 数据（推荐）
- 支持从单个 patch npz 文件读取（deprecated）
- 支持从 cube 动态裁剪（兼容模式）

**模式**：
- `mode="2d"`：返回 `[B, H, W]` 张量（用于 2D CNN）
- `mode="3d"`：返回 `[1, B, H, W]` 张量（用于 3D CNN）

**Split 参数**：
- `split="train"`：只使用训练集
- `split="test"`：只使用测试集
- `split="all"`：使用所有数据（用于导出 embedding）

**元数据维护**：
- 维护 `meta_df`，包含 `patch_id`, `source_sample_id`, `split`, `target` 等信息
- 确保与 DataLoader 顺序一致

---

## 7. 配置说明

### config/config.yaml

```yaml
hsi:
  patch_target: "phase_core"   # 可选：phase_core, metabo_state, rnaseq_state
  patch_seed: 42
  patch_size: 32
  patch_stride: 16            # 滑窗步长；与 max_patches_per_image 编码进输出路径，多组可并存
  max_patches_per_image: 200   # 每图最多 patch 数
```

---

## 8. 运行方法

### 运行完整 HSI 流程

```bash
# 运行所有 HSI 相关规则
snakemake -j 16 all_hsi --use-conda

# 或分步运行
snakemake -j 16 hsi_preprocess --use-conda
snakemake -j 16 hsi_clean_and_indices --use-conda
snakemake -j 16 hsi_aggregate --use-conda
```

### 运行 Patch 级深度学习

```bash
# 运行完整的 patch 流程（包括索引、特征提取、模型训练）
snakemake -j 16 patch_2dcnn patch_3dcnn --use-conda

# 或只运行特定步骤
snakemake -j 16 hsi_patch_index_base --use-conda
snakemake -j 16 hsi_patch_cubes --use-conda
snakemake -j 16 patch_2dcnn --use-conda
```

---

## 9. 依赖环境

### Conda 环境

- **hsi_env.yaml**：用于所有 HSI 相关脚本
  - 位置：`envs/hsi_env.yaml`
  - 包含：Python、pandas、numpy、scikit-learn、PyTorch、torchvision 等

---

## 10. 注意事项

1. **DataLoader 稳定性**：
   - 默认 `num_workers=0`，避免多进程 OOM 问题
   - 在 HPC 上可根据实际情况调整 `--num-workers` 参数

2. **Patch Cubes 格式**：
   - 每个 cube 一个 patch 集合文件，大幅减少 I/O 开销
   - dtype 为 `float32`，不要修改

3. **Embedding 导出**：
   - 2D/3D CNN 训练结束后自动导出
   - 包含完整的元数据，便于多组学关联分析

4. **模型结构**：所有 CNN 采用残差结构；2D/3D 训练默认开启 `--amp`，必要时可调小 batch 或使用 `--gradient-accumulation-steps`。

5. **生理状态流程**：`physiological.smk` 使用与 patch.smk 相同的 `param_suffix`，输出目录为 `results/hsi/patch/{physio_run_id}/`（如 `physiological_state_s16_maxp200`），与主 target 多组参数可并存。

---

## 11. 输出文件总结

### 图像级
- `results/hsi/image_features.tsv`：最终图像特征
- `results/hsi/session_features.tsv`：session 级别聚合
- `results/hsi/ml/{model}/`：各模型结果

### 叶片级
- `results/hsi/leaf_features.tsv`：叶片特征
- `results/hsi/leaf/{model}/`：各模型结果

### Patch 级
- `results/hsi/patch_cubes_{param_suffix}/`：patch 集合文件（每 cube 一个 npz；如 `s16_maxp200`）
- `results/hsi/patch/{run_id}/{model}/`：各模型结果（run_id 如 `phase_core_s16_maxp200`；rf、svm、knn、lr、xgb、pls、lda、1dcnn_image_*、2dcnn_*、3dcnn_*）
- `results/hsi/patch/2dcnn_{run_id}/patch_embeddings_2d.tsv`：2D CNN 导出的 patch embedding
- `results/hsi/patch/3dcnn_{run_id}/patch_embeddings_3d.tsv`：3D CNN 导出的 patch embedding

### 生理状态分类（physiological.smk）
- `results/hsi/patch_features_with_physiological.tsv`：带生理状态标签的 patch 特征表
- `results/hsi/patch/{physio_run_id}/{model}/`：各模型结果（如 `physiological_state_s16_maxp200`）

