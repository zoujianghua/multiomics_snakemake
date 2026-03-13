## HSI 模块论文展示 RULE（正文与附录）

本 RULE 用于约束 HSI 模块在论文中的**科学叙事、结果展示范围和导出规范**，目标是围绕：

- **核心问题**：HSI 能否在不依赖大规模组学测序的情况下，**预测由多组学定义的生理 / 分子状态**（physiological / omics state），并作为一种高通量、低成本的替代或前置监测手段。

并避免在正文中堆砌过多模型结果，把细节压缩到附录和代码仓库中。

---

## 1. 研究定位与章节结构

- **主线问题**：  
  - 使用 HSI（光谱 + 指数 + patch 级 CNN embedding）预测多组学整合得到的 `physiological_state` / `rnaseq_state` / `metabo_state`。  
  - 传统 `phase` / `phase_core` 仅作为对比与解释辅助。

- **推荐章节结构**：
  - **材料与方法（HSI 部分）**：  
    - 图像预处理与指数计算（对应 `hsi_module.md` 中描述的 `preprocess.py` + `add_indices.py` 流程）。  
    - 图像 / 叶片 / patch 三级特征构建（对应 `patch.smk` 及 `build_patch_*` 等脚本）。  
    - 传统 ML + CNN 模型与训练策略（early stopping / 正则 / 多次重复的说明）。  
    - 生理状态标签构建与映射（对应 `hsi_physiological_states_guide.md` 与 `phase_physiological_state_mapping.tsv`）。  
  - **结果**：  
    - 先展示 HSI 区分不同温度胁迫 / 恢复阶段的能力（phase / phase_core）。  
    - 再展示 HSI 对 `physiological_state` 等多组学标签的预测性能。  
    - 最后展示 HSI embedding 与多组学特征 / 模块的相关性与空间重叠。  
  - **讨论**：  
    - HSI 替代 / 提前预警多组学状态的潜力与局限（样本量、patch 相关性、设备条件等）。  

---

## 2. 模型与训练策略统一规范

### 2.1 CNN（1D / 2D / 3D）训练策略

- **早停策略（在 `hsi_1dcnn.py`、`hsi_patch_cnn2d.py`、`hsi_patch_cnn3d.py` 中实现）**：
  - 训练时统一记录每轮 `train_loss` / `val_loss` / `val_f1_weighted` / `val_f1_macro`。  
  - 使用 `val_f1_weighted` 作为选取 best epoch 的指标。  
  - 增加命令行参数（有默认值，无需在规则里强制改）：  
    - `--early-stopping-patience`（建议默认 20）  
    - `--min-epochs`（建议默认 20）  
  - 逻辑：当 `epoch >= min_epochs` 且连续 `patience` 轮没有提升时提前停止；模型始终保存 best epoch 的状态。

- **正则化与模型容量**：
  - 1D-CNN：保留现有残差结构，使用 `weight_decay`（L2 正则），默认 `1e-4`；可在附录设计 “高 / 低 weight_decay” 的 ablation。  
  - 2D/3D-CNN：模型宽度保持当前轻量 ResNet 设计（32/64/128 或 16/32/64），文中点明是“小型残差网络”；如需对比 compact 版本，可在附录再训一组。  
  - 训练脚本中统一支持 `--weight-decay` 参数（默认 `1e-4`），方便 patch 任务与 image 任务保持一致。

- **随机性与多次重复**：
  - 训练脚本均接受 `--seed` 参数，并在内部固定 `numpy` / `torch` 随机种子。  
  - **论文推荐做法**：对关键任务（尤其是 patch-level `physiological_state` 分类），至少运行 3 次不同 seed 的 2D-CNN（和 1D baseline），将 3 次 run 的 `val/test` 指标做均值 ± 标准差，整理到 `Tab-HSI-ModelMain`。  
  - Snakemake 目前只跑单一 seed；多 seed 重复可通过修改 `config.yaml` 中 `hsi.patch_seed` 并多次触发规则完成，结果在 RULE 中记录清楚即可。

### 2.2 传统 ML 模型的角色

- **RF / XGB / 其它线性模型**主要作为：
  - baseline（对比 HSI CNN）；  
  - 可解释性工具（feature importance / PLS 载荷）。  
- 正文只需要展示：  
  - RF / XGB / best-CNN 在**主任务（通常为 patch-level `physiological_state`）**上的性能对比；  
  - 完整的模型矩阵（所有任务 × 所有模型）集中放在附录表中。

---

## 3. physio_patch_rf 训练时间与 XGB 100% 准确率的检查要点

### 3.1 physio_patch_rf 训练时间

- **数据规模诊断**：  
  - 从 `results/hsi/patch_features_with_physiological.tsv` 中统计：  
    - patch 总数；  
    - 每个 `physiological_state` 的 patch 数；  
    - 同期 `patch_features_{patch_target}.tsv`（如 `phase`）的对应统计。  
  - 补充一张 `physio_patch_stats.tsv`，用于说明：  
    - `physiological_state` 任务在 patch 级别的样本数量是否远大于其它任务；  
    - 是否存在极度不平衡的类别。

- **RF 参数与搜索空间**（优先在 physio 规则中收紧，而不影响全局）：  
  - 减少 `n_estimators` 搜索范围，例如 `200–800`；  
  - 将 `--n-iter` 从 200 调整到 48–80；  
  - `threads` 设为中等值（如 16），配合合理 `n_jobs`；  
  - 如需进一步加速，可在生成 patch 特征前对每类 patch 进行子采样（例如每类随机取固定上限 patch 数），并在论文中写清这一点。

### 3.2 physio_patch_xgb 100% 准确率：数据泄露排查 checklist

在解释 100% 准确率前，必须先确认以下几点（可作为论文方法的“质量控制”小节）：

- **标签是否作为特征泄露**：  
  - 利用 `ml_utils.pick_feature_columns()` 的设定，确认：  
    - `physiological_state` / `phase` / `phase_core` 等字符串标签列不会进入特征（非数值列）。  
    - 特征由光谱列 `R_<nm>` 以及数值指数列构成。  

- **行级划分 vs 样本级划分**：  
  - 传统 ML（包括 physio_patch_rf / physio_patch_xgb）在 patch 特征表上使用 **行级分层划分**（多 patch 可能来自同一植物 / 图像）。  
  - 这一设计会让模型在“同一植株的不同 patch”上训练和测试，导致看起来非常高的 patch 准确率，但**不代表真正的新植株泛化性能**。  
  - 需要在论文中明确指出：  
    - CNN 流程（通过 `patch_index_*` 与 image-level split）是按 image 级划分，**避免了样本级数据泄露**；  
    - 传统 ML patch 模型属于“patch 内部区分能力”的评价，更偏上限估计。

- **XGB 100% 的 sanity check 建议**：  
  - 在 physio patch 特征上做一个极简实验：  
    - 只保留少量典型指数（如 NDVI / NDRE / PRI 等）训练 XGB；  
    - 或做“强正则 + 较小树数 + 较小特征子集”的模型，对比 full-feature 模型的性能变化。  
  - 如果 full-feature 模型接近 100%，而极简模型明显下降，则可以在讨论中解释为：  
    - 多指数 + 全谱在 patch 级高度冗余，足以近乎完美区分 `physiological_state`；  
    - 而不是标签直接泄露为特征。  

- **稳定评估建议**：  
  - 对 physio_patch_xgb 至少做 3 次不同随机种子（`--random-state`）的重复，记录 test 指标的均值 ± 标准差；  
  - 如果多次重复均稳定接近 100%，需结合混淆矩阵和 ROC/PR 曲线展示“所有类别都被正确分类”，而非仅主类主导。

---

## 4. HSI × 多组学的下游分析规范

### 4.1 HSI embedding 构建

- **来源文件**：  
  - 2D-CNN：`results/hsi/patch/2dcnn_{target}/patch_embeddings_2d.tsv`  
  - 3D-CNN：`results/hsi/patch/3dcnn_{target}/patch_embeddings_3d.tsv`  

- **推荐聚合方式**：  
  - 以 `source_sample_id` 为单位，对其所有 patch 的 embedding 做 `mean`（或 `median`）聚合，得到 image-level embedding。  
  - 生成统一表：`results/hsi/hsi_embedding_by_sample.tsv`，列包括：  
    - `sample_id` / `source_sample_id`（统一为下游多组学的样本 ID）；  
    - `phase` / `phase_core` / `physiological_state` / 温度 / 时间等元数据；  
    - `emb2d_*` / `emb3d_*` 等 embedding 向量列。

### 4.2 与转录组 / 代谢组的关联

- **典型分析（在本地 R 完成，模板见 `hsi_downstream_analysis_guide.md`）**：
  - HSI embedding 与转录组 PCA / WGCNA 模块 eigengene 的相关性热图；  
  - HSI embedding 与代谢组主成分 / 关键代谢通路的相关性热图；  
  - HSI embedding 的 PCA / UMAP 可视化，颜色按 `physiological_state` / `rnaseq_state` / `metabo_state` 着色，观察空间重叠。  

- **论文中需回答的关键问题**：  
  - 哪些 HSI embedding 维度与多组学的主轴或关键模块最强相关？  
  - 在嵌入空间中，组学上定义的几个典型状态是否被清晰分开？  
  - HSI 是否能在早期时间点（如 2h / 6h）预测后续恢复 / 死亡趋势？

---

## 5. 正文必须展示的表格与图（强制清单）

### 5.1 表格（正文）

- **Tab-HSI-DataSummary**  
  - 内容：  
    - image / leaf / patch 三级数据的样本数；  
    - 对应任务（phase_core / physiological_state / rnaseq_state / metabo_state 等）的类别分布。  
  - 来源：  
    - 由统计脚本汇总 `image_features.tsv`、`leaf_features.tsv`、`patch_features_*`、`patch_features_with_physiological.tsv`。

- **Tab-HSI-ModelMain**  
  - 内容：  
    - 主任务（建议：patch-level `physiological_state` 分类）上 RF / XGB / best-CNN 的 test accuracy、macro-F1、weighted-F1、AUC（如可）  
    - 指标以 “均值 ± SD（n=3–5 runs）” 形式给出。  
  - 来源：  
    - 来自 `results/hsi/patch/physiological_state/all_models_summary.csv` 和多次重复 run 的 `metrics.tsv`。

- **Tab-HSI-MultiomicsCompare**  
  - 内容：  
    - HSI 模型 vs 纯多组学模型 在预测 `physiological_state` / 其它 state 上的性能对比。  
    - 至少给出一个“仅 HSI”、“仅多组学”、“HSI+多组学”的横向表。  

### 5.2 图形（正文）

- **Fig-HSI-IndexTimeSeries**  
  - NDVI + 1–2 个红边 / 氮素相关指数的时间序列图，分组按 `phase_core` 或 `physiological_state`。  

- **Fig-HSI-SpectralCurves**  
  - 不同阶段（对照 / 胁迫 / 恢复）的平均光谱曲线（含置信区间），展示关键波段变化。  

- **Fig-HSI-EmbeddingPCA/UMAP**  
  - 使用 HSI embedding（image-level），颜色按 `physiological_state` / 组学 state，展示在表征空间的分离度。  

- **Fig-HSI-ConfusionMatrix**  
  - 主任务（如 patch-level `physiological_state`）的归一化混淆矩阵（best-CNN 或 best-XGB）。  

- **Fig-HSI-MultiomicsCorrelation**  
  - HSI embedding / 指数 与 多组学特征 / 模块 eigengene 的相关性热图。  

---

## 6. 附录优先展示内容

### 6.1 表格

- 所有 `all_models_summary.csv` 的清洗与合并（image / leaf / patch × phase_core / physiological_state 等），作为若干附录表。  
- CNN 训练超参数表（learning rate, batch size, epoch 上限, early stopping 参数, weight decay 等）。  
- patch 级采样 / stride / 正则化强度变化的 ablation 表（可选）。  

### 6.2 图形

- 各典型模型的训练 / 验证 loss 与 accuracy / F1 曲线（从 `history.tsv` 重绘）。  
- 更多指数分布（箱线图 / 小提琴图）与时间序列热图。  
- 额外的 PCA / UMAP 视图（按温度 / 时间分组）。  

---

## 7. “论文素材包”导出建议

- 在 Snakemake 中新增一个专门的入口（例如 `hsi_paper_materials` 规则，放在单独的 `rules/hsi_paper.smk` 中），只做**已有结果文件的收集与拷贝**，不触发重算：  
  - 统一输出到 `results/hsi/paper_materials/`：  
    - `tables/`：上述 Tab-HSI-*、模型 summary、统计表等 TSV/CSV；  
    - `fig_r_data/`：为 R 绘图准备的中间表 / 嵌入矩阵；  
    - `logs/`：关键训练 log（CNN / XGB / RF 等），用于写方法和讨论。  
- 日常使用：  
  - 在 HPC 上运行 `snakemake -j XX hsi_paper_materials --use-conda`；  
  - 然后 `scp -r results/hsi/paper_materials` 到本地，用 R / Python 生成最终论文图表。

---

## 8. 写作时需要特别强调的要点

- HSI 流程完全脚本化、可重现，**所有参数和规则均在 Snakemake 和脚本中明示**；论文可以引用本 RULE 与仓库文档减轻篇幅。  
- 明确区分：  
  - image-level / leaf-level / patch-level 任务；  
  - phase / phase_core / physiological_state / 组学状态等不同标签层次；  
  - patch 级传统 ML 的“行级划分”与 CNN 的“样本级划分”差异。  
- 在讨论中正面回应：  
  - 样本量有限、patch 高度相关的情况下，如何通过合理分割、早停和正则化减少过拟合；  
  - 为什么 HSI 仍然能在多组学定义的复杂状态上取得稳定、可解释的性能。

## HSI 模块论文展示 RULE（正文与附录）

本 RULE 用于约束 HSI 模块在论文中的**科学叙事、结果展示范围和导出规范**，目标是围绕：

- **核心问题**：HSI 能否在不依赖大规模组学测序的情况下，**预测由多组学定义的生理 / 分子状态**（physiological / omics state），并作为一种高通量、低成本的替代或前置监测手段。

并避免在正文中堆砌过多模型结果，把细节压缩到附录和代码仓库中。

---

## 1. 研究定位与章节结构

- **主线问题**：  
  - 使用 HSI（光谱 + 指数 + patch 级 CNN embedding）预测多组学整合得到的 `physiological_state` / `rnaseq_state` / `metabo_state`。  
  - 传统 `phase` / `phase_core` 仅作为对比与解释辅助。

- **推荐章节结构**：
  - **材料与方法（HSI 部分）**：  
    - 图像预处理与指数计算（对应 `hsi.smk` + `preprocess.py` + `add_indices.py`）。  
    - 图像 / 叶片 / patch 三级特征构建（对应 `patch.smk` 及 `build_patch_*` 等脚本）。  
    - 传统 ML + CNN 模型与训练策略（early stopping / 正则 / 多次重复的说明）。  
    - 生理状态标签构建与映射（对应 `hsi_physiological_states_guide.md` 与 `phase_physiological_state_mapping.tsv`）。  
  - **结果**：  
    - 先展示 HSI 区分不同温度胁迫 / 恢复阶段的能力（phase / phase_core）。  
    - 再展示 HSI 对 `physiological_state` 等多组学标签的预测性能。  
    - 最后展示 HSI embedding 与多组学特征 / 模块的相关性与空间重叠。  
  - **讨论**：  
    - HSI 替代 / 提前预警多组学状态的潜力与局限（样本量、patch 相关性、设备条件等）。  

---

## 2. 模型与训练策略统一规范

### 2.1 CNN（1D / 2D / 3D）训练策略

- **早停策略（将在 `hsi_1dcnn.py`、`hsi_patch_cnn2d.py`、`hsi_patch_cnn3d.py` 中实现）**：
  - 训练时统一记录每轮 `train_loss` / `val_loss` / `val_f1_weighted` / `val_f1_macro`。  
  - 使用 `val_f1_weighted` 作为选取 best epoch 的指标。  
  - 增加命令行参数（有默认值，无需在规则里强制改）：  
    - `--early-stopping-patience`（建议默认 20）  
    - `--min-epochs`（建议默认 20）  
  - 逻辑：当 `epoch >= min_epochs` 且连续 `patience` 轮没有提升时提前停止；模型始终保存 best epoch 的状态。

- **正则化与模型容量**：
  - 1D-CNN：保留现有残差结构，使用 `weight_decay`（L2 正则），默认 `1e-4`；可在附录设计 “高 / 低 weight_decay” 的 ablation。  
  - 2D/3D-CNN：模型宽度保持当前轻量 ResNet 设计（32/64/128 或 16/32/64），文中点明是“小型残差网络”；如需对比 compact 版本，可在附录再训一组。  
  - 训练脚本中统一支持 `--weight-decay` 参数（默认 `1e-4`），方便 patch 任务与 image 任务保持一致。

- **随机性与多次重复**：
  - 训练脚本均接受 `--seed` 参数（已存在），并在内部固定 `numpy` / `torch` 随机种子。  
  - **论文推荐做法**：对关键任务（尤其是 patch-level `physiological_state` 分类），至少运行 3 次不同 seed 的 2D-CNN（和 1D baseline），将 3 次 run 的 `val/test` 指标做均值 ± 标准差，整理到 `Tab-HSI-ModelMain`。  
  - Snakemake 目前只跑单一 seed；多 seed 重复可通过修改 `config.yaml` 中 `hsi.patch_seed` 并多次触发规则完成，结果在 RULE 中记录清楚即可。

### 2.2 传统 ML 模型的角色

- **RF / XGB / 其它线性模型**主要作为：
  - baseline（对比 HSI CNN）；  
  - 可解释性工具（feature importance / PLS 载荷）。  
- 正文只需要展示：  
  - RF / XGB / best-CNN 在**主任务（通常为 patch-level `physiological_state`）**上的性能对比；  
  - 完整的模型矩阵（所有任务 × 所有模型）集中放在附录表中。

---

## 3. physio_patch_rf 训练时间与 XGB 100% 准确率的检查要点

### 3.1 physio_patch_rf 训练时间

- **数据规模诊断**（建议在本地或 HPC 上运行一个统计脚本）：  
  - 从 `results/hsi/patch_features_with_physiological.tsv` 中统计：  
    - patch 总数；  
    - 每个 `physiological_state` 的 patch 数；  
    - 同期 `patch_features_{patch_target}.tsv`（如 `phase`）的对应统计。  
  - 补充一张 `physio_patch_stats.tsv`，用于说明：  
    - `physiological_state` 任务在 patch 级别的样本数量是否远大于其它任务；  
    - 是否存在极度不平衡的类别。

- **RF 参数与搜索空间**：  
  - 当前 RF 使用 `RandomizedSearchCV` + 大搜索空间（很多 `n_estimators` / `max_depth` 组合），在 patch 级巨大样本量下会显著拖慢。  
  - 对 physio 任务的推荐调整（可通过 `--n-iter`、参数空间收窄等实现）：  
    - 减少 `n_estimators` 搜索范围，例如 `200–800`；  
    - `--n-iter` 从 200 调整到 48–80；  
    - 将 `threads` 设为中等值（如 16），配合合理 `n_jobs`；  
    - 如需进一步加速，可在生成 patch 特征前对每类 patch 进行子采样（例如每类随机取固定上限 patch 数）。

### 3.2 physio_patch_xgb 100% 准确率：数据泄露排查 checklist

在解释 100% 准确率前，必须先确认以下几点（可作为论文方法的“质量控制”小节）：

- **标签是否作为特征泄露**：  
  - 利用 `ml_utils.pick_feature_columns()` 的设定，确认：  
    - `physiological_state` / `phase` / `phase_core` 等字符串标签列不会进入特征（非数值列）。  
    - 特征由光谱列 `R_<nm>` 以及数值指数列构成。  

- **行级划分 vs 样本级划分**：  
  - 传统 ML（包括 physio_patch_rf / physio_patch_xgb）在 patch 特征表上使用 **行级分层划分**（多 patch 可能来自同一植物 / 图像）。  
  - 这一设计会让模型在“同一植株的不同 patch”上训练和测试，导致看起来非常高的 patch 准确率，但**不代表真正的新植株泛化性能**。  
  - 需要在论文中明确指出：  
    - CNN 流程（通过 `patch_index_*` 与 image-level split）是按 image 级划分，**避免了样本级数据泄露**；  
    - 传统 ML patch 模型属于“patch 内部区分能力”的评价，更偏上限估计。

- **XGB 100% 的 sanity check 建议**：  
  - 在 physio patch 特征上做一个极简实验：  
    - 只保留少量典型指数（如 NDVI / NDRE / PRI 等）训练 XGB；  
    - 或做“强正则 + 较小树数 + 较小特征子集”的模型，对比 full-feature 模型的性能变化。  
  - 如果 full-feature 模型接近 100%，而极简模型明显下降，则可以在讨论中解释为：  
    - 多指数 + 全谱在 patch 级高度冗余，足以近乎完美区分 `physiological_state`；  
    - 而不是标签直接泄露为特征。  

- **稳定评估**：  
  - 对 physio_patch_xgb 至少做 3 次不同随机种子（`--random-state`）的重复，记录 test 指标的均值 ± 标准差；  
  - 如果多次重复均稳定接近 100%，需结合混淆矩阵和 ROC/PR 曲线展示“所有类别都被正确分类”，而非仅主类主导。

---

## 4. HSI × 多组学的下游分析规范

### 4.1 HSI embedding 构建

- **来源文件**：  
  - 2D-CNN：`results/hsi/patch/2dcnn_{target}/patch_embeddings_2d.tsv`  
  - 3D-CNN：`results/hsi/patch/3dcnn_{target}/patch_embeddings_3d.tsv`  

- **推荐聚合方式**：  
  - 以 `source_sample_id` 为单位，对其所有 patch 的 embedding 做 `mean`（或 `median`）聚合，得到 image-level embedding。  
  - 生成统一表：`results/hsi/hsi_embedding_by_sample.tsv`，列包括：  
    - `sample_id` / `source_sample_id`（统一为下游多组学的样本 ID）；  
    - `phase` / `phase_core` / `physiological_state` / 温度 / 时间等元数据；  
    - `emb2d_*` / `emb3d_*` 等 embedding 向量列。

### 4.2 与转录组 / 代谢组的关联

- **典型分析（在本地 R 完成，模板见 `hsi_downstream_analysis_guide.md`）**：
  - HSI embedding 与转录组 PCA / WGCNA 模块 eigengene 的相关性热图；  
  - HSI embedding 与代谢组主成分 / 关键代谢通路的相关性热图；  
  - HSI embedding 的 PCA / UMAP 可视化，颜色按 `physiological_state` / `rnaseq_state` / `metabo_state` 着色，观察空间重叠。  

- **论文中需回答的关键问题**：  
  - 哪些 HSI embedding 维度与多组学的主轴或关键模块最强相关？  
  - 在嵌入空间中，组学上定义的几个典型状态是否被清晰分开？  
  - HSI 是否能在早期时间点（如 2h / 6h）预测后续恢复 / 死亡趋势？

---

## 5. 正文必须展示的表格与图（强制清单）

### 5.1 表格（正文）

- **Tab-HSI-DataSummary**  
  - 内容：  
    - image / leaf / patch 三级数据的样本数；  
    - 对应任务（phase_core / physiological_state / rnaseq_state / metabo_state 等）的类别分布。  
  - 来源：  
    - 可由一个专门的统计脚本汇总 `image_features.tsv`、`leaf_features.tsv`、`patch_features_*`、`patch_features_with_physiological.tsv`。

- **Tab-HSI-ModelMain**  
  - 内容：  
    - 主任务（推荐：patch-level `physiological_state` 分类）上 RF / XGB / best-CNN 的 test accuracy、macro-F1、weighted-F1、AUC（如可）  
    - 指标以 “均值 ± SD（n=3–5 runs）” 形式给出。  
  - 来源：  
    - 来自 `results/hsi/patch/physiological_state/all_models_summary.csv` 和多次重复 run 的 `metrics.tsv`。

- **Tab-HSI-MultiomicsCompare**  
  - 内容：  
    - HSI 模型 vs 纯多组学模型 在预测 `physiological_state` / 其它 state 上的性能对比。  
    - 至少给出一个“仅 HSI”、“仅多组学”、“HSI+多组学”的横向表。  

### 5.2 图形（正文）

- **Fig-HSI-IndexTimeSeries**  
  - NDVI + 1–2 个红边 / 氮素相关指数的时间序列图，分组按 `phase_core` 或 `physiological_state`。  

- **Fig-HSI-SpectralCurves**  
  - 不同阶段（对照 / 胁迫 / 恢复）的平均光谱曲线（含置信区间），展示关键波段变化。  

- **Fig-HSI-EmbeddingPCA/UMAP**  
  - 使用 HSI embedding（image-level），颜色按 `physiological_state` / 组学 state，展示在表征空间的分离度。  

- **Fig-HSI-ConfusionMatrix**  
  - 主任务（如 patch-level `physiological_state`）的归一化混淆矩阵（best-CNN 或 best-XGB）。  

- **Fig-HSI-MultiomicsCorrelation**  
  - HSI embedding / 指数 与 多组学特征 / 模块 eigengene 的相关性热图。  

---

## 6. 附录优先展示内容

### 6.1 表格

- 所有 `all_models_summary.csv` 的清洗与合并（image / leaf / patch × phase_core / physiological_state 等），作为若干附录表。  
- CNN 训练超参数表（learning rate, batch size, epoch 上限, early stopping 参数, weight decay 等）。  
- patch 级采样 / stride / 正则化强度变化的 ablation 表（可选）。  

### 6.2 图形

- 各典型模型的训练 / 验证 loss 与 accuracy / F1 曲线（从 `history.tsv` 重绘）。  
- 更多指数分布（箱线图 / 小提琴图）与时间序列热图。  
- 额外的 PCA / UMAP 视图（按温度 / 时间分组）。  

---

## 7. “论文素材包”导出建议

- 在 Snakemake 中新增一个专门的入口（例如 `hsi_paper_materials` 规则，放在单独的 `rules/hsi_paper.smk` 中），只做**已有结果文件的收集与拷贝**，不触发重算：  
  - 统一输出到 `results/hsi/paper_materials/`：  
    - `tables/`：上述 Tab-HSI-*、模型 summary、统计表等 TSV/CSV；  
    - `fig_r_data/`：为 R 绘图准备的中间表 / 嵌入矩阵；  
    - `logs/`：关键训练 log（CNN / XGB / RF 等），用于写方法和讨论。  
- 日常使用：  
  - 在 HPC 上运行 `snakemake -j XX hsi_paper_materials --use-conda`；  
  - 然后 `scp -r results/hsi/paper_materials` 到本地，用 R / Python 生成最终论文图表。

---

## 8. 写作时需要特别强调的要点

- HSI 流程完全脚本化、可重现，**所有参数和规则均在 Snakemake 和脚本中明示**；论文可以引用本 RULE 与仓库文档减轻篇幅。  
- 明确区分：  
  - image-level / leaf-level / patch-level 任务；  
  - phase / phase_core / physiological_state / 组学状态等不同标签层次；  
  - patch 级传统 ML 的“行级划分”与 CNN 的“样本级划分”差异。  
- 在讨论中正面回应：  
  - 样本量有限、patch 高度相关的情况下，如何通过合理分割、早停和正则化减少过拟合；  
  - 为什么 HSI 仍然能在多组学定义的复杂状态上取得稳定、可解释的性能。

