# rules/patch.smk
# 【HSI patch pipeline - Patch 级深度学习与机器学习流水线】
#
# 重构后的 patch 流水线：
# 1. 通用几何索引（build_patch_index_base）-> patch_index_base_{param_suffix}.tsv
# 2. Patch 集合文件（build_patch_cubes）-> patch_cubes_{param_suffix}/{sample_id}_patches.npz + patch_index_base_cubes_{param_suffix}.tsv
# 3. 任务特定划分（make_split_for_target）-> split_{target}_seed{seed}.tsv
# 4. 任务特定索引（make_patch_index_for_target）-> patch_index_{target}_seed{seed}.tsv
# 5. 深度学习训练（2D/3D CNN）使用 patch_cubes，大幅减少 I/O 开销
# 6. 传统 ML（RF/SVM/XGB 等）使用 patch_features_{target}.tsv
#
# target 从 config["hsi"]["patch_target"] 读取，例如：
#   hsi:
#     patch_target: "phase_core"
#     patch_seed: 42
#
# 要切换 target（如 metabo_state / rnaseq_state），只需修改 config.yaml

HSE = "../envs/hsi_env.yaml"

patch_seed = config.get("hsi", {}).get("patch_seed", 42)
patch_size = config.get("hsi", {}).get("patch_size", 32)
# 单组兼容（无 patch_param_sets 时）
_def_stride = config.get("hsi", {}).get("patch_stride", 16)
_def_maxp = config.get("hsi", {}).get("max_patches_per_image", 200)
_def_target = config.get("hsi", {}).get("patch_target", "phase_core")

_param_sets = config.get("hsi", {}).get("patch_param_sets")
if _param_sets:
    PATCH_PARAM_SUFFIXES = list(_param_sets.keys())
else:
    PATCH_PARAM_SUFFIXES = [f"s{_def_stride}_maxp{_def_maxp}"]

_targets = config.get("hsi", {}).get("patch_targets")
if _targets:
    PATCH_TARGETS = list(_targets)
else:
    PATCH_TARGETS = [_def_target]

RUN_IDS = [f"{t}_{ps}" for t in PATCH_TARGETS for ps in PATCH_PARAM_SUFFIXES]
PHYSIO_RUN_IDS = [f"physiological_state_{ps}" for ps in PATCH_PARAM_SUFFIXES]

# 约束 wildcards，避免规则歧义
# - param_suffix: config 格式 s数字_maxp数字，避免与 hsi_patch_cubes 的 base_cubes_{param_suffix} 歧义
# - patch_target: 仅主流程标签（phase|phase_core），避免与 physiological 的 patch_features_with_physiological_* 歧义
wildcard_constraints:
    param_suffix="s[0-9]+_maxp[0-9]+",
    patch_target="|".join(PATCH_TARGETS)


def get_patch_stride_maxp(param_suffix):
    """从 config 或 param_suffix 解析 stride, max_patches_per_image。"""
    ps = config.get("hsi", {}).get("patch_param_sets", {})
    if param_suffix in ps:
        return ps[param_suffix].get("stride", 16), ps[param_suffix].get("max_patches_per_image", 200)
    import re
    m = re.match(r"s(\d+)_maxp(\d+)", str(param_suffix))
    if m:
        return int(m.group(1)), int(m.group(2))
    return _def_stride, _def_maxp


############################################
# 0. 通用几何索引生成
############################################

rule hsi_patch_index_base:
    """
    生成通用的 patch 几何索引（不含 target/split）
    从 image_features.tsv 读取 sample_id 和 cube_npz，通过滑窗生成候选 patch
    同时写出 patch_index_run_params_{param_suffix}.tsv 记录几何参数，便于追溯。
    """
    input:
        images = "results/hsi/image_features.tsv",
        code   = "scripts/hsi/build_patch_index_base.py",
    output:
        base_index   = "results/hsi/ml/patch_index_base_{param_suffix}.tsv",
        run_params   = "results/hsi/ml/patch_index_run_params_{param_suffix}.tsv",
    params:
        stride = lambda wc: get_patch_stride_maxp(wc.param_suffix)[0],
        max_patches = lambda wc: get_patch_stride_maxp(wc.param_suffix)[1],
    conda: HSE
    threads: 16
    resources:
        mem_mb  = 16000,
        runtime = 1800
    log:
        "logs/patch/hsi_patch_index_base_{param_suffix}.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/hsi/ml logs/patch

        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1

        python {input.code} \
          --images {input.images} \
          --out {output.base_index} \
          --run-params-out {output.run_params} \
          --patch-size {patch_size} \
          --stride {params.stride} \
          --min-mask-frac 1.0 \
          --max-patches-per-image {params.max_patches} \
          > {log} 2>&1

        test -s {output.base_index}
        test -s {output.run_params}
        """


############################################
# 1. Patch 集合文件生成（一 cube 一个文件）
############################################

rule hsi_patch_cubes:
    """
    将 base 索引中的所有 patch 按 cube_npz 分组，生成「一 cube 一个 patch 集合文件」
    大幅减少 I/O 开销，提升深度学习训练速度
    """
    input:
        base_index = "results/hsi/ml/patch_index_base_{param_suffix}.tsv",
        code       = "scripts/hsi/build_patch_cubes.py",
    output:
        cubes_index = "results/hsi/ml/patch_index_base_cubes_{param_suffix}.tsv",
        cubes_dir   = directory("results/hsi/patch_cubes_{param_suffix}"),
    conda: HSE
    threads: 16
    resources:
        mem_mb  = 32000,
        runtime = 3600
    log:
        "logs/patch/hsi_patch_cubes_{param_suffix}.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p {output.cubes_dir} results/hsi/ml logs/patch

        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1

        python {input.code} \
          --base-index {input.base_index} \
          --outdir {output.cubes_dir} \
          --dtype float32 \
          --workers {threads} \
          --out-index {output.cubes_index} \
          > {log} 2>&1

        test -s {output.cubes_index}
        test -d {output.cubes_dir}
        """


############################################
# 2. image-level 分层划分 (patch_target)
############################################

rule hsi_patch_split:
    """
    按指定 target 列生成分层 train/test 划分
    使用 make_split_for_target.py，支持 phase_core / phase / physiological_state 等
    """
    input:
        image_meta = "results/hsi/image_features.tsv",
        code       = "scripts/hsi/make_split_for_target.py",
    output:
        split_tsv = f"results/hsi/ml/split_{{patch_target}}_seed{patch_seed}.tsv",
    conda: HSE
    threads: 8
    params:
        seed = patch_seed,
    resources:
        mem_mb  = 8000,
        runtime = 60
    log:
        "logs/patch/hsi_patch_split_{patch_target}.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/hsi/ml logs/patch

        python {input.code} \
          --image-meta {input.image_meta} \
          --target-col {wildcards.patch_target} \
          --seed {params.seed} \
          --test-size 0.2 \
          --out {output.split_tsv} \
          > {log} 2>&1

        test -s {output.split_tsv}
        """


############################################
# 3. 任务特定索引生成
############################################

rule hsi_patch_make_index:
    """
    从 base_cubes 索引和 image metadata + split 生成任务特定索引
    包含 cube_patch_npz, patch_idx, split, target 等信息
    """
    input:
        base_cubes = "results/hsi/ml/patch_index_base_cubes_{param_suffix}.tsv",
        image_meta = "results/hsi/image_features.tsv",
        split      = lambda wc: f"results/hsi/ml/split_{wc.patch_target}_seed{patch_seed}.tsv",
        code       = "scripts/hsi/make_patch_index_for_target.py",
    output:
        index_tsv = f"results/hsi/ml/patch_index_{{patch_target}}_seed{patch_seed}_{{param_suffix}}.tsv",
    conda: HSE
    threads: 8
    resources:
        mem_mb  = 8000,
        runtime = 120
    log:
        "logs/patch/hsi_patch_make_index_{patch_target}_{param_suffix}.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/hsi/ml logs/patch

        python {input.code} \
          --base-index {input.base_cubes} \
          --image-meta {input.image_meta} \
          --split {input.split} \
          --target-col {wildcards.patch_target} \
          --out {output.index_tsv} \
          > {log} 2>&1

        test -s {output.index_tsv}
        """


############################################
# 4. patch_index -> raw_patch_features（传统 ML 用）
############################################
# 注意：此步骤仅用于传统 ML（RF/SVM/XGB 等），深度学习直接使用 patch_cubes


############################################
# 3. patch_index -> raw_patch_features_<patch_target>.tsv
############################################

rule hsi_patch_features_raw:
    """
    从 patch_index_{patch_target}_seed{patch_seed}_{param_suffix}.tsv 生成 raw_patch_features_{patch_target}_{param_suffix}.tsv
    用于传统 ML 模型（RF/SVM/XGB 等）
    """
    input:
        index = lambda wc: f"results/hsi/ml/patch_index_{wc.patch_target}_seed{patch_seed}_{wc.param_suffix}.tsv",
        code  = "scripts/hsi/build_patch_features.py",
    output:
        raw_patch = "results/hsi/raw_patch_features_{patch_target}_{param_suffix}.tsv",
    params:
        target_col = lambda wc: wc.patch_target,
        sg_window  = 9,
        sg_poly    = 2,
        workers    = 16
    conda: HSE
    threads: 16
    resources:
        mem_mb  = 24000,
        runtime = 600
    log:
        "logs/patch/hsi_patch_features_raw_{patch_target}_{param_suffix}.log"
    shell:
        r"""
        mkdir -p results/hsi logs/patch

        python {input.code} \
          --index {input.index} \
          --outdir results/hsi \
          --out-tsv {output.raw_patch} \
          --target-col {params.target_col} \
          --sg-window {params.sg_window} \
          --sg-poly {params.sg_poly} \
          --workers {params.workers} \
          > {log} 2>&1

        test -s {output.raw_patch}
        """


############################################
# 3. raw_patch -> clean_patch -> patch_features
############################################

rule hsi_patch_clean_and_indices:
    """
    仿照 hsi_leaf_clean_and_indices：
    - 清洗异常 patch
    - 计算植被指数
    """
    input:
        raw   = "results/hsi/raw_patch_features_{patch_target}_{param_suffix}.tsv",
        code1 = "scripts/hsi/clean_image_features.py",
        code2 = "scripts/hsi/add_indices.py",
    output:
        clean = "results/hsi/clean_patch_features_{patch_target}_{param_suffix}.tsv",
        final = "results/hsi/patch_features_{patch_target}_{param_suffix}.tsv",
    conda: HSE
    threads: 16
    resources:
        mem_mb  = 16000,
        runtime = 600
    log:
        "logs/patch/hsi_patch_clean_and_indices_{patch_target}_{param_suffix}.log"
    shell:
        r"""
        mkdir -p results/hsi logs/patch

        # 1) 清洗异常 patch（使用 patch_target 作为分组列）
        python {input.code1} \
          --images-raw {input.raw} \
          --out {output.clean} \
          --min-r800 0.04 \
          --max-z 3.0 \
          --group-col {wildcards.patch_target} \
          > {log} 2>&1

        # 2) 计算全部植被指数
        python {input.code2} \
          --images {output.clean} \
          --out {output.final} \
          >> {log} 2>&1

        test -s {output.clean}
        test -s {output.final}
        """


############################################
# 3.5. Patch 特征表多组学导出
############################################

rule hsi_patch_features_omics:
    """
    从 patch_features_{patch_target}_{param_suffix}.tsv + image_features.tsv
    生成可用于多组学关联分析的 patch 特征表 patch_features_omics_{patch_target}_{param_suffix}.tsv
    （附加 phase / temp / time / session_id 等 image-level 信息）
    """
    input:
        patch  = "results/hsi/patch_features_{patch_target}_{param_suffix}.tsv",
        images = "results/hsi/image_features.tsv",
        code   = "scripts/hsi/patch_features_omics.py",
    output:
        tsv    = "results/hsi/patch_features_omics_{patch_target}_{param_suffix}.tsv",
    conda: HSE
    threads: 8
    resources:
        mem_mb  = 8000,
        runtime = 600
    log:
        "logs/patch/hsi_patch_features_omics_{patch_target}_{param_suffix}.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p $(dirname {output.tsv}) logs/patch

        export MPLBACKEND=Agg
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1

        python {input.code} \
          --patch {input.patch} \
          --images {input.images} \
          --out {output.tsv} \
          > {log} 2>&1

        test -s {output.tsv}
        """


############################################
# 4. Patch-level 传统 ML（RF / SVM / KNN / LR / XGB / PLS / LDA）
############################################

rule patch_rf:
    input:
        images = "results/hsi/patch_features_{patch_target}_{param_suffix}.tsv",
        code   = "scripts/hsi/session_classify_rf.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/patch/{patch_target}_{param_suffix}/rf/rf_best_model.joblib",
        summary = "results/hsi/patch/{patch_target}_{param_suffix}/rf/rf_summary.csv",
        cm_png  = "results/hsi/patch/{patch_target}_{param_suffix}/rf/rf_confusion_matrix_norm.png",
        curve   = "results/hsi/patch/{patch_target}_{param_suffix}/rf/rf_hparam_curve_n_estimators.png",
    params:
        target = lambda wc: wc.patch_target,
        outdir = lambda wc: f"results/hsi/patch/{wc.patch_target}_{wc.param_suffix}/rf",
    conda: HSE
    threads: 32
    resources:
        mem_mb  = 32000,
        runtime = 2400
    log:
        "logs/patch/patch_rf_{patch_target}_{param_suffix}.log"
    shell:
        r"""
        mkdir -p {params.outdir} logs/patch
        export MPLBACKEND=Agg
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

        python {input.code} \
          --images {input.images} \
          --target {params.target} \
          --outdir {params.outdir} \
          --test-size 0.2 \
          --cv-folds 5 \
          --random-state 42 \
          --n-jobs {threads} \
          > {log} 2>&1

        test -s {output.model}
        test -s {output.summary}
        test -s {output.cm_png} || true
        test -s {output.curve}  || true
        """


rule patch_svm:
    input:
        images = "results/hsi/patch_features_{patch_target}_{param_suffix}.tsv",
        code   = "scripts/hsi/session_classify_svm.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/patch/{patch_target}_{param_suffix}/svm/svm_best_model.joblib",
        summary = "results/hsi/patch/{patch_target}_{param_suffix}/svm/svm_summary.csv",
        cm_png  = "results/hsi/patch/{patch_target}_{param_suffix}/svm/svm_confusion_matrix_norm.png",
        curve   = "results/hsi/patch/{patch_target}_{param_suffix}/svm/svm_hparam_curve_C.png",
    params:
        target = lambda wc: wc.patch_target,
        outdir = lambda wc: f"results/hsi/patch/{wc.patch_target}_{wc.param_suffix}/svm",
    conda: HSE
    threads: 32
    resources:
        mem_mb  = 32000,
        runtime = 2400
    log:
        "logs/patch/patch_svm_{patch_target}_{param_suffix}.log"
    shell:
        r"""
        mkdir -p {params.outdir} logs/patch
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export MPLBACKEND=Agg

        python {input.code} \
          --images {input.images} \
          --target {params.target} \
          --outdir {params.outdir} \
          --test-size 0.2 \
          --cv-folds 5 \
          --random-state 42 \
          --n-jobs {threads} \
          > {log} 2>&1

        test -s {output.summary}
        test -s {output.cm_png} || true
        test -s {output.curve}  || true
        touch {output.model}
        """


rule patch_knn:
    input:
        images = "results/hsi/patch_features_{patch_target}_{param_suffix}.tsv",
        code   = "scripts/hsi/session_classify_knn.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/patch/{patch_target}_{param_suffix}/knn/knn_best_model.joblib",
        summary = "results/hsi/patch/{patch_target}_{param_suffix}/knn/knn_summary.csv",
        cm_png  = "results/hsi/patch/{patch_target}_{param_suffix}/knn/knn_confusion_matrix_norm.png",
        curve   = "results/hsi/patch/{patch_target}_{param_suffix}/knn/knn_hparam_curve_n_neighbors.png",
    params:
        target = lambda wc: wc.patch_target,
        outdir = lambda wc: f"results/hsi/patch/{wc.patch_target}_{wc.param_suffix}/knn",
    conda: HSE
    threads: 16
    resources:
        mem_mb  = 16000,
        runtime = 600
    log:
        "logs/patch/patch_knn_{patch_target}_{param_suffix}.log"
    shell:
        r"""
        mkdir -p {params.outdir} logs/patch
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export MPLBACKEND=Agg

        python {input.code} \
          --images {input.images} \
          --target {params.target} \
          --outdir {params.outdir} \
          --test-size 0.2 \
          --cv-folds 5 \
          --random-state 42 \
          --n-jobs {threads} \
          > {log} 2>&1

        test -s {output.summary}
        test -s {output.cm_png} || true
        test -s {output.curve}  || true
        touch {output.model}
        """


rule patch_lr:
    input:
        images = "results/hsi/patch_features_{patch_target}_{param_suffix}.tsv",
        code   = "scripts/hsi/session_classify_lr.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/patch/{patch_target}_{param_suffix}/lr/lr_best_model.joblib",
        summary = "results/hsi/patch/{patch_target}_{param_suffix}/lr/lr_summary.csv",
        cm_png  = "results/hsi/patch/{patch_target}_{param_suffix}/lr/lr_confusion_matrix_norm.png",
        curve   = "results/hsi/patch/{patch_target}_{param_suffix}/lr/lr_hparam_curve_C.png",
    params:
        target = lambda wc: wc.patch_target,
        outdir = lambda wc: f"results/hsi/patch/{wc.patch_target}_{wc.param_suffix}/lr",
    conda: HSE
    threads: 32
    resources:
        mem_mb  = 32000,
        runtime = 600
    log:
        "logs/patch/patch_lr_{patch_target}_{param_suffix}.log"
    shell:
        r"""
        mkdir -p {params.outdir} logs/patch
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export MPLBACKEND=Agg

        python {input.code} \
          --images {input.images} \
          --target {params.target} \
          --outdir {params.outdir} \
          --test-size 0.2 \
          --cv-folds 5 \
          --random-state 42 \
          --n-jobs {threads} \
          > {log} 2>&1

        test -s {output.summary}
        test -s {output.cm_png} || true
        test -s {output.curve}  || true
        touch {output.model}
        """


rule patch_xgb:
    input:
        images = "results/hsi/patch_features_{patch_target}_{param_suffix}.tsv",
        code   = "scripts/hsi/session_classify_xgb.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/patch/{patch_target}_{param_suffix}/xgb/xgb_best_model.joblib",
        summary = "results/hsi/patch/{patch_target}_{param_suffix}/xgb/xgb_summary.csv",
        cm_png  = "results/hsi/patch/{patch_target}_{param_suffix}/xgb/xgb_confusion_matrix_norm.png",
        curve   = "results/hsi/patch/{patch_target}_{param_suffix}/xgb/xgb_hparam_curve_n_estimators.png",
    params:
        target = lambda wc: wc.patch_target,
        outdir = lambda wc: f"results/hsi/patch/{wc.patch_target}_{wc.param_suffix}/xgb",
    threads: 16
    resources:
        runtime   = 120000,
        mem_mb    = 32000,
        partition = "gpu-3090",
        gres      = "--gres=gpu:1",
    log:
        "logs/patch/patch_xgb_{patch_target}_{param_suffix}.log"
    shell:
        r"""
        set -euo pipefail

        mkdir -p {params.outdir} logs/patch

        # 激活 xgb_gpu 环境
        source /public/home/zoujianghua/miniconda3/etc/profile.d/conda.sh
        conda activate xgb_gpu

        # 让 conda 自己的 libstdc++ 优先
        export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"

        echo "NODE=$(hostname)"
        echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-<empty>}}"
        nvidia-smi || true

        # 线程数和绘图后端
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export MPLBACKEND=Agg

        python {input.code} \
          --images {input.images} \
          --target {params.target} \
          --outdir {params.outdir} \
          --test-size 0.2 \
          --cv-folds 5 \
          --random-state 42 \
          --n-jobs {threads} \
          --xgb-gpu \
          > {log} 2>&1

        test -s {output.model}
        test -s {output.summary}
        test -s {output.cm_png} || true
        test -s {output.curve}  || true
        """


rule patch_pls:
    input:
        images = "results/hsi/patch_features_{patch_target}_{param_suffix}.tsv",
        code   = "scripts/hsi/session_classify_pls.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/patch/{patch_target}_{param_suffix}/pls/pls_best_model.joblib",
        summary = "results/hsi/patch/{patch_target}_{param_suffix}/pls/pls_summary.csv",
        report  = "results/hsi/patch/{patch_target}_{param_suffix}/pls/pls_classification_report.csv",
        cm_png  = "results/hsi/patch/{patch_target}_{param_suffix}/pls/pls_confusion_matrix_norm.png",
        hcurve  = "results/hsi/patch/{patch_target}_{param_suffix}/pls/pls_hparam_curve_n_components.png",
    params:
        target = lambda wc: wc.patch_target,
        outdir = lambda wc: f"results/hsi/patch/{wc.patch_target}_{wc.param_suffix}/pls",
    conda: HSE
    threads: 32
    resources:
        mem_mb  = 32000,
        runtime = 600
    log:
        "logs/patch/patch_pls_{patch_target}_{param_suffix}.log"
    shell:
        r"""
        mkdir -p {params.outdir} logs/patch
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export MPLBACKEND=Agg

        python {input.code} \
          --images {input.images} \
          --target {params.target} \
          --outdir {params.outdir} \
          --test-size 0.20 \
          --cv-folds 5 \
          --random-state 42 \
          --n-jobs {threads} \
          > {log} 2>&1

        test -s {output.model}
        test -s {output.summary}
        test -s {output.report}
        test -s {output.cm_png}
        test -s {output.hcurve} || true
        """


rule patch_lda:
    input:
        images = "results/hsi/patch_features_{patch_target}_{param_suffix}.tsv",
        code   = "scripts/hsi/session_classify_lda.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/patch/{patch_target}_{param_suffix}/lda/lda_best_model.joblib",
        summary = "results/hsi/patch/{patch_target}_{param_suffix}/lda/lda_summary.csv",
        report  = "results/hsi/patch/{patch_target}_{param_suffix}/lda/lda_classification_report.csv",
        cm_png  = "results/hsi/patch/{patch_target}_{param_suffix}/lda/lda_confusion_matrix_norm.png",
        hcurve  = "results/hsi/patch/{patch_target}_{param_suffix}/lda/lda_hparam_curve_shrinkage.png",
    params:
        target = lambda wc: wc.patch_target,
        outdir = lambda wc: f"results/hsi/patch/{wc.patch_target}_{wc.param_suffix}/lda",
    conda: HSE
    threads: 32
    resources:
        mem_mb  = 32000,
        runtime = 600
    log:
        "logs/patch/patch_lda_{patch_target}_{param_suffix}.log"
    shell:
        r"""
        mkdir -p {params.outdir} logs/patch
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export MPLBACKEND=Agg

        python {input.code} \
          --images {input.images} \
          --target {params.target} \
          --outdir {params.outdir} \
          --test-size 0.20 \
          --cv-folds 5 \
          --random-state 42 \
          --n-jobs {threads} \
          > {log} 2>&1

        test -s {output.model}
        test -s {output.summary}
        test -s {output.report}
        test -s {output.cm_png}
        test -s {output.hcurve} || true
        """


############################################
# 5. Patch-level 1D-CNN（光谱序列）
############################################

rule patch_1dcnn:
    input:
        images = "results/hsi/patch_features_{patch_target}_{param_suffix}.tsv",
        code   = "scripts/hsi/hsi_1dcnn.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/patch/1dcnn_image_{patch_target}_{param_suffix}/best_model.pt",
        metrics = "results/hsi/patch/1dcnn_image_{patch_target}_{param_suffix}/metrics.tsv",
    params:
        outdir = lambda wc: f"results/hsi/patch/1dcnn_image_{wc.patch_target}_{wc.param_suffix}",
        target = lambda wc: wc.patch_target,
        seed   = 42,
        batch  = 64,
        epochs = 200,
    threads: 16
    resources:
        mem_mb   = 16000,
        runtime  = 1440,
        partition = "gpu-3090",
        gres     = "--gres=gpu:1",
    log:
        "logs/patch/patch_1dcnn_{patch_target}_{param_suffix}.log"
    shell:
        r"""
        set -euo pipefail

        mkdir -p {params.outdir} logs/patch

        source /public/home/zoujianghua/miniconda3/etc/profile.d/conda.sh
        conda activate hsi_dl

        export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"
        echo "NODE=$(hostname)"
        echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-<empty>}}"
        nvidia-smi || true

        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export MPLBACKEND=Agg

        python {input.code} \
          --images {input.images} \
          --target {params.target} \
          --outdir {params.outdir} \
          --batch-size {params.batch} \
          --epochs {params.epochs} \
          --seed {params.seed} \
          > {log} 2>&1

        test -s {output.model}
        test -s {output.metrics}
        """


############################################
# 6. Patch-level 2D / 3D CNN（空间+光谱 patch）
############################################

rule patch_2dcnn:
    input:
        index = lambda wc: f"results/hsi/ml/patch_index_{wc.patch_target}_seed{patch_seed}_{wc.param_suffix}.tsv",
        code  = "scripts/hsi/hsi_patch_cnn2d.py",
        ds    = "scripts/hsi/hsi_patch_dataset.py",
    output:
        model      = "results/hsi/patch/2dcnn_{patch_target}_{param_suffix}/best_model.pt",
        metrics    = "results/hsi/patch/2dcnn_{patch_target}_{param_suffix}/metrics.tsv",
        run_params = "results/hsi/patch/2dcnn_{patch_target}_{param_suffix}/run_params.tsv",
    params:
        outdir = lambda wc: f"results/hsi/patch/2dcnn_{wc.patch_target}_{wc.param_suffix}",
        seed   = 42,
        batch  = 64,
        epochs = 200,
    threads: 16
    resources:
        mem_mb   = 32000,
        runtime  = 14400,
        partition = "gpu-3090",
        gres     = "--gres=gpu:1",
    log:
        "logs/patch/patch_2dcnn_{patch_target}_{param_suffix}.log"
    shell:
        r"""
        set -euo pipefail

        mkdir -p {params.outdir} logs/patch

        source /public/home/zoujianghua/miniconda3/etc/profile.d/conda.sh
        conda activate hsi_dl

        export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"
        echo "NODE=$(hostname)"
        echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-<empty>}}"
        nvidia-smi || true

        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export MPLBACKEND=Agg

        python {input.code} \
          --index-tsv {input.index} \
          --outdir {params.outdir} \
          --batch-size {params.batch} \
          --epochs {params.epochs} \
          --seed {params.seed} \
          --amp \
          > {log} 2>&1

        test -s {output.model}
        test -s {output.metrics}
        test -s {output.run_params}
        """


rule patch_3dcnn:
    input:
        index = lambda wc: f"results/hsi/ml/patch_index_{wc.patch_target}_seed{patch_seed}_{wc.param_suffix}.tsv",
        code  = "scripts/hsi/hsi_patch_cnn3d.py",
        ds    = "scripts/hsi/hsi_patch_dataset.py",
    output:
        model      = "results/hsi/patch/3dcnn_{patch_target}_{param_suffix}/best_model.pt",
        metrics    = "results/hsi/patch/3dcnn_{patch_target}_{param_suffix}/metrics.tsv",
        run_params = "results/hsi/patch/3dcnn_{patch_target}_{param_suffix}/run_params.tsv",
    params:
        outdir = lambda wc: f"results/hsi/patch/3dcnn_{wc.patch_target}_{wc.param_suffix}",
        seed   = 42,
        batch  = 16,
        epochs = 200,
    threads: 16
    resources:
        mem_mb   = 32000,
        runtime  = 14400,
        partition = "gpu-3090",
        gres     = "--gres=gpu:1",
    log:
        "logs/patch/patch_3dcnn_{patch_target}_{param_suffix}.log"
    shell:
        r"""
        set -euo pipefail

        mkdir -p {params.outdir} logs/patch

        source /public/home/zoujianghua/miniconda3/etc/profile.d/conda.sh
        conda activate hsi_dl

        export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"
        echo "NODE=$(hostname)"
        echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-<empty>}}"
        nvidia-smi || true

        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export MPLBACKEND=Agg

        python {input.code} \
          --index-tsv {input.index} \
          --outdir {params.outdir} \
          --batch-size {params.batch} \
          --epochs {params.epochs} \
          --seed {params.seed} \
          --amp \
          > {log} 2>&1

        test -s {output.model}
        test -s {output.metrics}
        test -s {output.run_params}
        """


############################################
# 7. Patch 模型结果汇总
############################################

rule patch_merge:
    """
    汇总 patch_features_{patch_target}_{param_suffix}.tsv 上的各模型结果
    """
    input:
        merge_code = "scripts/hsi/session_classify_merge.py",
        rf   = "results/hsi/patch/{patch_target}_{param_suffix}/rf/rf_summary.csv",
        svm  = "results/hsi/patch/{patch_target}_{param_suffix}/svm/svm_summary.csv",
        knn  = "results/hsi/patch/{patch_target}_{param_suffix}/knn/knn_summary.csv",
        lr   = "results/hsi/patch/{patch_target}_{param_suffix}/lr/lr_summary.csv",
        xgb  = "results/hsi/patch/{patch_target}_{param_suffix}/xgb/xgb_summary.csv",
        pls  = "results/hsi/patch/{patch_target}_{param_suffix}/pls/pls_summary.csv",
        lda  = "results/hsi/patch/{patch_target}_{param_suffix}/lda/lda_summary.csv",
    output:
        summary = "results/hsi/patch/{patch_target}_{param_suffix}/all_models_summary.csv",
        f1_png  = "results/hsi/patch/{patch_target}_{param_suffix}/all_models_f1_bar.png",
        acc_png = "results/hsi/patch/{patch_target}_{param_suffix}/all_models_accuracy_bar.png",
    conda: HSE
    threads: 8
    resources:
        mem_mb  = 8000,
        runtime = 60
    log:
        "logs/patch/patch_merge_{patch_target}_{param_suffix}.log"
    shell:
        r"""
        mkdir -p results/hsi/patch/{wildcards.patch_target}_{wildcards.param_suffix} logs/patch
        export MPLBACKEND=Agg

        python {input.merge_code} \
          --summaries {input.rf} {input.svm} {input.knn} {input.lr} {input.xgb} {input.pls} {input.lda} \
          --outdir results/hsi/patch/{wildcards.patch_target}_{wildcards.param_suffix} \
          > {log} 2>&1 || true

        test -s {output.summary} || true
        test -s {output.f1_png}  || true
        test -s {output.acc_png} || true
        """

