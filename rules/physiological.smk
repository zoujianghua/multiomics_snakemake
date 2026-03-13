# rules/physiological.smk
# 【HSI patch pipeline - 基于生理状态标签的分类】
#
# 功能：
# 1. 复用 patch.smk 中已生成的 patch_index_base_cubes_{param_suffix}.tsv、patch_cubes_{param_suffix}/、patch_features_{run_id}.tsv
# 2. 从 patch_features_{patch_target}.tsv + phase_physiological_state_mapping.tsv 生成 patch_features_with_physiological.tsv
# 3. 从 image_features.tsv + mapping 生成 image_features_with_physio.tsv（临时，不覆盖原文件）
# 4. 为 physiological_state 生成新的 split 和 patch_index（image-level 划分，避免数据泄露）
# 5. 运行所有模型（传统 ML + 深度学习），比较与 phase 分类的效果
#
# 注意：
# - 不修改 image_features.tsv，避免触发整个项目重跑
# - 充分利用 patch.smk 中已完成的数据整理
# - 所有模型代码都复用，不做修改
# - 生理状态标签可能导致类别不平衡（make_split 已做分层采样，模型内部可考虑 class_weight）

HSE = "../envs/hsi_env.yaml"

patch_seed = config.get("hsi", {}).get("patch_seed", 42)
physio_target = "physiological_state"
# PATCH_PARAM_SUFFIXES、PHYSIO_RUN_IDS 来自 patch.smk（include 顺序保证已定义）
physio_image_meta = "results/hsi/ml/image_features_with_physio.tsv"


############################################
# 0. image_features + mapping -> image_features_with_physio（临时，不覆盖原文件）
############################################

rule physio_image_meta:
    """
    从 image_features.tsv 和 phase_physiological_state_mapping.tsv 生成 image_features_with_physio.tsv
    用于 split 和 patch_index，不修改原 image_features.tsv
    """
    input:
        image_features = "results/hsi/image_features.tsv",
        mapping = "config/phase_physiological_state_mapping.tsv",
        code = "scripts/hsi/add_physiological_state_to_image_features.py",
    output:
        img_meta = physio_image_meta,
    conda: HSE
    threads: 8
    resources:
        mem_mb = 8000,
        runtime = 30
    log:
        "logs/physiological/physio_image_meta.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/hsi/ml logs/physiological

        python {input.code} \
          --image-features {input.image_features} \
          --mapping {input.mapping} \
          --out {output.img_meta} \
          > {log} 2>&1

        test -s {output.img_meta}
        """


############################################
# 1. 生成 patch_features_with_physiological.tsv
############################################

rule physio_patch_features:
    """
    从 patch_features_phase_core_{param_suffix}.tsv 和映射表生成 patch_features_with_physiological_{param_suffix}.tsv
    每个 param_suffix 一条链，与 patch.smk 多组参数并行
    """
    input:
        patch_features = "results/hsi/patch_features_phase_{param_suffix}.tsv",
        image_features = "results/hsi/image_features.tsv",
        mapping = "config/phase_physiological_state_mapping.tsv",
        code = "scripts/hsi/add_physiological_state_to_patch_features.py",
    output:
        physio_patch = "results/hsi/patch_features_with_physiological_{param_suffix}.tsv",
    conda: HSE
    threads: 8
    resources:
        mem_mb = 8000,
        runtime = 60
    log:
        "logs/physiological/physio_patch_features_{param_suffix}.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/hsi logs/physiological

        python {input.code} \
          --patch-features {input.patch_features} \
          --image-features {input.image_features} \
          --mapping {input.mapping} \
          --out {output.physio_patch} \
          > {log} 2>&1

        test -s {output.physio_patch}
        """


############################################
# 2. 为 physiological_state 生成 split（基于 image-level，避免数据泄露）
############################################

rule physio_split:
    """
    从 image_features_with_physio.tsv 按 physiological_state 分层划分 train/test
    划分在 image 级别进行，patch 继承所属 image 的 split
    """
    input:
        img_meta = physio_image_meta,
        split_code = "scripts/hsi/make_split_for_target.py",
    output:
        split_tsv = f"results/hsi/ml/split_physiological_state_seed{patch_seed}.tsv",
    conda: HSE
    threads: 8
    resources:
        mem_mb = 8000,
        runtime = 60
    log:
        "logs/physiological/physio_split.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/hsi/ml logs/physiological

        python {input.split_code} \
          --image-meta {input.img_meta} \
          --target-col {physio_target} \
          --seed {patch_seed} \
          --test-size 0.2 \
          --out {output.split_tsv} \
          > {log} 2>&1

        test -s {output.split_tsv}
        """


############################################
# 3. 为 physiological_state 生成 patch_index（复用 base_cubes）
############################################

rule physio_patch_index:
    """
    从 base_cubes 索引和 image_features_with_physio + split 生成任务特定索引
    复用 patch.smk 中已生成的 patch_index_base_cubes_{param_suffix}.tsv
    """
    input:
        base_cubes = "results/hsi/ml/patch_index_base_cubes_{param_suffix}.tsv",
        img_meta = physio_image_meta,
        split = f"results/hsi/ml/split_physiological_state_seed{patch_seed}.tsv",
        code = "scripts/hsi/make_patch_index_for_target.py",
    output:
        index_tsv = f"results/hsi/ml/patch_index_physiological_state_seed{patch_seed}_{{param_suffix}}.tsv",
    conda: HSE
    threads: 8
    resources:
        mem_mb = 8000,
        runtime = 120
    log:
        "logs/physiological/physio_patch_index_{param_suffix}.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p results/hsi/ml logs/physiological

        python {input.code} \
          --base-index {input.base_cubes} \
          --image-meta {input.img_meta} \
          --split {input.split} \
          --target-col {physio_target} \
          --out {output.index_tsv} \
          > {log} 2>&1

        test -s {output.index_tsv}
        """


############################################
# 3. Patch-level 传统 ML（RF / SVM / KNN / LR / XGB / PLS / LDA）
############################################

rule physio_patch_rf:
    input:
        images = "results/hsi/patch_features_with_physiological_{param_suffix}.tsv",
        code   = "scripts/hsi/session_classify_rf.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/patch/physiological_state_{param_suffix}/rf/rf_best_model.joblib",
        summary = "results/hsi/patch/physiological_state_{param_suffix}/rf/rf_summary.csv",
        cm_png  = "results/hsi/patch/physiological_state_{param_suffix}/rf/rf_confusion_matrix_norm.png",
        curve   = "results/hsi/patch/physiological_state_{param_suffix}/rf/rf_hparam_curve_n_estimators.png",
    params:
        target = physio_target,
        outdir = lambda wc: f"results/hsi/patch/physiological_state_{wc.param_suffix}/rf",
    conda: HSE
    threads: 32
    resources:
        mem_mb  = 32000,
        runtime = 24000
    log:
        "logs/physiological/patch_rf_physiological_state_{param_suffix}.log"
    shell:
        r"""
        mkdir -p results/hsi/patch/physiological_state_{wildcards.param_suffix} {params.outdir} logs/physiological
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


rule physio_patch_svm:
    input:
        images = "results/hsi/patch_features_with_physiological_{param_suffix}.tsv",
        code   = "scripts/hsi/session_classify_svm.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/patch/physiological_state_{param_suffix}/svm/svm_best_model.joblib",
        summary = "results/hsi/patch/physiological_state_{param_suffix}/svm/svm_summary.csv",
        cm_png  = "results/hsi/patch/physiological_state_{param_suffix}/svm/svm_confusion_matrix_norm.png",
        curve   = "results/hsi/patch/physiological_state_{param_suffix}/svm/svm_hparam_curve_C.png",
    params:
        target = physio_target,
        outdir = lambda wc: f"results/hsi/patch/physiological_state_{wc.param_suffix}/svm",
    conda: HSE
    threads: 32
    resources:
        mem_mb  = 32000,
        runtime = 24000
    log:
        "logs/physiological/patch_svm_physiological_state_{param_suffix}.log"
    shell:
        r"""
        mkdir -p results/hsi/patch/physiological_state_{wildcards.param_suffix} {params.outdir} logs/physiological
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
          --n-iter 48 \
          > {log} 2>&1

        test -s {output.model}
        test -s {output.summary}
        test -s {output.cm_png} || true
        test -s {output.curve}  || true
        """


rule physio_patch_knn:
    input:
        images = "results/hsi/patch_features_with_physiological_{param_suffix}.tsv",
        code   = "scripts/hsi/session_classify_knn.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/patch/physiological_state_{param_suffix}/knn/knn_best_model.joblib",
        summary = "results/hsi/patch/physiological_state_{param_suffix}/knn/knn_summary.csv",
        cm_png  = "results/hsi/patch/physiological_state_{param_suffix}/knn/knn_confusion_matrix_norm.png",
        curve   = "results/hsi/patch/physiological_state_{param_suffix}/knn/knn_hparam_curve_n_neighbors.png",
    params:
        target = physio_target,
        outdir = lambda wc: f"results/hsi/patch/physiological_state_{wc.param_suffix}/knn",
    conda: HSE
    threads: 16
    resources:
        mem_mb  = 16000,
        runtime = 600
    log:
        "logs/physiological/patch_knn_physiological_state_{param_suffix}.log"
    shell:
        r"""
        mkdir -p results/hsi/patch/physiological_state_{wildcards.param_suffix} {params.outdir} logs/physiological
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


rule physio_patch_lr:
    input:
        images = "results/hsi/patch_features_with_physiological_{param_suffix}.tsv",
        code   = "scripts/hsi/session_classify_lr.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/patch/physiological_state_{param_suffix}/lr/lr_best_model.joblib",
        summary = "results/hsi/patch/physiological_state_{param_suffix}/lr/lr_summary.csv",
        cm_png  = "results/hsi/patch/physiological_state_{param_suffix}/lr/lr_confusion_matrix_norm.png",
        curve   = "results/hsi/patch/physiological_state_{param_suffix}/lr/lr_hparam_curve_C.png",
    params:
        target = physio_target,
        outdir = lambda wc: f"results/hsi/patch/physiological_state_{wc.param_suffix}/lr",
    conda: HSE
    threads: 32
    resources:
        mem_mb  = 32000,
        runtime = 6000
    log:
        "logs/physiological/patch_lr_physiological_state_{param_suffix}.log"
    shell:
        r"""
        mkdir -p results/hsi/patch/physiological_state_{wildcards.param_suffix} {params.outdir} logs/physiological
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


rule physio_patch_xgb:
    input:
        images = "results/hsi/patch_features_with_physiological_{param_suffix}.tsv",
        code   = "scripts/hsi/session_classify_xgb.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/patch/physiological_state_{param_suffix}/xgb/xgb_best_model.joblib",
        summary = "results/hsi/patch/physiological_state_{param_suffix}/xgb/xgb_summary.csv",
        cm_png  = "results/hsi/patch/physiological_state_{param_suffix}/xgb/xgb_confusion_matrix_norm.png",
        curve   = "results/hsi/patch/physiological_state_{param_suffix}/xgb/xgb_hparam_curve_n_estimators.png",
    params:
        target = physio_target,
        outdir = lambda wc: f"results/hsi/patch/physiological_state_{wc.param_suffix}/xgb",
    threads: 16
    resources:
        runtime   = 120000,
        mem_mb    = 32000,
        partition = "gpu-3090",
        gres      = "--gres=gpu:1",
    log:
        "logs/physiological/patch_xgb_physiological_state_{param_suffix}.log"
    shell:
        r"""
        set -euo pipefail

        mkdir -p results/hsi/patch/physiological_state_{wildcards.param_suffix} {params.outdir} logs/physiological

        source /public/home/zoujianghua/miniconda3/etc/profile.d/conda.sh
        conda activate xgb_gpu

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


rule physio_patch_pls:
    input:
        images = "results/hsi/patch_features_with_physiological_{param_suffix}.tsv",
        code   = "scripts/hsi/session_classify_pls.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/patch/physiological_state_{param_suffix}/pls/pls_best_model.joblib",
        summary = "results/hsi/patch/physiological_state_{param_suffix}/pls/pls_summary.csv",
        report  = "results/hsi/patch/physiological_state_{param_suffix}/pls/pls_classification_report.csv",
        cm_png  = "results/hsi/patch/physiological_state_{param_suffix}/pls/pls_confusion_matrix_norm.png",
        hcurve  = "results/hsi/patch/physiological_state_{param_suffix}/pls/pls_hparam_curve_n_components.png",
    params:
        target = physio_target,
        outdir = lambda wc: f"results/hsi/patch/physiological_state_{wc.param_suffix}/pls",
    conda: HSE
    threads: 32
    resources:
        mem_mb  = 32000,
        runtime = 6000
    log:
        "logs/physiological/patch_pls_physiological_state_{param_suffix}.log"
    shell:
        r"""
        mkdir -p results/hsi/patch/physiological_state_{wildcards.param_suffix} {params.outdir} logs/physiological
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


rule physio_patch_lda:
    input:
        images = "results/hsi/patch_features_with_physiological_{param_suffix}.tsv",
        code   = "scripts/hsi/session_classify_lda.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/patch/physiological_state_{param_suffix}/lda/lda_best_model.joblib",
        summary = "results/hsi/patch/physiological_state_{param_suffix}/lda/lda_summary.csv",
        report  = "results/hsi/patch/physiological_state_{param_suffix}/lda/lda_classification_report.csv",
        cm_png  = "results/hsi/patch/physiological_state_{param_suffix}/lda/lda_confusion_matrix_norm.png",
        hcurve  = "results/hsi/patch/physiological_state_{param_suffix}/lda/lda_hparam_curve_shrinkage.png",
    params:
        target = physio_target,
        outdir = lambda wc: f"results/hsi/patch/physiological_state_{wc.param_suffix}/lda",
    conda: HSE
    threads: 32
    resources:
        mem_mb  = 32000,
        runtime = 6000
    log:
        "logs/physiological/patch_lda_physiological_state_{param_suffix}.log"
    shell:
        r"""
        mkdir -p results/hsi/patch/physiological_state_{wildcards.param_suffix} {params.outdir} logs/physiological
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
# 4. Patch-level 1D-CNN（光谱序列）
############################################

rule physio_patch_1dcnn:
    input:
        images = "results/hsi/patch_features_with_physiological_{param_suffix}.tsv",
        code   = "scripts/hsi/hsi_1dcnn.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/patch/1dcnn_image_physiological_state_{param_suffix}/best_model.pt",
        metrics = "results/hsi/patch/1dcnn_image_physiological_state_{param_suffix}/metrics.tsv",
    params:
        outdir = lambda wc: f"results/hsi/patch/1dcnn_image_physiological_state_{wc.param_suffix}",
        target = physio_target,
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
        "logs/physiological/patch_1dcnn_physiological_state_{param_suffix}.log"
    shell:
        r"""
        set -euo pipefail

        mkdir -p {params.outdir} logs/physiological

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
# 5. Patch-level 2D / 3D CNN（空间+光谱 patch）
############################################

rule physio_patch_2dcnn:
    input:
        index = f"results/hsi/ml/patch_index_physiological_state_seed{patch_seed}_{{param_suffix}}.tsv",
        code  = "scripts/hsi/hsi_patch_cnn2d.py",
        ds    = "scripts/hsi/hsi_patch_dataset.py",
    output:
        model      = "results/hsi/patch/2dcnn_physiological_state_{param_suffix}/best_model.pt",
        metrics    = "results/hsi/patch/2dcnn_physiological_state_{param_suffix}/metrics.tsv",
        run_params = "results/hsi/patch/2dcnn_physiological_state_{param_suffix}/run_params.tsv",
    params:
        outdir = lambda wc: f"results/hsi/patch/2dcnn_physiological_state_{wc.param_suffix}",
        seed   = 42,
        batch  = 4,
        epochs = 200,
    threads: 32
    resources:
        mem_mb   = 32000,
        runtime  = 144000,
        partition = "gpu-3090",
        gres     = "--gres=gpu:1",
    log:
        "logs/physiological/patch_2dcnn_physiological_state_{param_suffix}.log"
    shell:
        r"""
        set -euo pipefail

        mkdir -p {params.outdir} logs/physiological

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


rule physio_patch_3dcnn:
    input:
        index = f"results/hsi/ml/patch_index_physiological_state_seed{patch_seed}_{{param_suffix}}.tsv",
        code  = "scripts/hsi/hsi_patch_cnn3d.py",
        ds    = "scripts/hsi/hsi_patch_dataset.py",
    output:
        model      = "results/hsi/patch/3dcnn_physiological_state_{param_suffix}/best_model.pt",
        metrics    = "results/hsi/patch/3dcnn_physiological_state_{param_suffix}/metrics.tsv",
        run_params = "results/hsi/patch/3dcnn_physiological_state_{param_suffix}/run_params.tsv",
    params:
        outdir = lambda wc: f"results/hsi/patch/3dcnn_physiological_state_{wc.param_suffix}",
        seed   = 42,
        batch  = 4,
        epochs = 200,
    threads: 16
    resources:
        mem_mb   = 32000,
        runtime  = 144000,
        partition = "gpu-3090",
        gres     = "--gres=gpu:1",
    log:
        "logs/physiological/patch_3dcnn_physiological_state_{param_suffix}.log"
    shell:
        r"""
        set -euo pipefail

        mkdir -p {params.outdir} logs/physiological

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
# 6. Patch 模型结果汇总
############################################

rule physio_patch_merge:
    """
    汇总 patch_features_with_physiological.tsv 上的各模型结果
    """
    input:
        merge_code = "scripts/hsi/session_classify_merge.py",
        rf   = "results/hsi/patch/physiological_state_{param_suffix}/rf/rf_summary.csv",
        svm  = "results/hsi/patch/physiological_state_{param_suffix}/svm/svm_summary.csv",
        knn  = "results/hsi/patch/physiological_state_{param_suffix}/knn/knn_summary.csv",
        lr   = "results/hsi/patch/physiological_state_{param_suffix}/lr/lr_summary.csv",
        xgb  = "results/hsi/patch/physiological_state_{param_suffix}/xgb/xgb_summary.csv",
        pls  = "results/hsi/patch/physiological_state_{param_suffix}/pls/pls_summary.csv",
        lda  = "results/hsi/patch/physiological_state_{param_suffix}/lda/lda_summary.csv",
    output:
        summary = "results/hsi/patch/physiological_state_{param_suffix}/all_models_summary.csv",
        f1_png  = "results/hsi/patch/physiological_state_{param_suffix}/all_models_f1_bar.png",
        acc_png = "results/hsi/patch/physiological_state_{param_suffix}/all_models_accuracy_bar.png",
    conda: HSE
    threads: 8
    resources:
        mem_mb  = 8000,
        runtime = 60
    log:
        "logs/physiological/patch_merge_physiological_state_{param_suffix}.log"
    shell:
        r"""
        mkdir -p results/hsi/patch/physiological_state_{wildcards.param_suffix} logs/physiological
        export MPLBACKEND=Agg

        python {input.merge_code} \
          --summaries {input.rf} {input.svm} {input.knn} {input.lr} {input.xgb} {input.pls} {input.lda} \
          --outdir results/hsi/patch/physiological_state_{wildcards.param_suffix} \
          > {log} 2>&1 || true

        test -s {output.summary} || true
        test -s {output.f1_png}  || true
        test -s {output.acc_png} || true
        """
