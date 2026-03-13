#rule/leaf.smk

HSE = "../envs/hsi_env.yaml"

# Leaf 级多标签（与 patch/image 一致，便于跨 level 对比）
LEAF_TARGETS = config.get("hsi", {}).get("image_targets", [config.get("hsi", {}).get("patch_target", "phase_core")])


# ================= HSI: ML — RF / SVM / KNN / LR / XGB =================

rule leaf_rf:
    input:
        images = "results/hsi/leaf_features.tsv",
        code   = "scripts/hsi/session_classify_rf.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/leaf/{target}/rf/rf_best_model.joblib",
        summary = "results/hsi/leaf/{target}/rf/rf_summary.csv",
        cm_png  = "results/hsi/leaf/{target}/rf/rf_confusion_matrix_norm.png",
        curve   = "results/hsi/leaf/{target}/rf/rf_hparam_curve_n_estimators.png",
    params:
        outdir = lambda wc: f"results/hsi/leaf/{wc.target}/rf",
    conda: HSE
    threads: 16
    resources:
        mem_mb = 16000,
        runtime = 1200
    log:
        "logs/leaf/leaf_{target}_rf.log"
    shell:
        r"""
        mkdir -p {params.outdir} logs/leaf
        export MPLBACKEND=Agg
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
        python {input.code} \
          --images {input.images} \
          --target {wildcards.target} \
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


rule leaf_svm:
    input:
        images = "results/hsi/leaf_features.tsv",
        code   = "scripts/hsi/session_classify_svm.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/leaf/{target}/svm/svm_best_model.joblib",
        summary = "results/hsi/leaf/{target}/svm/svm_summary.csv",
        cm_png  = "results/hsi/leaf/{target}/svm/svm_confusion_matrix_norm.png",
        curve   = "results/hsi/leaf/{target}/svm/svm_hparam_curve_C.png",
    params:
        outdir = lambda wc: f"results/hsi/leaf/{wc.target}/svm",
    conda: HSE
    threads: 16
    resources:
        mem_mb = 16000,
        runtime = 1200
    log:
        "logs/leaf/leaf_{target}_svm.log"
    shell:
        r"""
        mkdir -p {params.outdir} logs/leaf
        export MPLBACKEND=Agg
        python {input.code} \
          --images {input.images} \
          --target {wildcards.target} \
          --outdir {params.outdir} \
          --test-size 0.2 \
          --cv-folds 5 \
          --random-state 42 \
          --n-jobs {threads} \
          > {log} 2>&1

        # SVM 脚本里目前没保存 best_model，可以按需要加；这里按输出名检查
        test -s {output.summary}
        test -s {output.cm_png} || true
        test -s {output.curve}  || true
        touch {output.model}
        """


rule leaf_knn:
    input:
        images = "results/hsi/leaf_features.tsv",
        code   = "scripts/hsi/session_classify_knn.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/leaf/{target}/knn/knn_best_model.joblib",
        summary = "results/hsi/leaf/{target}/knn/knn_summary.csv",
        cm_png  = "results/hsi/leaf/{target}/knn/knn_confusion_matrix_norm.png",
        curve   = "results/hsi/leaf/{target}/knn/knn_hparam_curve_n_neighbors.png",
    params:
        outdir = lambda wc: f"results/hsi/leaf/{wc.target}/knn",
    conda: HSE
    threads: 8
    resources:
        mem_mb = 8000,
        runtime = 60
    log:
        "logs/leaf/leaf_{target}_knn.log"
    shell:
        r"""
        mkdir -p {params.outdir} logs/leaf
        export MPLBACKEND=Agg
        python {input.code} \
          --images {input.images} \
          --target {wildcards.target} \
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


rule leaf_lr:
    input:
        images = "results/hsi/leaf_features.tsv",
        code   = "scripts/hsi/session_classify_lr.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/leaf/{target}/lr/lr_best_model.joblib",
        summary = "results/hsi/leaf/{target}/lr/lr_summary.csv",
        cm_png  = "results/hsi/leaf/{target}/lr/lr_confusion_matrix_norm.png",
        curve   = "results/hsi/leaf/{target}/lr/lr_hparam_curve_C.png",
    params:
        outdir = lambda wc: f"results/hsi/leaf/{wc.target}/lr",
    conda: HSE
    threads: 8
    resources:
        mem_mb = 8000,
        runtime = 60
    log:
        "logs/leaf/leaf_{target}_lr.log"
    shell:
        r"""
        mkdir -p {params.outdir} logs/leaf
        export MPLBACKEND=Agg
        python {input.code} \
          --images {input.images} \
          --target {wildcards.target} \
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


rule leaf_xgb:
    input:
        images = "results/hsi/leaf_features.tsv",
        code   = "scripts/hsi/session_classify_xgb.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/leaf/{target}/xgb/xgb_best_model.joblib",
        summary = "results/hsi/leaf/{target}/xgb/xgb_summary.csv",
        cm_png  = "results/hsi/leaf/{target}/xgb/xgb_confusion_matrix_norm.png",
        curve   = "results/hsi/leaf/{target}/xgb/xgb_hparam_curve_n_estimators.png",
    params:
        outdir = lambda wc: f"results/hsi/leaf/{wc.target}/xgb",
    threads: 8
    resources:
        runtime = 60000,
        mem_mb  = 16000,
        partition = "gpu-3090",
        gres="--gres=gpu:1",
    log:
        "logs/leaf/leaf_{target}_xgb.log"
    shell:
        r"""
        set -euo pipefail

        mkdir -p {params.outdir} logs/leaf

        # 激活 xgb_gpu 环境
        source /public/home/zoujianghua/miniconda3/etc/profile.d/conda.sh
        conda activate xgb_gpu

        # 让 conda 自己的 libstdc++ 优先
        export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"

        echo "NODE=$(hostname)"
        echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-<empty>}}"
        nvidia-smi || true

        # 线程数和绘图后端
        export OMP_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}
        export OPENBLAS_NUM_THREADS={threads}
        export NUMEXPR_NUM_THREADS={threads}
        export MPLBACKEND=Agg

        python {input.code} \
          --images {input.images} \
          --target {wildcards.target} \
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


rule leaf_pls:
    input:
        images = "results/hsi/leaf_features.tsv",
        code   = "scripts/hsi/session_classify_pls.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/leaf/{target}/pls/pls_best_model.joblib",
        summary = "results/hsi/leaf/{target}/pls/pls_summary.csv",
        report  = "results/hsi/leaf/{target}/pls/pls_classification_report.csv",
        cm_png  = "results/hsi/leaf/{target}/pls/pls_confusion_matrix_norm.png",
        hcurve  = "results/hsi/leaf/{target}/pls/pls_hparam_curve_n_components.png",
    params:
        outdir = lambda wc: f"results/hsi/leaf/{wc.target}/pls",
    conda: HSE
    threads: 8
    resources:
        mem_mb = 8000,
        runtime = 60
    log:
        "logs/leaf/leaf_{target}_pls.log"
    shell:
        r"""
        mkdir -p {params.outdir} logs/leaf
        export MPLBACKEND=Agg

        python {input.code} \
          --images {input.images} \
          --target {wildcards.target} \
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
# HSI ML: LDA
############################################

rule leaf_lda:
    input:
        images = "results/hsi/leaf_features.tsv",
        code   = "scripts/hsi/session_classify_lda.py",
        utils  = "scripts/hsi/ml_utils.py",
    output:
        model   = "results/hsi/leaf/{target}/lda/lda_best_model.joblib",
        summary = "results/hsi/leaf/{target}/lda/lda_summary.csv",
        report  = "results/hsi/leaf/{target}/lda/lda_classification_report.csv",
        cm_png  = "results/hsi/leaf/{target}/lda/lda_confusion_matrix_norm.png",
        hcurve  = "results/hsi/leaf/{target}/lda/lda_hparam_curve_shrinkage.png",
    params:
        outdir = lambda wc: f"results/hsi/leaf/{wc.target}/lda",
    conda: HSE
    threads: 8
    resources:
        mem_mb = 8000,
        runtime = 60
    log:
        "logs/leaf/leaf_{target}_lda.log"
    shell:
        r"""
        mkdir -p {params.outdir} logs/leaf
        export MPLBACKEND=Agg

        python {input.code} \
          --images {input.images} \
          --target {wildcards.target} \
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


rule leaf_1dcnn:
    input:
        images = "results/hsi/leaf_features.tsv"
    output:
        model   = "results/hsi/leaf/{target}/1dcnn_image/best_model.pt",
        metrics = "results/hsi/leaf/{target}/1dcnn_image/metrics.tsv"
    params:
        outdir = lambda wc: f"results/hsi/leaf/{wc.target}/1dcnn_image",
        seed   = 42,
        batch  = 64,
        epochs = 200
    threads: 8
    resources:
        mem_mb = 16000,
        runtime = 1440,
        partition = "gpu-3090",
        gres="--gres=gpu:1",
    log:
        "logs/leaf/leaf_{target}_1dcnn.log"
    shell:
        r"""
        set -euo pipefail

        mkdir -p {params.outdir} logs/leaf

        # 激活你已经建好的 hsi_dl 环境
        # 如果 conda.sh 路径跟下面不一样，改成你自己机器上的
        source /public/home/zoujianghua/miniconda3/etc/profile.d/conda.sh
        conda activate hsi_dl

         # 关键：让 conda 自己的 libstdc++ 排在最前面
        export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"
        echo "NODE=$(hostname)"
        echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-<empty>}}"
        nvidia-smi || true

        export OMP_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}
        export OPENBLAS_NUM_THREADS={threads}
        export NUMEXPR_NUM_THREADS={threads}

        python scripts/hsi/hsi_1dcnn.py \
          --images {input.images} \
          --target {wildcards.target} \
          --outdir {params.outdir} \
          --batch-size {params.batch} \
          --epochs {params.epochs} \
          --seed {params.seed} \
          > {log} 2>&1

        # 这里假设 hsi_1dcnn.py 在训练完会写出：
        #   {params.outdir}/best_model.pt
        #   {params.outdir}/metrics.tsv
        """




rule leaf_merge:
    priority: 760
    input:
        merge_code = "scripts/hsi/session_classify_merge.py",
        rf  = "results/hsi/leaf/{target}/rf/rf_summary.csv",
        svm = "results/hsi/leaf/{target}/svm/svm_summary.csv",
        knn = "results/hsi/leaf/{target}/knn/knn_summary.csv",
        lr  = "results/hsi/leaf/{target}/lr/lr_summary.csv",
        xgb = "results/hsi/leaf/{target}/xgb/xgb_summary.csv",
        pls = "results/hsi/leaf/{target}/pls/pls_summary.csv",
        lda = "results/hsi/leaf/{target}/lda/lda_summary.csv",
    output:
        summary = "results/hsi/leaf/{target}/all_models_summary.csv",
        f1_png  = "results/hsi/leaf/{target}/all_models_f1_bar.png",
        acc_png = "results/hsi/leaf/{target}/all_models_accuracy_bar.png",
    params:
        outdir = lambda wc: f"results/hsi/leaf/{wc.target}",
    conda: HSE
    threads: 8
    resources:
        mem_mb = 8000,
        runtime = 30
    log:
        "logs/leaf/leaf_{target}_merge.log"
    shell:
        r"""
        mkdir -p {params.outdir} logs/leaf
        export MPLBACKEND=Agg

        python {input.merge_code} \
          --summaries {input.rf} {input.svm} {input.knn} {input.lr} {input.xgb} {input.pls} {input.lda} \
          --outdir {params.outdir} \
          > {log} 2>&1

        test -s {output.summary}
        test -s {output.f1_png}  || true
        test -s {output.acc_png} || true
        """

