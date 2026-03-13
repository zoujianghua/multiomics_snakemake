# rules/ml_pipeline.smk
# -------------------------------------------------------------------------
# HSI 机器学习与轻量 DL 独立流水线 (Image & Leaf 级)
# -------------------------------------------------------------------------

HSE = "../envs/hsi_env.yaml"

LEVELS  = ["image", "leaf"]
TARGETS = ["phase", "physiological_state"]
ML_MODELS = ["rf", "svm", "knn", "lr", "xgb", "pls", "lda"]

rule ml_inject_labels:
    input:
        img_raw = "results/hsi/image_features.tsv",
        leaf_raw = "results/hsi/leaf_features.tsv",
        mapping = "config/phase_physiological_state_mapping.tsv",
        code = "scripts/hsi/inject_physiological_labels.py",
    output:
        img_out = "results/hsi/ml_pipeline/data/image_features_labeled.tsv",
        leaf_out = "results/hsi/ml_pipeline/data/leaf_features_labeled.tsv",
    conda: HSE
    threads: 4
    resources: mem_mb = 4000, runtime = 30
    log: "logs/ml_pipeline/inject_labels.log"
    shell:
        r"""
        mkdir -p results/hsi/ml_pipeline/data logs/ml_pipeline
        python {input.code} \
          --image-in {input.img_raw} \
          --leaf-in {input.leaf_raw} \
          --mapping {input.mapping} \
          --image-out {output.img_out} \
          --leaf-out {output.leaf_out} \
          > {log} 2>&1
        """

rule ml_safe_split:
    input:
        features = lambda wc: f"results/hsi/ml_pipeline/data/{wc.level}_features_labeled.tsv",
        code = "scripts/hsi/make_split_safe.py",
    output:
        split_tsv = "results/hsi/ml_pipeline/splits/split_{level}_{target}_seed42.tsv"
    conda: HSE
    threads: 4
    resources: mem_mb = 4000, runtime = 30
    log: "logs/ml_pipeline/split_{level}_{target}.log"
    shell:
        r"""
        mkdir -p results/hsi/ml_pipeline/splits
        python {input.code} \
          --features {input.features} \
          --target-col {wildcards.target} \
          --level {wildcards.level} \
          --seed 42 \
          --test-size 0.2 \
          --out {output.split_tsv} \
          > {log} 2>&1
        """

# =========================================================================
# 3A. 传统 CPU 机器学习模型库 (RF, SVM, KNN, LR, PLS, LDA)
# =========================================================================

CPU_MODELS = ["rf", "svm", "knn", "lr", "pls", "lda"]

rule ml_train_cpu:
    """
    运行无需 GPU 的传统 ML 模型。
    不指定 partition，让 SLURM 自动投递到默认的 CPU 队列。
    """
    input:
        features = lambda wc: f"results/hsi/ml_pipeline/data/{wc.level}_features_labeled.tsv",
        split = "results/hsi/ml_pipeline/splits/split_{level}_{target}_seed42.tsv",
        code = lambda wc: f"scripts/hsi/session_classify_{wc.model}.py",
        utils = "scripts/hsi/ml_utils.py",
    output:
        summary = "results/hsi/ml_pipeline/{level}/{target}/{model}/{model}_summary.csv",
        model = "results/hsi/ml_pipeline/{level}/{target}/{model}/{model}_best_model.joblib",
    # 使用通配符约束，确保这个 rule 只接管 CPU 模型
    wildcard_constraints:
        model = "|".join(CPU_MODELS)
    params:
        outdir = lambda wc: f"results/hsi/ml_pipeline/{wc.level}/{wc.target}/{wc.model}",
    conda: HSE
    threads: 16
    resources:
        mem_mb = 16000,
        runtime = 120
        # 移除了 partition，使用系统默认 CPU 队列
    log:
        "logs/ml_pipeline/train_{level}_{target}_{model}.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p {params.outdir} logs/ml_pipeline

        export MPLBACKEND=Agg
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

        python {input.code} \
          --images {input.features} \
          --target {wildcards.target} \
          --outdir {params.outdir} \
          --test-size 0.2 \
          --cv-folds 5 \
          --random-state 42 \
          --n-jobs {threads} \
          --split-path {input.split} \
          > {log} 2>&1

        test -s {output.summary}
        """

# =========================================================================
# 3B. 专属 XGBoost GPU 训练管线
# =========================================================================

rule ml_train_xgb:
    """
    专门为 XGBoost 开辟的 GPU 加速通道。
    环境、资源请求、启动参数完全物理隔离，避免调度冲突。
    """
    input:
        features = lambda wc: f"results/hsi/ml_pipeline/data/{wc.level}_features_labeled.tsv",
        split = "results/hsi/ml_pipeline/splits/split_{level}_{target}_seed42.tsv",
        code = "scripts/hsi/session_classify_xgb.py",
        utils = "scripts/hsi/ml_utils.py",
    output:
        summary = "results/hsi/ml_pipeline/{level}/{target}/xgb/xgb_summary.csv",
        model = "results/hsi/ml_pipeline/{level}/{target}/xgb/xgb_best_model.joblib",
    params:
        outdir = lambda wc: f"results/hsi/ml_pipeline/{wc.level}/{wc.target}/xgb",
    threads: 16
    resources:
        mem_mb = 16000,
        runtime = 600,
        partition = "gpu-3090",
        gres = "--gres=gpu:1"
    log:
        "logs/ml_pipeline/train_{level}_{target}_xgb.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p {params.outdir} logs/ml_pipeline

        # 独立激活 GPU 环境
        source /public/home/zoujianghua/miniconda3/etc/profile.d/conda.sh
        conda activate xgb_gpu
        export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"

        export MPLBACKEND=Agg
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

        python {input.code} \
          --images {input.features} \
          --target {wildcards.target} \
          --outdir {params.outdir} \
          --test-size 0.2 \
          --cv-folds 5 \
          --random-state 42 \
          --n-jobs {threads} \
          --split-path {input.split} \
          --xgb-gpu \
          > {log} 2>&1

        test -s {output.summary}
        """

rule ml_train_1dcnn:
    input:
        features = lambda wc: f"results/hsi/ml_pipeline/data/{wc.level}_features_labeled.tsv",
        split = "results/hsi/ml_pipeline/splits/split_{level}_{target}_seed42.tsv",
        code = "scripts/hsi/hsi_1dcnn.py",
        utils = "scripts/hsi/ml_utils.py",
    output:
        model = "results/hsi/ml_pipeline/{level}/{target}/1dcnn/best_model.pt",
        metrics = "results/hsi/ml_pipeline/{level}/{target}/1dcnn/metrics.tsv"
    params:
        outdir = lambda wc: f"results/hsi/ml_pipeline/{wc.level}/{wc.target}/1dcnn",
    threads: 16
    resources: mem_mb = 16000, runtime = 1600, partition = "gpu-3090", gres = "--gres=gpu:1"
    log: "logs/ml_pipeline/train_{level}_{target}_1dcnn.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p {params.outdir} logs/ml_pipeline

        source /public/home/zoujianghua/miniconda3/etc/profile.d/conda.sh
        conda activate hsi_dl
        export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"

        export MPLBACKEND=Agg
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

        python {input.code} \
          --images {input.features} \
          --target {wildcards.target} \
          --outdir {params.outdir} \
          --split-path {input.split} \
          --batch-size 64 \
          --epochs 200 \
          --seed 42 \
          > {log} 2>&1

        test -s {output.metrics}
        """

rule ml_merge_results:
    priority: 800
    input:
        merge_code = "scripts/hsi/session_classify_merge.py",
        ml_summaries = lambda wc: expand(
            "results/hsi/ml_pipeline/{level}/{target}/{model}/{model}_summary.csv",
            level=[wc.level], target=[wc.target], model=ML_MODELS
        )
    output:
        summary = "results/hsi/ml_pipeline/{level}/{target}/all_models_summary.csv",
        f1_png  = "results/hsi/ml_pipeline/{level}/{target}/all_models_f1_bar.png"
    params:
        outdir = lambda wc: f"results/hsi/ml_pipeline/{wc.level}/{wc.target}"
    conda: HSE
    threads: 4
    resources: mem_mb = 4000, runtime = 30
    log: "logs/ml_pipeline/merge_{level}_{target}.log"
    shell:
        r"""
        mkdir -p {params.outdir} logs/ml_pipeline
        export MPLBACKEND=Agg

        python {input.merge_code} \
          --summaries {input.ml_summaries} \
          --outdir {params.outdir} \
          > {log} 2>&1 || true

        test -s {output.summary} || true
        """

rule all_ml_pipeline:
    input:
        expand("results/hsi/ml_pipeline/{level}/{target}/all_models_summary.csv", level=LEVELS, target=TARGETS),
        expand("results/hsi/ml_pipeline/{level}/{target}/1dcnn/metrics.tsv", level=LEVELS, target=TARGETS)
