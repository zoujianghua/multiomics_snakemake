# rules/hsi.smk
SAMPLES_HSI = "config/samples_hsi.csv"
HSE = "../envs/hsi_env.yaml"

rule hsi_preprocess:
    input:
        SAMPLES_HSI,
        code   = "scripts/hsi/preprocess.py",
    output:
        raw = "results/hsi/raw_image_features.tsv"
    conda: HSE
    threads: 16
    resources: mem_mb=32000, runtime=600
    log: "logs/hsi_preprocess.log"
    shell:
        r"""
        mkdir -p results/hsi logs
        export MPLBACKEND=Agg
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

        python scripts/hsi/preprocess.py \
          --samples {input[0]} \
          --outdir results/hsi \
          --sg-window 9 --sg-poly 2 \
          --min-area 2000 --min-ndvi 0.10 --min-ndre 0.02 --aspect-ratio-max 10 \
          --leaf-ndvi-quantile 0.25 --leaf-refine-min-area 1000 \
          --leaf-ndre-quantile 0.1 --roi-min-area-frac 0.3 \
          --fixed-white-hdr /public/agis/wanfanghao_group/zoujianghua/ggpdata/CK25_T1d/CK25_T1d_1_2025-04-23_06-21-58/capture/WHITEREF_CK25_T1d_1_2025-04-23_06-21-58.hdr \
          --fixed-dark-hdr  /public/agis/wanfanghao_group/zoujianghua/ggpdata/CK25_T1d/CK25_T1d_1_2025-04-23_06-21-58/capture/DARKREF_CK25_T1d_1_2025-04-23_06-21-58.hdr \
          --ref-center-frac 0.6 \
          --workers {threads} \
          >> {log} 2>&1
        """


rule hsi_clean_and_indices:
    input:
        raw = "results/hsi/raw_image_features.tsv",
        code1 = "scripts/hsi/clean_image_features.py",
        code2 = "scripts/hsi/add_indices.py",
    output:
        clean = "results/hsi/clean_image_features.tsv",
        final = "results/hsi/image_features.tsv"
    conda: HSE
    threads: 8
    resources:
        mem_mb = 8000,
        runtime = 30
    log:
        "logs/hsi_clean_and_indices.log"
    shell:
        r"""
        mkdir -p results/hsi logs

        # 1) 清洗异常样本
        python scripts/hsi/clean_image_features.py \
          --images-raw {input.raw} \
          --out {output.clean} \
          --min-r800 0.04 \
          --max-z 3.0

        # 2) 计算全部植被指数
        python scripts/hsi/add_indices.py \
          --images {output.clean} \
          --out {output.final} \
          >> {log} 2>&1
        """


# ======= HSI: aggregate （固定使用 mean 聚合） =======

rule hsi_aggregate:
    input:
        images ="results/hsi/image_features.tsv",
        code   = "scripts/hsi/aggregate.py",

    output:
        sess    = "results/hsi/session_features.tsv",
        delt    = "results/hsi/delta_hsi.tsv",
        dyn     = "results/hsi/resilience_metrics.tsv",

    conda: HSE
    threads: 8
    resources:
        mem_mb = 8000,
        runtime = 60
    log:
        "logs/hsi_aggregate.log"
    shell:
        r"""
        mkdir -p results/hsi logs
        export MPLBACKEND=Agg

        python scripts/hsi/aggregate.py \
          --image-features {input[0]} \
          --outdir results/hsi \
          --control-temp 25 \
          > {log} 2>&1

        # 存在性检查
        test -s {output.sess}
        test -s {output.delt}
        test -s {output.dyn}
        """



# ======= HSI: quicklook （固定使用 mean±SE 绘图） =======

rule hsi_quicklook:
    input:
        images = "results/hsi/image_features.tsv",
        sess = "results/hsi/session_features.tsv",
        delt = "results/hsi/delta_hsi.tsv",
        resil  = "results/hsi/resilience_metrics.tsv"

    output:
        ndvi_png  = "results/hsi/plot_ndvi_timeseries.png",
        drep_png  = "results/hsi/plot_drep_timeseries.png",
        dndvi_png = "results/hsi/plot_dndvi_heatmap.png"
    conda: HSE
    threads: 8
    resources:
        mem_mb = 8000,
        runtime = 60
    log:
        "logs/hsi_quicklook.log"
    shell:
        r"""
        mkdir -p results/hsi "$(dirname {log})"
        export MPLBACKEND=Agg
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

        python scripts/hsi/quicklook.py \
          --images    {input.images} \
          --features  {input.sess} \
          --delta     {input.delt} \
          --resilience {input.resil} \
          --outdir    results/hsi \
          --specdir   results/hsi/session_spectra \
          --delta-mode vs25 \
          > {log} 2>&1

        test -s {output.ndvi_png}
        test -s {output.drep_png}  || true
        test -s {output.dndvi_png} || true
        """




rule hsi_preprocess_leaf:
    input:
        samples = SAMPLES_HSI,
        cube_dir = "results/hsi/cube",
        code = "scripts/hsi/preprocess_leaf.py",
    output:
        raw_leaf = "results/hsi/raw_leaf_features.tsv"
    conda: HSE
    threads: 16
    resources: mem_mb=24000, runtime=300
    log: "logs/hsi_preprocess_leaf.log"
    shell:
        r"""
        mkdir -p results/hsi_leaf logs
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

        python scripts/hsi/preprocess_leaf.py \
          --samples {input.samples} \
          --cube-dir results/hsi/cube \
          --outdir results/hsi \
          --sg-window 9 --sg-poly 2 \
          --leaf-min-area 800 \
          --leaf-aspect-ratio-max 8.0 \
          --leaf-ndvi-mean-min 0.2 \
          --workers {threads} \
          > {log} 2>&1
        """

rule hsi_leaf_clean_and_indices:
    input:
        raw = "results/hsi/raw_leaf_features.tsv",
        code1 = "scripts/hsi/clean_image_features.py",
        code2 = "scripts/hsi/add_indices.py",
    output:
        clean = "results/hsi/clean_leaf_features.tsv",
        final = "results/hsi/leaf_features.tsv"
    conda: HSE
    threads: 8
    resources:
        mem_mb = 8000,
        runtime = 30
    log:
        "logs/hsi_leaf_clean_and_indices.log"
    shell:
        r"""
        mkdir -p results/hsi logs

        # 1) 清洗异常样本
        python scripts/hsi/clean_image_features.py \
          --images-raw {input.raw} \
          --out {output.clean} \
          --min-r800 0.04 \
          --max-z 3.0

        # 2) 计算全部植被指数
        python scripts/hsi/add_indices.py \
          --images {output.clean} \
          --out {output.final} \
          >> {log} 2>&1
        """


