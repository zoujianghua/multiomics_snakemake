# Image/Leaf 多标签：与 config image_targets 一致
IMAGE_TARGETS = config.get("hsi", {}).get("image_targets", [config.get("hsi", {}).get("patch_target", "phase_core")])
LEAF_TARGETS  = config.get("hsi", {}).get("image_targets", [config.get("hsi", {}).get("patch_target", "phase_core")])

# 汇总全 HSI 结果为一张表（level × target × model），便于对比
rule hsi_all_models_index:
    input:
        code    = "scripts/hsi/collect_hsi_all_models_index.py",
        ml      = expand("results/hsi/ml/{target}/all_models_summary.csv", target=IMAGE_TARGETS),
        leaf    = expand("results/hsi/leaf/{target}/all_models_summary.csv", target=LEAF_TARGETS),
        patch   = expand("results/hsi/patch/{run_id}/all_models_summary.csv", run_id=RUN_IDS)
                  + expand("results/hsi/patch/{physio_run_id}/all_models_summary.csv", physio_run_id=PHYSIO_RUN_IDS),
    output:
        index = "results/hsi/hsi_all_models_index.tsv",
    conda: HSE
    resources:
        mem_mb  = 8000,
        runtime = 60,
    log:
        "logs/hsi/hsi_all_models_index.log",
    shell:
        r"""
        mkdir -p results/hsi logs/hsi
        python {input.code} --out {output.index} --results-dir results/hsi > {log} 2>&1
        test -s {output.index}
        """


rule all_hsi:
    input:
        rules.hsi_aggregate.output.sess,
        rules.hsi_aggregate.output.delt,
        rules.hsi_aggregate.output.dyn,
        rules.hsi_quicklook.output.ndvi_png,
        rules.hsi_quicklook.output.drep_png,
        rules.hsi_quicklook.output.dndvi_png,
        rules.hsi_preprocess_leaf.output,
        rules.hsi_leaf_clean_and_indices.output,
        # Image 多标签：每个 target 的 1dcnn + merge
        expand("results/hsi/dl/1dcnn_image_{target}/metrics.tsv", target=IMAGE_TARGETS),
        expand("results/hsi/ml/{target}/all_models_summary.csv", target=IMAGE_TARGETS),
        # Leaf 多标签：每个 target 的 1dcnn + merge
        expand("results/hsi/leaf/{target}/1dcnn_image/metrics.tsv", target=LEAF_TARGETS),
        expand("results/hsi/leaf/{target}/all_models_summary.csv", target=LEAF_TARGETS),
        # Patch 多标签 × 多参数：每个 run_id 的 merge 与 1d/2d/3d CNN
        expand("results/hsi/patch/{run_id}/all_models_summary.csv", run_id=RUN_IDS),
        expand("results/hsi/patch/1dcnn_image_{run_id}/metrics.tsv", run_id=RUN_IDS),
        expand("results/hsi/patch/2dcnn_{run_id}/metrics.tsv", run_id=RUN_IDS),
        expand("results/hsi/patch/3dcnn_{run_id}/metrics.tsv", run_id=RUN_IDS),
        # 生理状态分类结果（多 param_suffix）
        expand("results/hsi/patch/{physio_run_id}/all_models_summary.csv", physio_run_id=PHYSIO_RUN_IDS),
        # 全 HSI 结果一张表（level × target × model）
        rules.hsi_all_models_index.output.index,





