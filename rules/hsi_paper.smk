HSE = "../envs/hsi_env.yaml"


############################################
# 1. HSI embedding 聚合为 sample 级
############################################

rule hsi_embedding_by_sample:
    """
    从 2D/3D patch embedding 表 + image_features.tsv
    生成 sample 级 HSI embedding 表 hsi_embedding_by_sample.tsv
    """
    input:
        image_meta = "results/hsi/image_features.tsv",
        emb2d = "results/hsi/patch/2dcnn_phase/patch_embeddings_2d.tsv",
        emb3d = "results/hsi/patch/3dcnn_phase/patch_embeddings_3d.tsv",
        code = "scripts/hsi/make_hsi_embeddings_by_sample.py",
    output:
        tsv = "results/hsi/hsi_embedding_by_sample.tsv",
    conda: HSE
    threads: 8
    resources:
        mem_mb = 8000,
        runtime = 600
    log:
        "logs/hsi_paper/hsi_embedding_by_sample.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p $(dirname {output.tsv}) logs/hsi_paper

        python {input.code} \
          --emb-2d {input.emb2d} \
          --emb-3d {input.emb3d} \
          --image-meta {input.image_meta} \
          --out {output.tsv} \
          > {log} 2>&1

        test -s {output.tsv}
        """


############################################
# 2. 论文素材包导出（只做收集，不重算）
############################################

rule hsi_paper_materials:
    """
    收集 HSI 模块论文相关的关键表格，集中到
    results/hsi/paper_materials/tables 下，便于拷贝到本地作图。

    注意：此规则只做文件复制，不触发任何上游重算。
    """
    input:
        image_features = "results/hsi/image_features.tsv",
        session_features = "results/hsi/session_features.tsv",
        delta_hsi = "results/hsi/delta_hsi.tsv",
        resilience = "results/hsi/resilience_metrics.tsv",
        patch_summary_phase = "results/hsi/patch/phase/all_models_summary.csv",
        #patch_summary_phase_core = "results/hsi/patch/phase_core/all_models_summary.csv",
        physio_summary = "results/hsi/patch/physiological_state/all_models_summary.csv",
        hsi_embedding = "results/hsi/hsi_embedding_by_sample.tsv",
    output:
        tables_dir = directory("results/hsi/paper_materials/tables"),
    conda: HSE
    threads: 8
    resources:
        mem_mb = 8000,
        runtime = 120
    log:
        "logs/hsi_paper/hsi_paper_materials.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p {output.tables_dir} logs/hsi_paper

        cp {input.image_features} {output.tables_dir}/image_features.tsv
        cp {input.session_features} {output.tables_dir}/session_features.tsv
        cp {input.delta_hsi} {output.tables_dir}/delta_hsi.tsv
        cp {input.resilience} {output.tables_dir}/resilience_metrics.tsv

        cp {input.patch_summary_phase} {output.tables_dir}/patch_all_models_phase.csv || true
        #cp {input.patch_summary_phase_core} {output.tables_dir}/patch_all_models_phase_core.csv || true
        cp {input.physio_summary} {output.tables_dir}/patch_all_models_physiological_state.csv || true

        cp {input.hsi_embedding} {output.tables_dir}/hsi_embedding_by_sample.tsv || true

        """

