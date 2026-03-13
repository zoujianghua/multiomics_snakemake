configfile: "config/config.yaml"

include: "rules/references.smk"
include: "rules/hsi.smk"          # 先定义 HSI 基础流程
include: "rules/hsi_ml.smk"
include: "rules/leaf.smk"
include: "rules/patch.smk"
include: "rules/physiological.smk"  # 生理状态分类（依赖 patch.smk）
include: "rules/hsi_all.smk"
include: "rules/rnaseq/rnaseq_all.smk"
#include: "rules/wgs/wgs_all.smk"
# include: "rules/metabo/metabo_all.smk"  # 已移除代谢组模块（未使用）




rule all:
    input:
        rules.all_references.input,
        rules.all_hsi.input,
        rules.all_rnaseq.input,
        # rules.all_wgs.input

