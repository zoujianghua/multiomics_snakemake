#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总 RNA-seq 上游 QC 结果：

结合 MultiQC 的 general stats 与 featureCounts 的 summary，输出
每个样本一行的 QC 汇总表，便于论文正文/附录直接引用。

输入：
  --multiqc-data           MultiQC 输出目录（通常为 results/rnaseq/qc/multiqc/multiqc_data）
  --featurecounts-summary  featureCounts summary 文件（featurecounts_hisat2.tsv.summary）

输出：
  --output                 TSV，包含 sample_id、MultiQC general stats 原始列、
                           以及 featureCounts 的 Assigned/total/assigned_frac 等列。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _general_stats_from_json(data_dir: Path) -> pd.DataFrame:
    """
    从 MultiQC v1.26+ 的 multiqc_data.json 中解析 report_general_stats_data，
    得到每样本一行的 DataFrame（第一列 sample_id）。
    """
    jpath = data_dir / "multiqc_data.json"
    if not jpath.is_file():
        return None
    with open(jpath) as f:
        data = json.load(f)
    g = data.get("report_general_stats_data")
    if not g or not isinstance(g, list):
        return None
    # g: list of dicts. Each dict: sample_id -> {metric: value} or single value.
    all_samples = set()
    for row in g:
        all_samples.update(k for k in row.keys() if not k.startswith("config_"))
    records = []
    for s in sorted(all_samples):
        r = {"sample_id": s}
        for row in g:
            v = row.get(s)
            if v is None:
                continue
            if isinstance(v, dict):
                r.update(v)
            else:
                r[f"metric_{len(r)}"] = v
        records.append(r)
    if not records:
        return None
    return pd.DataFrame.from_records(records)


def load_multiqc_general_stats(multiqc_data_dir: Path) -> pd.DataFrame:
    """
    读取 MultiQC general stats 表。
    支持 MultiQC 旧版目录 multiqc_data 与新版 v1.26+ 的 multiqc_report_data。
    尝试 multiqc_general_stats.txt/.tsv，或目录下任意含 general 的 .txt/.tsv。
    """
    base = Path(multiqc_data_dir)
    # 若传入路径不存在，可能是 MultiQC v1.26+ 写入了 multiqc_report_data 而非 multiqc_data
    if base.is_dir():
        data_dir = base
    else:
        parent = base.parent
        data_dir = None
        for sub in ("multiqc_report_data", "multiqc_data"):
            d = parent / sub
            if d.is_dir():
                data_dir = d
                break
        if data_dir is None:
            data_dir = base  # 让下面的 raise 触发
    multiqc_data_dir = data_dir
    if not multiqc_data_dir.is_dir():
        raise FileNotFoundError(
            f"MultiQC 数据目录不存在: {multiqc_data_dir}\n"
            "说明：转录组与重测序流程已解耦，RNA-seq 的 MultiQC 不依赖任何 WGS 输出。\n"
            "请先确保 RNA-seq 的 MultiQC 已成功运行：\n"
            "  snakemake --profile profiles/slurm results/rnaseq/qc/multiqc/multiqc_report.html\n"
            "MultiQC v1.26+ 会生成 multiqc_report_data，旧版为 multiqc_data。若已运行仍报错，请查看 logs/rnaseq/multiqc/multiqc.log。"
        )
    candidates = [
        multiqc_data_dir / "multiqc_general_stats.txt",
        multiqc_data_dir / "multiqc_general_stats.tsv",
    ]
    for f in multiqc_data_dir.iterdir():
        if f.suffix in (".txt", ".tsv") and "general" in f.name.lower():
            candidates.append(f)
    for p in candidates:
        if p.is_file():
            try:
                df = pd.read_csv(p, sep="\t")
            except Exception:
                continue
            if df.shape[1] == 0:
                continue
            df = df.rename(columns={df.columns[0]: "sample_id"})
            return df
    # MultiQC v1.26+ 不再写 multiqc_general_stats.txt，数据在 multiqc_data.json 的 report_general_stats_data
    df = _general_stats_from_json(multiqc_data_dir)
    if df is not None and not df.empty:
        return df
    raise FileNotFoundError(
        f"在 {multiqc_data_dir} 下未找到 general stats 文件（如 multiqc_general_stats.txt/.tsv）。\n"
        "说明：转录组与重测序流程已解耦，RNA-seq MultiQC 不依赖 WGS。\n"
        "请先成功运行 RNA-seq 的 MultiQC：\n"
        "  snakemake --profile profiles/slurm results/rnaseq/qc/multiqc/multiqc_report.html\n"
        "并检查 logs/rnaseq/multiqc/multiqc.log 无报错；若目录内无 .txt/.tsv，可能是 MultiQC 版本差异，请查看该目录下实际生成的文件名。"
    )


def load_featurecounts_summary(summary_path: Path) -> pd.DataFrame:
    """
    解析 featureCounts summary（Status 为行，样本为列）为每样本一行。
    """
    df = pd.read_csv(summary_path, sep="\t")
    if df.empty or df.shape[1] < 2:
        raise RuntimeError(f"featureCounts summary 格式异常: {summary_path}")

    status_col = df.columns[0]
    statuses = df[status_col].astype(str).to_numpy()
    sample_cols = df.columns[1:]

    records = []
    for sample in sample_cols:
        counts = df[sample].to_numpy(dtype=float)
        total = float(np.nansum(counts))
        rec = {"sample_id": str(sample), "fc_total": total}
        for s, c in zip(statuses, counts):
            rec[f"fc_{s}"] = float(c)
        # Assigned 比例（若存在该行）
        assigned_mask = statuses == "Assigned"
        if assigned_mask.any() and total > 0:
            assigned_val = float(np.nansum(counts[assigned_mask]))
            rec["fc_assigned_frac"] = assigned_val / total
        else:
            rec["fc_assigned_frac"] = np.nan
        records.append(rec)

    return pd.DataFrame.from_records(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--multiqc-data",
        required=True,
        help="MultiQC 输出目录（results/rnaseq/qc/multiqc/multiqc_data）",
    )
    ap.add_argument(
        "--featurecounts-summary",
        required=True,
        help="featureCounts summary 文件（featurecounts_hisat2.tsv.summary）",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="输出 TSV 路径（例如 results/rnaseq/qc/upstream_qc_summary.tsv）",
    )
    args = ap.parse_args()

    multiqc_data_dir = Path(args.multiqc_data)
    fc_summary_path = Path(args.featurecounts_summary)
    out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    mqc_df = load_multiqc_general_stats(multiqc_data_dir)
    fc_df = load_featurecounts_summary(fc_summary_path)
    merged = pd.merge(mqc_df, fc_df, on="sample_id", how="outer")
    merged = merged.sort_values("sample_id")
    merged.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main()

