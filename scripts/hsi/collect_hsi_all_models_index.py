#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总全 HSI 结果为一张表：level × target × model，便于对比与论文展示。

输出：results/hsi/hsi_all_models_index.tsv
列：level, target, model, val_accuracy, val_f1_weighted, val_f1_macro, val_auc,
    cv_best_score, best_params, best_epoch, path_summary, path_metrics
"""

import argparse
from pathlib import Path

import pandas as pd

# 输出表列顺序（缺失的列填空字符串）
OUTPUT_COLS = [
    "level", "target", "model",
    "val_accuracy", "val_f1_weighted", "val_f1_macro", "val_auc",
    "cv_best_score", "best_params", "best_epoch",
    "path_summary", "path_metrics",
]


def _norm_acc_f1(df):
    """统一列为 val_accuracy, val_f1_weighted。"""
    if "test_accuracy" in df.columns and "val_accuracy" not in df.columns:
        df = df.rename(columns={"test_accuracy": "val_accuracy"})
    if "test_f1_weighted" in df.columns and "val_f1_weighted" not in df.columns:
        df = df.rename(columns={"test_f1_weighted": "val_f1_weighted"})
    return df


def _row_base(level, target, model, path_summary="", path_metrics=""):
    """基础行字典，扩展列填空。"""
    return {
        "level": level,
        "target": target,
        "model": str(model),
        "val_accuracy": "",
        "val_f1_weighted": "",
        "val_f1_macro": "",
        "val_auc": "",
        "cv_best_score": "",
        "best_params": "",
        "best_epoch": "",
        "path_summary": path_summary,
        "path_metrics": path_metrics,
    }


def _pick_auc(r):
    """从一行中尝试读取 AUC（多种列名）。"""
    for key in ("test_roc_auc", "roc_auc_ovr", "val_auc", "test_auc", "roc_auc"):
        val = r.get(key)
        if val is not None and str(val).strip() != "":
            return val
    return ""


def _format_run_params_tsv(run_params_path):
    """将 run_params.tsv（param/value 两列）格式化为短字符串，便于表格展示。"""
    if not run_params_path or not Path(run_params_path).is_file():
        return ""
    try:
        df = pd.read_csv(run_params_path, sep="\t")
        if "param" in df.columns and "value" in df.columns:
            parts = [f"{row['param']}={row['value']}" for _, row in df.iterrows()]
            return "; ".join(parts[:12])  # 最多 12 个参数，避免过长
    except Exception:
        pass
    return ""


def main():
    ap = argparse.ArgumentParser(description="汇总 HSI 所有 level/target/model 结果为一张表")
    ap.add_argument("--out", default="results/hsi/hsi_all_models_index.tsv", help="输出 TSV 路径")
    ap.add_argument("--results-dir", default="results/hsi", help="results/hsi 根目录")
    args = ap.parse_args()

    results = Path(args.results_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # ----- Image: ml/{target}/all_models_summary.csv (7 ML，多标签并行) -----
    ml_dir = results / "ml"
    if ml_dir.is_dir():
        for target_dir in sorted(ml_dir.iterdir()):
            if not target_dir.is_dir():
                continue
            target = target_dir.name
            ml_summary = target_dir / "all_models_summary.csv"
            if ml_summary.is_file():
                df = pd.read_csv(ml_summary)
                df = _norm_acc_f1(df)
                for _, r in df.iterrows():
                    model = str(r.get("model", "unknown"))
                    row = _row_base("image", target, model, path_summary=str(ml_summary), path_metrics="")
                    row["val_accuracy"] = r.get("val_accuracy", r.get("test_accuracy"))
                    row["val_f1_weighted"] = r.get("val_f1_weighted", r.get("test_f1_weighted"))
                    row["val_f1_macro"] = r.get("val_f1_macro", r.get("test_f1_macro"))
                    row["val_auc"] = _pick_auc(r)
                    row["cv_best_score"] = r.get("cv_best_score")
                    row["best_params"] = r.get("best_params", "")
                    rows.append(row)

    # ----- Image: dl/1dcnn_image_{target}/metrics.tsv -----
    dl_dir = results / "dl"
    if dl_dir.is_dir():
        for sub in sorted(dl_dir.iterdir()):
            if not sub.is_dir() or not sub.name.startswith("1dcnn_image_"):
                continue
            target = sub.name.replace("1dcnn_image_", "")
            img_1d = sub / "metrics.tsv"
            if img_1d.is_file():
                df = pd.read_csv(img_1d, sep="\t")
                df = _norm_acc_f1(df)
                for _, r in df.iterrows():
                    row = _row_base("image", target, "1dcnn", path_metrics=str(img_1d))
                    row["val_accuracy"] = r.get("val_accuracy", r.get("best_test_accuracy"))
                    row["val_f1_weighted"] = r.get("val_f1_weighted", r.get("best_test_f1_weighted"))
                    row["val_f1_macro"] = r.get("val_f1_macro", r.get("best_test_f1_macro"))
                    row["val_auc"] = _pick_auc(r)
                    row["best_epoch"] = r.get("best_epoch", "")
                    row["best_params"] = _format_run_params_tsv(sub / "run_params.tsv")
                    rows.append(row)

    # ----- Leaf: leaf/{target}/all_models_summary.csv (7 ML，多标签并行) -----
    leaf_dir = results / "leaf"
    if leaf_dir.is_dir():
        for target_dir in sorted(leaf_dir.iterdir()):
            if not target_dir.is_dir():
                continue
            target = target_dir.name
            leaf_summary = target_dir / "all_models_summary.csv"
            if leaf_summary.is_file():
                df = pd.read_csv(leaf_summary)
                df = _norm_acc_f1(df)
                for _, r in df.iterrows():
                    model = str(r.get("model", "unknown"))
                    row = _row_base("leaf", target, model, path_summary=str(leaf_summary), path_metrics="")
                    row["val_accuracy"] = r.get("val_accuracy", r.get("test_accuracy"))
                    row["val_f1_weighted"] = r.get("val_f1_weighted", r.get("test_f1_weighted"))
                    row["val_f1_macro"] = r.get("val_f1_macro", r.get("test_f1_macro"))
                    row["val_auc"] = _pick_auc(r)
                    row["cv_best_score"] = r.get("cv_best_score")
                    row["best_params"] = r.get("best_params", "")
                    rows.append(row)

    # ----- Leaf: leaf/{target}/1dcnn_image/metrics.tsv -----
    if leaf_dir.is_dir():
        for target_dir in sorted(leaf_dir.iterdir()):
            if not target_dir.is_dir():
                continue
            target = target_dir.name
            leaf_1d = target_dir / "1dcnn_image" / "metrics.tsv"
            outdir_1d = target_dir / "1dcnn_image"
            if leaf_1d.is_file():
                df = pd.read_csv(leaf_1d, sep="\t")
                df = _norm_acc_f1(df)
                for _, r in df.iterrows():
                    row = _row_base("leaf", target, "1dcnn", path_metrics=str(leaf_1d))
                    row["val_accuracy"] = r.get("val_accuracy", r.get("best_test_accuracy"))
                    row["val_f1_weighted"] = r.get("val_f1_weighted", r.get("best_test_f1_weighted"))
                    row["val_f1_macro"] = r.get("val_f1_macro", r.get("best_test_f1_macro"))
                    row["val_auc"] = _pick_auc(r)
                    row["best_epoch"] = r.get("best_epoch", "")
                    row["best_params"] = _format_run_params_tsv(outdir_1d / "run_params.tsv")
                    rows.append(row)

    # ----- Patch: 按 target 扫描 patch/<target>/all_models_summary.csv 与 1dcnn/2dcnn/3dcnn -----
    # target 可能带参数后缀（如 phase_core_s16_maxp200），便于多组 stride/max_patches 并存
    patch_dir = results / "patch"
    if patch_dir.is_dir():
        # 从子目录名得到 target：patch/phase_core_s16_maxp200, patch/physiological_state_s16_maxp200, ...
        for sub in sorted(patch_dir.iterdir()):
            if not sub.is_dir():
                continue
            target = sub.name
            # 跳过以模型名为目录名的（如 2dcnn_phase_core）
            if target.startswith(("1dcnn_image_", "2dcnn_", "3dcnn_")):
                continue

            # 7 ML
            p_summary = sub / "all_models_summary.csv"
            if p_summary.is_file():
                df = pd.read_csv(p_summary)
                df = _norm_acc_f1(df)
                for _, r in df.iterrows():
                    model = str(r.get("model", "unknown"))
                    row = _row_base("patch", target, model, path_summary=str(p_summary), path_metrics="")
                    row["val_accuracy"] = r.get("val_accuracy", r.get("test_accuracy"))
                    row["val_f1_weighted"] = r.get("val_f1_weighted", r.get("test_f1_weighted"))
                    row["val_f1_macro"] = r.get("val_f1_macro", r.get("test_f1_macro"))
                    row["val_auc"] = _pick_auc(r)
                    row["cv_best_score"] = r.get("cv_best_score")
                    row["best_params"] = r.get("best_params", "")
                    rows.append(row)

        # Patch CNN: 1dcnn_image_<target>, 2dcnn_<target>, 3dcnn_<target>
        for sub in sorted(patch_dir.iterdir()):
            if not sub.is_dir():
                continue
            name = sub.name
            if name.startswith("1dcnn_image_"):
                target = name.replace("1dcnn_image_", "")
                model = "1dcnn"
            elif name.startswith("2dcnn_"):
                target = name.replace("2dcnn_", "")
                model = "2dcnn"
            elif name.startswith("3dcnn_"):
                target = name.replace("3dcnn_", "")
                model = "3dcnn"
            else:
                continue
            m_path = sub / "metrics.tsv"
            if m_path.is_file():
                df = pd.read_csv(m_path, sep="\t")
                df = _norm_acc_f1(df)
                for _, r in df.iterrows():
                    row = _row_base("patch", target, model, path_metrics=str(m_path))
                    row["val_accuracy"] = r.get("val_accuracy", r.get("best_test_accuracy"))
                    row["val_f1_weighted"] = r.get("val_f1_weighted")
                    row["val_f1_macro"] = r.get("val_f1_macro")
                    row["val_auc"] = _pick_auc(r)
                    row["best_epoch"] = r.get("best_epoch", "")
                    row["best_params"] = _format_run_params_tsv(sub / "run_params.tsv")
                    rows.append(row)

    if not rows:
        pd.DataFrame(columns=OUTPUT_COLS).to_csv(out_path, sep="\t", index=False)
        print(f"[collect_hsi_index] 无任何结果，已写出空表: {out_path}")
        return

    out_df = pd.DataFrame(rows)
    for col in OUTPUT_COLS:
        if col not in out_df.columns:
            out_df[col] = ""
    out_df = out_df[OUTPUT_COLS]
    # 数值列转为字符串，避免科学计数法；空值保持为空字符串
    for col in ("val_accuracy", "val_f1_weighted", "val_f1_macro", "val_auc", "cv_best_score", "best_epoch"):
        if col in out_df.columns:
            out_df[col] = out_df[col].apply(lambda x: "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"[collect_hsi_index] 已写入 {len(out_df)} 行 -> {out_path}")


if __name__ == "__main__":
    main()
