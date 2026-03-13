#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通用 ML 工具：数据加载 + 常用画图
供 session_classify_*.py 复用。
"""

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.decomposition import PCA


# ----------- 基础工具 -----------

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_float_str(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def pick_feature_columns(df: pd.DataFrame,feature_mode="full"):
    """
    选特征列：

    - 光谱列：所有 R_<nm>，nm 是数值（比如 R_550, R_710）
    - 指数/其它数值特征列：表里所有数值型列，去掉一批明确不要作为特征的
      元数据/路径/标记列（包括 R800_med），剩下的全部作为特征。

    注意：
    - 不再把 R800_med 当作特征。
    """

    # 1) 光谱列：R_ 开头 + 数字波长
    spec_cols = [
        c for c in df.columns
        if c.startswith("r_") and is_float_str(str(c)[2:])
    ]
    # 按波长排序一下（可选）
    try:
        spec_cols = sorted(spec_cols, key=lambda c: float(c[2:]))
    except Exception:
        spec_cols = sorted(spec_cols)
    
    if feature_mode =="spec":
        return spec_cols

    # 2) 明确不要作为特征的一批列（元数据 / 路径 / 标签 / 质控）
    banned_raw = {
        # 你刚才列出来的
        "sample_id",
        "group_dir",
        "sample_dir",
        "capture_dir",
        "sample_hdr",
        "white_hdr",
        "dark_hdr",
        "temp",
        "time",
        "time_h",
        "phase",
        "replicate",
        "roi_area",
        "ndre_thr",
        "seg_plan",
        "spec_npz",
        "mask_png",
        "overlay_png",
        "use_raw_mode",
        "R800_med",   # 原始列名
        "r800_med",   # 以防有小写版本

        # 其它常见的“标签/分组”类列
        "session_id",
        "group",
        "label",
        "y",
    }
    # 用 lower() 再做一层保险，避免大小写问题
    banned_lower = {c.lower() for c in banned_raw}

    # 3) 数值型特征列：所有 numeric 列，去掉光谱列和 banned 列
    num_cols = []
    for c in df.columns:
        # 已经作为光谱列的，不再重复收集
        if c in spec_cols:
            continue

        # 不要在特征里出现的，直接跳过
        if (c in banned_raw) or (c.lower() in banned_lower):
            continue

        # 只要是数值型，就当成“指数/数值特征”
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)

    num_cols = sorted(num_cols)

    feature_cols = spec_cols + num_cols
    return feature_cols



def safe_y(df: pd.DataFrame, target: str):
    y = df[target].astype(str)
    m = y.notna() & (y.str.len() > 0)
    return y[m], m


def load_dataset(
    images_path,
    sep,
    target,
    test_size,
    random_state,
    split_path=None,        # 新增
    feature_mode: str = "full",   # 新增参数，默认还是 full
):
    images_path = Path(images_path)
    df = pd.read_csv(images_path, sep=sep)
    df.columns = [c.strip().lower() for c in df.columns]

    target = target.lower()
    if target not in df.columns:
        raise RuntimeError(f"目标列 {target} 不存在.")

    feat_cols = pick_feature_columns(df,feature_mode=feature_mode)
    if not feat_cols:
        raise RuntimeError("未找到任何特征列")

    y_raw, mask = safe_y(df, target)
    X = df.loc[mask, feat_cols].to_numpy(dtype=float)
    y_raw = y_raw.to_numpy(str)

    # 去 NaN 行
    ok = np.isfinite(X).all(axis=1)
    X = X[ok]
    y_raw = y_raw[ok]

    # label 编码
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_

    if split_path is None:
        # 老逻辑：内部自己 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    else:
        # 新逻辑：按外部 split_phase_seed42.tsv 来切
        split_df = pd.read_csv(split_path, sep="\t")
        split_df["sample_id"] = split_df["sample_id"].astype(str)

        # image_features 里 sample_id 已经小写过
        sid_col = "sample_id"
        if sid_col not in df.columns:
            raise RuntimeError("image_features.tsv 缺 sample_id 列，无法按 split 对齐")

        sid_all = df.loc[mask, sid_col].astype(str).to_numpy()
        split_map = dict(zip(split_df["sample_id"], split_df["split"]))

        # 为每一行分配 split 标记
        split_tag = [split_map.get(s, "train") for s in sid_all]
        split_tag = np.array(split_tag)

        train_idx = split_tag == "train"
        test_idx  = split_tag == "test"

        if train_idx.sum() == 0 or test_idx.sum() == 0:
            raise RuntimeError("split 文件导致 train 或 test 为空，请检查 split 和 image_features 对齐情况")

        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

    return X_train, X_test, y_train, y_test, classes, feat_cols, le



# ----------- 通用画图 -----------

def plot_confusion_matrix(y_true, y_pred, class_names, out_png, normalize=True):
    """
    y_true/y_pred 是数值 label（0..K-1），坐标轴上用 class_names。
    """
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.5),
                                    max(4.5, len(class_names) * 0.35)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (normalized)" if normalize else ""))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            s = f"{val:.2f}" if normalize else str(int(val))
            ax.text(j, i, s, ha="center", va="center", color="black", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # 同时保存数值版混淆矩阵为 TSV，便于后续在 R / Python 中重新绘图。
    # 命名中尽量包含“标签名 + 模型名”，方便在本地批量整理。
    out_png = Path(out_png)
    # 约定目录结构类似：
    #   results/hsi/patch/physiological_state/rf/rf_confusion_matrix_norm.png
    #   results/hsi/patch/phase_core/rf/rf_confusion_matrix_norm.png
    #   results/hsi/leaf/rf/rf_confusion_matrix_norm.png
    #   results/hsi/ml/rf/rf_confusion_matrix_norm.png
    model_name = out_png.parent.name              # 例如 rf
    target_name = out_png.parent.parent.name      # 例如 physiological_state / phase_core / leaf / ml

    # 生成带 target + model 的文件名，保持在同一目录下
    cm_tsv_name = f"{target_name}_{model_name}_confusion_matrix_values.tsv"
    cm_tsv = out_png.with_name(cm_tsv_name)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(cm_tsv, sep="\t")


def plot_pca_scatter(X, y_labels_str, out_png, title="PCA (first two components)"):
    """
    X: ndarray [n, d]
    y_labels_str: array[str]，已经映射为类别名
    """
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X)
    labels = np.unique(y_labels_str)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    cmap = plt.cm.get_cmap("tab20", len(labels))
    for i, lab in enumerate(labels):
        mask = (y_labels_str == lab)
        ax.scatter(X2[mask, 0], X2[mask, 1], s=15, alpha=0.7, label=str(lab), color=cmap(i))

    var_ratio = pca.explained_variance_ratio_
    ax.set_title(f"{title} | var={var_ratio[0]:.2f}+{var_ratio[1]:.2f}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(ncol=3, fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def try_plot_roc_pr(clf, X_test, y_test, out_prefix):
    """
    多分类 ROC / PR (OvR)
    """
    # 取概率/decision_function
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_test)
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_test)
        if scores.ndim == 1:
            return
    else:
        return

    n_classes = scores.shape[1]
    y_bin = np.eye(n_classes)[y_test]

    # ROC
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
        ax.plot(fpr, tpr, label=f"class {i} (AUC={auc(fpr, tpr):.3f})", linewidth=1.4)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC (OvR)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_roc.png", bbox_inches="tight")
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for i in range(n_classes):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], scores[:, i])
        ap = average_precision_score(y_bin[:, i], scores[:, i])
        ax.plot(rec, prec, label=f"class {i} (AP={ap:.3f})", linewidth=1.4)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR (OvR)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_pr.png", bbox_inches="tight")
    plt.close(fig)


def plot_top_features(feat_names, importances, out_png, topk=30, title="Feature importance"):
    order = np.argsort(importances)[::-1][:topk]
    fig, ax = plt.subplots(figsize=(8.6, max(4.2, topk * 0.25)))
    ax.barh(range(len(order)), importances[order][::-1], align="center")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feat_names[i] for i in order][::-1], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_hparam_curve(cv_df: pd.DataFrame, param_col: str, out_png,
                      scoring="mean_test_score", log_x=False, title=None):
    """
    从 Grid/RandomizedSearchCV.cv_results_ 里，对单个超参数画
    “值 vs 该值下最优 CV 分数” 趋势（best-on-each-param 模式）。

    做法：对每个 param_col 的取值，找出 scoring 最高的那一行，
    然后画 param_value -> best_score。
    """
    if param_col not in cv_df.columns or scoring not in cv_df.columns:
        return

    sub = cv_df[[param_col, scoring]].copy()
    sub[param_col] = pd.to_numeric(sub[param_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return

    # 每个参数值，取该值下的最高得分
    agg = sub.groupby(param_col)[scoring].max().reset_index()

    xs = agg[param_col].to_numpy()
    ys = agg[scoring].to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(xs, ys, "o-", linewidth=1.4)
    ax.set_xlabel(param_col)
    ax.set_ylabel(scoring)
    if title is None:
        title = f"{param_col} vs {scoring} (best per {param_col})"
    ax.set_title(title)

    if log_x:
        try:
            ax.set_xscale("log")
        except Exception:
            pass

    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


