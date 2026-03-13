#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HSI session classification — LDA 单模型版

- 模型：StandardScaler + LinearDiscriminantAnalysis
- 超参数：
  * solver ∈ {"lsqr", "eigen"}
  * shrinkage ∈ [None, 0.0, 一堆小数, "auto"]
  * 对 eigen 再加 n_components（不能超过 n_classes-1）
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score

# ---- 导入 ml_utils ----
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ml_utils import (  # type: ignore
    ensure_dir,
    load_dataset,
    plot_confusion_matrix,
    plot_pca_scatter,
    try_plot_roc_pr,
    plot_top_features,
    plot_hparam_curve,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="results/hsi/image_features.tsv")
    ap.add_argument("--sep", default="\t")
    ap.add_argument("--target", default="session_id")
    ap.add_argument("--outdir", default="results/hsi/ml/lda")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument('--split-path', default=None, help='预先定义好的数据划分文件 (tsv)')
    ap.add_argument(
        "--scoring",
        default="f1_weighted",
        choices=["f1_weighted", "f1_macro", "accuracy"],
    )
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    print(f"[LDA] outdir = {outdir}")

    # -------- 数据加载 --------
    (
        X_train,
        X_test,
        y_train,
        y_test,
        classes,
        feat_cols,
        le,
    ) = load_dataset(
        images_path=args.images,
        sep=args.sep,
        target=args.target,
        test_size=args.test_size,
        split_path=args.split_path,
        random_state=args.random_state,
    )

    n_classes = len(classes)
    max_n_comp = max(1, n_classes - 1)

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    lda = LinearDiscriminantAnalysis()  # 具体参数交给 GridSearchCV

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", lda),
        ]
    )

    # shrinkage 候选：None / 0 / 一些小值 / auto
    shrinkage_grid = (
        [None, 0.0]
        + [round(s, 3) for s in np.linspace(0.001, 0.05, 5)]
        + ["auto"]
    )

    # n_components 候选，注意不能 > n_classes - 1
    base_n_comp = [3, 5, 10, 13, 15, 17, 19, 21, 22]
    n_comp_grid = [nc for nc in base_n_comp if nc <= max_n_comp]
    n_comp_grid = sorted(set(n_comp_grid))
    n_comp_grid = [None] + n_comp_grid  # None 表示使用全部判别方向

    param_grid = [
        # 方案 1：lsqr + 各种 shrinkage（不调 n_components）
        {
            "clf__solver": ["lsqr"],
            "clf__shrinkage": shrinkage_grid,
        },
        # 方案 2：eigen + shrinkage + n_components
        {
            "clf__solver": ["eigen"],
            "clf__shrinkage": [None, 0.0, 0.01, 0.05, "auto"],
            "clf__n_components": n_comp_grid,
        },
    ]

    print(f"[LDA] candidate shrinkage (lsqr grid): {shrinkage_grid} (len={len(shrinkage_grid)})")
    print(f"[LDA] candidate n_components (eigen grid, <= {max_n_comp}): {n_comp_grid}")

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
        refit=True,
        verbose=0,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.to_csv(outdir / "lda_cv_results.csv", index=False)

    best_model = search.best_estimator_
    print("[LDA] best params:", search.best_params_)
    print(f"[LDA] cv best {args.scoring} = {search.best_score_:.4f}")

    # -------- 测试集评估 --------
    y_pred = best_model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(outdir / "lda_classification_report.csv")

    test_f1w = f1_score(y_test, y_pred, average="weighted")
    test_mac = f1_score(y_test, y_pred, average="macro")
    test_acc = accuracy_score(y_test, y_pred)

    summary = {
        "model": "lda",
        "cv_best_score": search.best_score_,
        "test_f1_weighted": test_f1w,
        "test_f1_macro": test_mac,
        "test_accuracy": test_acc,
        "best_params": str(search.best_params_),
    }
    pd.DataFrame([summary]).to_csv(outdir / "lda_summary.csv", index=False)

    import joblib

    joblib.dump(best_model, outdir / "lda_best_model.joblib")

    # -------- 可视化 --------
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=classes,
        out_png=outdir / "lda_confusion_matrix_norm.png",
        normalize=True,
    )

    y_test_str = np.array([classes[i] for i in y_test])
    plot_pca_scatter(
        X_test,
        y_labels_str=y_test_str,
        out_png=outdir / "lda_pca_scatter.png",
        title="PCA (test) - LDA",
    )

    try:
        try_plot_roc_pr(
            best_model,
            X_test,
            y_test,
            out_prefix=str(outdir / "lda_prob"),
        )
    except Exception as e:
        with open(outdir / "lda_prob_error.txt", "w") as f:
            f.write(str(e))

    # -------- 基于 coef_ 的“重要性” --------
    try:
        base = best_model
        if hasattr(base, "named_steps") and "clf" in base.named_steps:
            lda_step = base.named_steps["clf"]
        else:
            lda_step = base

        if hasattr(lda_step, "coef_"):
            coef = np.asarray(lda_step.coef_, float)  # (n_components, n_features)
            if coef.ndim == 1:
                coef = coef.reshape(1, -1)
            imp = np.linalg.norm(coef, axis=0)  # 每个特征，对所有判别方向的 L2 范数

            feat_imp_df = (
                pd.DataFrame({"feature": feat_cols, "importance": imp})
                .sort_values("importance", ascending=False)
            )
            feat_imp_df.to_csv(outdir / "lda_feature_importance.csv", index=False)

            plot_top_features(
                feat_names=feat_cols,
                importances=imp,
                out_png=outdir / "lda_feature_importance_top30.png",
                topk=min(30, len(imp)),
                title="Feature importance (LDA)",
            )
    except Exception as e:
        with open(outdir / "lda_feature_importance_error.txt", "w") as f:
            f.write(str(e))

    # -------- shrinkage vs CV 分数 曲线 --------
    try:
        plot_hparam_curve(
            cv_df=cv_df,
            param_col="param_clf__shrinkage",
            out_png=outdir / "lda_hparam_curve_shrinkage.png",
            scoring="mean_test_score",
            log_x=False,
            title="LDA: CV score vs shrinkage",
        )
    except Exception as e:
        with open(outdir / "lda_hparam_curve_error.txt", "w") as f:
            f.write(str(e))

    print(f"[LDA] done. best test f1_weighted={test_f1w:.4f}")


if __name__ == "__main__":
    main()

