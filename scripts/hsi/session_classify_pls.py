#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HSI session classification — PLS-DA 单模型版

- 使用 PLSRegression 做多分类的 PLS-DA：
  * X: 高维光谱 + 指数
  * y: session_id（或 phase 等）
- 超参数：
  * n_components：主调参，自动根据样本数/特征数生成候选列表
- 输出：
  * pls_cv_results.csv           （完整 CV 日志）
  * pls_classification_report.csv
  * pls_summary.csv              （单行汇总）
  * pls_best_model.joblib        （joblib.dump）
  * pls_confusion_matrix_norm.png
  * pls_pca_scatter.png          （test 集 PCA 可视化）
  * pls_prob_roc.png / pls_prob_pr.png （如果有 predict_proba）
  * pls_feature_importance.csv   （基于回归系数的 L2 范数）
  * pls_feature_importance_top30.png
  * pls_hparam_curve_n_components.png  （n_components vs CV 分数）
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
)

# ---- 导入同目录下的 ml_utils ----
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


class PLSDA(BaseEstimator, ClassifierMixin):
    """
    简单 PLS-DA 封装：
    - 使用 PLSRegression 对 one-hot Y 做回归
    - 预测时对回归输出按行 argmax 取类别
    - predict_proba 简单归一化到 [0,1] 并按行归一
    """

    def __init__(self, n_components: int = 2, scale: bool = True):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        # 二分类/多分类都会得到 (n_samples, n_classes) 的矩阵
        Y = label_binarize(y, classes=self.classes_)
        self.pls_ = PLSRegression(
            n_components=self.n_components,
            scale=self.scale,
        )
        self.pls_.fit(X, Y)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        Y_pred = self.pls_.predict(X)  # (n_samples, n_classes)
        # 防守式处理：若出现 NaN，先填 0
        Y_pred = np.nan_to_num(Y_pred)
        idx = np.argmax(Y_pred, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        """
        非严格概率，只是将回归输出裁剪到 >=0 后按行归一。
        主要用于画 ROC/PR 曲线。
        """
        X = np.asarray(X, float)
        Y_pred = self.pls_.predict(X)
        Y_pred = np.nan_to_num(Y_pred)
        # 截断负值，避免概率为负
        Y_pred = np.clip(Y_pred, 0.0, None)
        row_sum = Y_pred.sum(axis=1, keepdims=True) + 1e-12
        proba = Y_pred / row_sum
        return proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='results/hsi/image_features.tsv')
    ap.add_argument('--sep', default='\t')
    ap.add_argument('--target', default='session_id')
    ap.add_argument('--outdir', default='results/hsi/ml/pls')
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--cv-folds', type=int, default=5)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--split-path', default=None, help='预先定义好的数据划分文件 (tsv)')
    ap.add_argument('--scoring', default='f1_weighted',
                    choices=['f1_weighted', 'f1_macro', 'accuracy'])
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    print(f"[PLS] outdir = {outdir}")

    # ---- 载入数据（复用 ml_utils.load_dataset）----
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

    # ---- CV 配置 ----
    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    # ---- 构造 PLS-DA 模型 & 超参数网格 ----
    from sklearn.pipeline import Pipeline

    pls_da = PLSDA()

    pipe = Pipeline([('clf', pls_da)])

    # n_components 上限不能超过 min(n_samples, n_features)
    max_comp_theory = min(X_train.shape[0] - 1, X_train.shape[1])
    max_comp = max(2, min(150, max_comp_theory))  # 最多 40，至少 2

    if max_comp <= 2:
        n_comp_list = [2]
    else:
        # 使用列表推导生成候选：2,4,6,...,max_comp
        n_comp_list = [k for k in range(2, max_comp + 1, 3)]

    param_grid = {
        'clf__n_components': n_comp_list,
    }

    print(f"[PLS] candidate n_components: {n_comp_list} (len={len(n_comp_list)})")

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
    cv_df.to_csv(outdir / 'pls_cv_results.csv', index=False)

    best_model = search.best_estimator_
    print("[PLS] best params:", search.best_params_)
    print(f"[PLS] cv best {args.scoring} = {search.best_score_:.4f}")

    # ---- 测试集评估 ----
    y_pred = best_model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(outdir / 'pls_classification_report.csv')

    test_f1w = f1_score(y_test, y_pred, average='weighted')
    test_mac = f1_score(y_test, y_pred, average='macro')
    test_acc = accuracy_score(y_test, y_pred)

    summary = {
        'model': 'pls',
        'cv_best_score': search.best_score_,
        'test_f1_weighted': test_f1w,
        'test_f1_macro': test_mac,
        'test_accuracy': test_acc,
        'best_params': str(search.best_params_),
    }
    pd.DataFrame([summary]).to_csv(outdir / 'pls_summary.csv', index=False)

    # ---- 保存模型 ----
    import joblib
    joblib.dump(best_model, outdir / 'pls_best_model.joblib')

    # ---- 图：混淆矩阵、PCA、ROC/PR ----
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=classes,
        out_png=outdir / 'pls_confusion_matrix_norm.png',
        normalize=True,
    )

    y_test_str = np.array([classes[i] for i in y_test])
    plot_pca_scatter(
        X_test,
        y_labels_str=y_test_str,
        out_png=outdir / 'pls_pca_scatter.png',
        title='PCA (test) - PLS-DA',
    )

    try:
        try_plot_roc_pr(
            best_model,
            X_test,
            y_test,
            out_prefix=str(outdir / 'pls_prob'),
        )
    except Exception as e:
        with open(outdir / 'pls_prob_error.txt', 'w') as f:
            f.write(str(e))

    # ---- Feature "importance"（基于 PLS 回归系数的 L2 范数）----
    try:
        base = best_model
        if hasattr(base, "named_steps") and 'clf' in base.named_steps:
            pls_step = base.named_steps['clf']
        else:
            pls_step = base

        if hasattr(pls_step, 'pls_') and hasattr(pls_step.pls_, 'coef_'):
            coef = np.asarray(pls_step.pls_.coef_, float)  # (n_features, n_classes)
            if coef.ndim == 1:
                coef = coef.reshape(-1, 1)
            # 对每个特征：对所有类别的系数做 L2 范数
            imp = np.linalg.norm(coef, axis=1)
            feat_imp_df = pd.DataFrame(
                {'feature': feat_cols, 'importance': imp}
            ).sort_values('importance', ascending=False)
            feat_imp_df.to_csv(outdir / 'pls_feature_importance.csv', index=False)

            plot_top_features(
                feat_names=feat_cols,
                importances=imp,
                out_png=outdir / 'pls_feature_importance_top30.png',
                topk=30,
                title='Feature importance (PLS-DA)',
            )
    except Exception as e:
        with open(outdir / 'pls_feature_importance_error.txt', 'w') as f:
            f.write(str(e))

    # ---- 超参数曲线：n_components vs CV score ----
    try:
        plot_hparam_curve(
            cv_df=cv_df,
            param_col='param_clf__n_components',
            out_png=outdir / 'pls_hparam_curve_n_components.png',
            scoring='mean_test_score',
            log_x=False,
            title='PLS-DA: CV score vs n_components',
        )
    except Exception as e:
        with open(outdir / 'pls_hparam_curve_error.txt', 'w') as f:
            f.write(str(e))

    print(f"[PLS] done. best test f1_weighted={test_f1w:.4f}")


if __name__ == '__main__':
    main()

