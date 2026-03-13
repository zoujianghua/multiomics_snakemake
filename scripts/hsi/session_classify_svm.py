#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HSI session classification — SVM 单模型版
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ml_utils import (
    ensure_dir,
    load_dataset,
    plot_confusion_matrix,
    plot_pca_scatter,
    try_plot_roc_pr,
    plot_hparam_curve,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='results/hsi/image_features.tsv')
    ap.add_argument('--sep', default='\t')
    ap.add_argument('--target', default='session_id')
    ap.add_argument('--outdir', default='results/hsi/ml/svm')
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--cv-folds', type=int, default=5)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--split-path', default=None, help='预先定义好的数据划分文件 (tsv)')
    ap.add_argument('--scoring', default='f1_weighted',
                    choices=['f1_weighted', 'f1_macro', 'accuracy'])
    ap.add_argument('--n-iter', type=int, default=48,
                    help='RandomizedSearchCV 采样次数，默认 48，兼顾速度与搜索覆盖')
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    print(f"[SVM] outdir = {outdir}")

    X_train, X_test, y_train, y_test, classes, feat_cols, le = load_dataset(
        images_path=args.images,
        sep=args.sep,
        target=args.target,
        test_size=args.test_size,
        split_path=args.split_path,
        random_state=args.random_state,
    )

    cv = StratifiedKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.random_state
    )

    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, class_weight='balanced',
                   random_state=args.random_state)),
    ])

    # 参数空间：RandomizedSearchCV 采样，避免全网格（原 2*15*15=450 次过慢）
    param_distributions = {
        'clf__kernel': ['linear', 'rbf'],
        'clf__C': loguniform(1e-3, 1e3),
        'clf__gamma': loguniform(1e-4, 10),
    }

    print(f"[SVM] RandomizedSearchCV n_iter={args.n_iter}")

    search = RandomizedSearchCV(
        estimator=svm,
        param_distributions=param_distributions,
        n_iter=getattr(args, 'n_iter', 48),
        cv=cv,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
        refit=True,
        random_state=args.random_state,
        verbose=0,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.to_csv(outdir / 'svm_cv_results.csv', index=False)

    best_svm = search.best_estimator_
    print("[SVM] best params:", search.best_params_)
    print(f"[SVM] cv best {args.scoring} = {search.best_score_:.4f}")
    import joblib
    joblib.dump(best_svm, outdir / 'svm_best_model.joblib')

    y_pred = best_svm.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(outdir / 'svm_classification_report.csv')

    test_f1w = f1_score(y_test, y_pred, average='weighted')
    test_mac = f1_score(y_test, y_pred, average='macro')
    test_acc = accuracy_score(y_test, y_pred)

    summary = {
        'model': 'svm',
        'cv_best_score': search.best_score_,
        'test_f1_weighted': test_f1w,
        'test_f1_macro': test_mac,
        'test_accuracy': test_acc,
        'best_params': str(search.best_params_),
    }
    pd.DataFrame([summary]).to_csv(outdir / 'svm_summary.csv', index=False)

    # 混淆矩阵
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=classes,
        out_png=outdir / 'svm_confusion_matrix_norm.png',
        normalize=True,
    )

    # PCA
    y_test_str = np.array([classes[i] for i in y_test])
    plot_pca_scatter(
        X_test,
        y_labels_str=y_test_str,
        out_png=outdir / 'svm_pca_scatter.png',
        title='PCA (test) - SVM',
    )

    # ROC / PR
    try_plot_roc_pr(best_svm, X_test, y_test, out_prefix=str(outdir / 'svm_prob'))

    # 超参数趋势图：C vs CV
    plot_hparam_curve(
        cv_df,
        param_col='param_clf__C',
        out_png=outdir / 'svm_hparam_curve_C.png',
        scoring='mean_test_score',
        log_x=True,
        title='SVM: CV score vs C',
    )

    print(f"[SVM] done. best test f1_weighted={test_f1w:.4f}")


if __name__ == '__main__':
    main()

