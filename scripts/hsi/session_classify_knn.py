#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HSI session classification — KNN 单模型版
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    ap.add_argument('--outdir', default='results/hsi/ml/knn')
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--cv-folds', type=int, default=5)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--split-path', default=None, help='预先定义好的数据划分文件 (tsv)')
    ap.add_argument('--scoring', default='f1_weighted',
                    choices=['f1_weighted', 'f1_macro', 'accuracy'])
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    print(f"[KNN] outdir = {outdir}")

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

    knn = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier()),
    ])

    param_grid = {
        'clf__n_neighbors': [k for k in range(3, 51, 2)],
        'clf__weights': ['uniform', 'distance'],
        'clf__p': [1, 2],
    }

    print("[KNN] Grid size:", np.prod([len(v) for v in param_grid.values()]))

    search = GridSearchCV(
        estimator=knn,
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
    cv_df.to_csv(outdir / 'knn_cv_results.csv', index=False)

    best_knn = search.best_estimator_
    print("[KNN] best params:", search.best_params_)
    print(f"[KNN] cv best {args.scoring} = {search.best_score_:.4f}")

    y_pred = best_knn.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(outdir / 'knn_classification_report.csv')

    test_f1w = f1_score(y_test, y_pred, average='weighted')
    test_mac = f1_score(y_test, y_pred, average='macro')
    test_acc = accuracy_score(y_test, y_pred)

    summary = {
        'model': 'knn',
        'cv_best_score': search.best_score_,
        'test_f1_weighted': test_f1w,
        'test_f1_macro': test_mac,
        'test_accuracy': test_acc,
        'best_params': str(search.best_params_),
    }
    pd.DataFrame([summary]).to_csv(outdir / 'knn_summary.csv', index=False)

    import joblib
    joblib.dump(best_knn, outdir / 'knn_best_model.joblib')

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=classes,
        out_png=outdir / 'knn_confusion_matrix_norm.png',
        normalize=True,
    )

    y_test_str = np.array([classes[i] for i in y_test])
    plot_pca_scatter(
        X_test,
        y_labels_str=y_test_str,
        out_png=outdir / 'knn_pca_scatter.png',
        title='PCA (test) - KNN',
    )

    try_plot_roc_pr(best_knn, X_test, y_test, out_prefix=str(outdir / 'knn_prob'))

    # 主超参数趋势：n_neighbors
    plot_hparam_curve(
        cv_df,
        param_col='param_clf__n_neighbors',
        out_png=outdir / 'knn_hparam_curve_n_neighbors.png',
        scoring='mean_test_score',
        log_x=False,
        title='KNN: CV score vs n_neighbors',
    )

    print(f"[KNN] done. best test f1_weighted={test_f1w:.4f}")


if __name__ == '__main__':
    main()

