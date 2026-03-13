#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HSI session classification — RandomForest 单模型版（RandomizedSearchCV 加速版）
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
)

# 把当前目录加到 sys.path，方便 import ml_utils
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ml_utils import (
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
    ap.add_argument('--images', default='results/hsi/image_features.tsv')
    ap.add_argument('--sep', default='\t')
    ap.add_argument('--target', default='session_id')
    ap.add_argument('--outdir', default='results/hsi/ml/rf')
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--cv-folds', type=int, default=5)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--split-path', default=None, help='预先定义好的数据划分文件 (tsv)')
    ap.add_argument(
        '--scoring',
        default='f1_weighted',
        choices=['f1_weighted', 'f1_macro', 'accuracy'],
    )
    ap.add_argument(
        '--n-iter',
        type=int,
        default=40,
        help='RandomizedSearchCV 采样的超参数组合数（默认 40，适合大样本 patch 任务）',
    )
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    print(f"[RF] outdir = {outdir}")

    # ---------------- 数据 ----------------
    X_train, X_test, y_train, y_test, classes, feat_cols, le = load_dataset(
        images_path=args.images,
        sep=args.sep,
        target=args.target,
        test_size=args.test_size,
        split_path=args.split_path,
        random_state=args.random_state,
    )

    # ---------------- CV ----------------
    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    # ---------------- 模型基底 ----------------
    rf = RandomForestClassifier(
        n_estimators=200,                 # 只是初始值，后面会被搜索覆盖
        class_weight='balanced_subsample',
        n_jobs=1,                         # 并行交给 RandomizedSearchCV 控制
        random_state=args.random_state,
    )

    # ---------------- 超参数搜索空间（离散列表，用随机搜索抽样） ----------------
    param_dist = {
        # 更紧凑的参数空间：适合大样本 patch 任务，显著减少单次 CV 开销
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ["sqrt", "log2",0.2],
    }

    grid_size = int(np.prod([len(v) for v in param_dist.values()]))
    print(f"[RF] Param space size (theoretical grid) = {grid_size}")
    print(f"[RF] RandomizedSearchCV n_iter = {args.n_iter}")

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        cv=cv,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        refit=True,
        verbose=1,
        return_train_score=True,
    )

    # ---------------- 训练 + CV ----------------
    search.fit(X_train, y_train)

    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.to_csv(outdir / 'rf_cv_results.csv', index=False)

    best_rf = search.best_estimator_
    print("[RF] best params:", search.best_params_)
    print(f"[RF] cv best {args.scoring} = {search.best_score_:.4f}")

    # ---------------- 测试集评估 ----------------
    y_pred = best_rf.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(outdir / 'rf_classification_report.csv')

    test_f1w = f1_score(y_test, y_pred, average='weighted')
    test_mac = f1_score(y_test, y_pred, average='macro')
    test_acc = accuracy_score(y_test, y_pred)

    summary = {
        'model': 'rf',
        'cv_best_score': search.best_score_,
        'test_f1_weighted': test_f1w,
        'test_f1_macro': test_mac,
        'test_accuracy': test_acc,
        'best_params': str(search.best_params_),
    }
    pd.DataFrame([summary]).to_csv(outdir / 'rf_summary.csv', index=False)

    # ---------------- 保存模型 ----------------
    import joblib
    joblib.dump(best_rf, outdir / 'rf_best_model.joblib')

    # ---------------- 混淆矩阵 ----------------
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=classes,
        out_png=outdir / 'rf_confusion_matrix_norm.png',
        normalize=True,
    )

    # ---------------- PCA（测试集） ----------------
    y_test_str = np.array([classes[i] for i in y_test])
    plot_pca_scatter(
        X_test,
        y_labels_str=y_test_str,
        out_png=outdir / 'rf_pca_scatter.png',
        title='PCA (test) - RF',
    )

    # ---------------- ROC / PR ----------------
    try_plot_roc_pr(best_rf, X_test, y_test, out_prefix=str(outdir / 'rf_prob'))

    # ---------------- 特征重要性 ----------------
    if hasattr(best_rf, 'feature_importances_'):
        imp = np.asarray(best_rf.feature_importances_, float)
        pd.DataFrame(
            {'feature': feat_cols, 'importance': imp}
        ).sort_values('importance', ascending=False).to_csv(
            outdir / 'rf_feature_importance.csv', index=False
        )
        plot_top_features(
            feat_names=feat_cols,
            importances=imp,
            out_png=outdir / 'rf_feature_importance_top30.png',
            topk=30,
            title='Feature importance (RF)',
        )

    # ---------------- 超参数趋势图（n_estimators） ----------------
    plot_hparam_curve(
        cv_df,
        param_col='param_n_estimators',
        out_png=outdir / 'rf_hparam_curve_n_estimators.png',
        scoring='mean_test_score',
        log_x=False,
        title='RF: CV score vs n_estimators',
    )

    print(f"[RF] done. best test_f1_weighted={test_f1w:.4f}")


if __name__ == '__main__':
    main()

