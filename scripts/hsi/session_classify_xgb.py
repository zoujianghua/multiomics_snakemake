#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HSI session classification — XGBoost 单模型版（使用 RandomizedSearchCV）
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
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
    plot_top_features,
    plot_hparam_curve,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='results/hsi/image_features.tsv')
    ap.add_argument('--sep', default='\t')
    ap.add_argument('--target', default='session_id')
    ap.add_argument('--outdir', default='results/hsi/ml/xgb')
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--cv-folds', type=int, default=5)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--split-path', default=None, help='预先定义好的数据划分文件 (tsv)')
    ap.add_argument(
        '--scoring',
        default='f1_weighted',
        choices=['f1_weighted', 'f1_macro', 'accuracy']
    )
    ap.add_argument(
        '--xgb-gpu',
        action='store_true',
        help='启用 GPU（device=cuda）'
    )
    ap.add_argument(
        '--n-iter',
        type=int,
        default=50,
        help='RandomizedSearchCV 采样的超参数组合数（默认 200，可按算力调节）'
    )
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    print(f"[XGB] outdir = {outdir}")

    # ---------------- XGBoost 导入 ----------------
    try:
        import xgboost as xgb
        from xgboost import XGBClassifier
    except Exception as e:
        raise RuntimeError("未安装 xgboost，请先在 hsi 环境中安装.") from e

    # ---------------- 版本与 build_info 打印 ----------------
    print(f"[XGB] xgboost version = {xgb.__version__}")
    try:
        info = xgb.build_info()
        # build_info() 可能返回字典或字符串，尝试多种方式获取 USE_CUDA
        if isinstance(info, dict):
            use_cuda = info.get("USE_CUDA", None)
        else:
            # 如果是字符串，尝试解析
            use_cuda = None
            if hasattr(info, "get"):
                use_cuda = info.get("USE_CUDA", None)
        print(f"[XGB] build_info.USE_CUDA = {use_cuda}")
        # 打印完整的 build_info 以便调试
        if isinstance(info, dict):
            print(f"[XGB] build_info keys: {list(info.keys())}")
    except Exception as e:
        print(f"[XGB][WARN] failed to query build_info(): {e}")
        info = {}
        use_cuda = None

    # ---------------- 数据加载 ----------------
    X_train, X_test, y_train, y_test, classes, feat_cols, le = load_dataset(
        images_path=args.images,
        sep=args.sep,
        target=args.target,
        test_size=args.test_size,
        split_path=args.split_path,
        random_state=args.random_state,
    )

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state
    )

    # ---------------- GPU 自检 + 退回 CPU 的逻辑 ----------------
    tree_method = "hist"
    want_gpu = args.xgb_gpu
    device = "cuda" if want_gpu else "cpu"

    if want_gpu:
        print("[XGB] --xgb-gpu 已指定，开始 GPU 自检...")
        # 1) 编译期是否启用 CUDA
        if not use_cuda:
            print("[XGB][WARN] xgboost build_info() 表示未启用 CUDA (USE_CUDA=False/None)，自动退回 CPU。")
            device = "cpu"
        else:
            # 2) 运行时做一个极小的 smoke test，确认 device='cuda' 真能用
            try:
                print("[XGB] 尝试在 device='cuda' 上做一个极小的 smoke test ...")
                X_dummy = np.random.randn(100, 10).astype(np.float32)
                y_dummy = (X_dummy[:, 0] > 0).astype(int)
                dtrain = xgb.DMatrix(X_dummy, label=y_dummy)
                params_test = {
                    "tree_method": "hist",
                    "device": "cuda",
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "objective": "binary:logistic",
                }
                # 执行 smoke test
                bst_test = xgb.train(params_test, dtrain, num_boost_round=1, verbose_eval=False)
                # 尝试预测以确认 GPU 真的在工作
                dtest = xgb.DMatrix(X_dummy[:10], label=y_dummy[:10])
                bst_test.predict(dtest)
                print("[XGB] smoke test OK，GPU 训练可用。")
            except Exception as e:
                print(f"[XGB][WARN] 在 device='cuda' 上做 smoke test 失败，自动退回 CPU。")
                print(f"[XGB][WARN] 错误详情: {type(e).__name__}: {e}")
                device = "cpu"
    else:
        print("[XGB] 未指定 --xgb-gpu，使用 CPU 训练。")

    # 最终确认使用的设备
    if device == "cpu" and want_gpu:
        print("[XGB][WARN] =========================================")
        print("[XGB][WARN] GPU 不可用，已自动降级到 CPU 模式。")
        print("[XGB][WARN] 请检查：")
        print("[XGB][WARN]   1. CUDA 驱动是否正确安装")
        print("[XGB][WARN]   2. GPU 是否可用（nvidia-smi）")
        print("[XGB][WARN]   3. XGBoost 是否使用 GPU 支持编译")
        print("[XGB][WARN] =========================================")
    
    print(f"[XGB] final device = {device}, tree_method = {tree_method}")

    # ---------------- 模型定义（适配 xgboost>=2） ----------------
    xgb_clf = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method=tree_method,
        device=device,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )

    # ---------------- 超参数空间（仍然是“大空间”，但随机采样） ----------------
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [0.1, 1, 5, 10],
    }

    # 理论上的网格大小（仅用于打印，不实际穷举）
    grid_size = int(np.prod([len(v) for v in param_dist.values()]))
    print(f"[XGB] Param space size (theoretical grid) = {grid_size}")
    print(f"[XGB] RandomizedSearchCV n_iter = {args.n_iter}")
    print(f"[XGB] training with device={device}, n_jobs={args.n_jobs}")

    search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        cv=cv,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        refit=True,
        verbose=1,
        return_train_score=True,)

    # ---------------- 训练 + CV ----------------
    search.fit(X_train, y_train)

    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.to_csv(outdir / 'xgb_cv_results.csv', index=False)

    best_xgb = search.best_estimator_
    print("[XGB] best params:", search.best_params_)
    print(f"[XGB] cv best {args.scoring} = {search.best_score_:.4f}")

    # ---------------- 测试集评估 ----------------
    y_pred = best_xgb.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(outdir / 'xgb_classification_report.csv')

    test_f1w = f1_score(y_test, y_pred, average='weighted')
    test_mac = f1_score(y_test, y_pred, average='macro')
    test_acc = accuracy_score(y_test, y_pred)

    summary = {
        'model': 'xgb',
        'cv_best_score': search.best_score_,
        'test_f1_weighted': test_f1w,
        'test_f1_macro': test_mac,
        'test_accuracy': test_acc,
        'best_params': str(search.best_params_),
    }
    pd.DataFrame([summary]).to_csv(outdir / 'xgb_summary.csv', index=False)

    import joblib
    joblib.dump(best_xgb, outdir / 'xgb_best_model.joblib')

    # ---------------- 各种图 ----------------
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=classes,
        out_png=outdir / 'xgb_confusion_matrix_norm.png',
        normalize=True,
    )

    y_test_str = np.array([classes[i] for i in y_test])
    plot_pca_scatter(
        X_test,
        y_labels_str=y_test_str,
        out_png=outdir / 'xgb_pca_scatter.png',
        title='PCA (test) - XGB',
    )

    try_plot_roc_pr(best_xgb, X_test, y_test, out_prefix=str(outdir / 'xgb_prob'))

    # ---------------- 特征重要性（修掉长度不一致的 bug） ----------------
    if hasattr(best_xgb, 'feature_importances_'):
        imp = np.asarray(best_xgb.feature_importances_, float)
        n_imp = imp.shape[0]
        n_feat = len(feat_cols)

        feat_for_imp = list(feat_cols)
        try:
            booster = best_xgb.get_booster()
            fnames = booster.feature_names
            if fnames is not None and len(fnames) == n_imp:
                feat_for_imp = list(fnames)
        except Exception:
            pass

        if n_imp != len(feat_for_imp):
            k = min(n_imp, len(feat_for_imp))
            print(
                f"[XGB][WARN] len(feature_importances_)={n_imp}, "
                f"len(features)={len(feat_for_imp)}, truncate to {k}",
                flush=True,
            )
            imp = imp[:k]
            feat_for_imp = feat_for_imp[:k]

        df_imp = (
            pd.DataFrame({'feature': feat_for_imp, 'importance': imp})
            .sort_values('importance', ascending=False)
        )
        df_imp.to_csv(outdir / 'xgb_feature_importance.csv', index=False)

        plot_top_features(
            feat_names=feat_for_imp,
            importances=imp,
            out_png=outdir / 'xgb_feature_importance_top30.png',
            topk=min(30, len(imp)),
            title='Feature importance (XGB)',)
   
    plot_hparam_curve(
        cv_df,
        param_col='param_n_estimators',
        out_png=outdir / 'xgb_hparam_curve_n_estimators.png',
        scoring='mean_test_score',
        log_x=False,
        title='XGB: CV score vs n_estimators',
    )

    print(f"[XGB] done. best test f1_weighted={test_f1w:.4f}")


if __name__ == '__main__':
    main()

