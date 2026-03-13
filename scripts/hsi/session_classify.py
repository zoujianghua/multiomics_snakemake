#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSI session classification (CSV only, dense hyper-param search, robust outputs)

- 读取 results/hsi/image_features.tsv（含 R_* 光谱列与指数）
- 以 --target (默认 session_id) 作为标签
- 模型集合：RF / SVM / KNN / LR / (可选) XGBoost
- 大模型用 RandomizedSearchCV，其余 GridSearchCV
- 统一前缀 'clf__'，自动过滤无效键；空网格→[{}]
- 输出：最佳模型、CV日志、测试报告、混淆矩阵、PCA、ROC/PR（若可）、特征重要性、SHAP（若可）
"""

import argparse, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, accuracy_score
)
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# ----------------- small utils -----------------
def ensure_dir(p: Path | str) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def is_float_str(s: str) -> bool:
    try: float(s); return True
    except: return False

def pick_feature_columns(df: pd.DataFrame) -> list[str]:
    """R_*光谱 + 常用指数/质量列"""
    cols = [c for c in df.columns if c.lower().startswith('r_') and is_float_str(str(c)[2:])]
    extra = [c for c in ['ndvi','gndvi','pri','ari','rep_d1','r800_med',] if c in df.columns]
    return cols + extra

def safe_y(df: pd.DataFrame, target: str):
    y = df[target].astype(str)
    m = y.notna() & (y.str.len() > 0)
    return y[m], m

def plot_confusion_matrix(y_true, y_pred, labels, out_png, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        with np.errstate(invalid="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)*0.5), max(4.5, len(labels)*0.35)))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion matrix' + (' (normalized)' if normalize else ''))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            s = f"{val:.2f}" if normalize else str(int(val))
            ax.text(j, i, s, ha='center', va='center', color='black', fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)

def plot_pca_scatter(X, y, out_png, title='PCA (first two components)'):
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X)
    labels = np.unique(y)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    cmap = plt.cm.get_cmap('tab20', len(labels))
    for i, lab in enumerate(labels):
        mask = (y == lab)
        ax.scatter(X2[mask,0], X2[mask,1], s=15, alpha=0.7, label=str(lab), color=cmap(i))
    ax.set_title(title + f" | var={pca.explained_variance_ratio_[0]:.2f}+{pca.explained_variance_ratio_[1]:.2f}")
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(ncol=3, fontsize=8, frameon=True)
    fig.tight_layout(); fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)

def try_plot_roc_pr(clf, X_test, y_test, classes, out_prefix):
    # get scores
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_test)
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_test)
        if scores.ndim == 1:
            scores = None
    else:
        scores = None
    if scores is None: return

    y_bin = label_binarize(y_test, classes=classes)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1-y_bin, y_bin])
    if scores.shape[1] != y_bin.shape[1]:
        return

    # ROC
    fig, ax = plt.subplots(figsize=(8,5.2))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
        ax.plot(fpr, tpr, label=f'{c} (AUC={auc(fpr,tpr):.3f})', linewidth=1.4)
    ax.plot([0,1],[0,1],'k--',alpha=0.6)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC (OvR)')
    ax.grid(True, linestyle='--', alpha=0.5); ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out_prefix}_roc.png", bbox_inches='tight'); plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(8,5.2))
    for i, c in enumerate(classes):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], scores[:, i])
        ap = average_precision_score(y_bin[:, i], scores[:, i])
        ax.plot(rec, prec, label=f'{c} (AP={ap:.3f})', linewidth=1.4)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('PR (OvR)')
    ax.grid(True, linestyle='--', alpha=0.5); ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out_prefix}_pr.png", bbox_inches='tight'); plt.close(fig)

def plot_top_features(feat_names, importances, out_png, topk=30, title='Feature importance'):
    order = np.argsort(importances)[::-1][:topk]
    fig, ax = plt.subplots(figsize=(8.6, max(4.2, topk*0.25)))
    ax.barh(range(len(order)), importances[order][::-1], align='center')
    ax.set_yticks(range(len(order))); ax.set_yticklabels([feat_names[i] for i in order][::-1], fontsize=8)
    ax.set_xlabel('Importance'); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)

# ------------- model & search spaces -------------
def add_prefix(step, grid: dict) -> dict:
    return {f"{step}__{k}": v for k, v in grid.items()}

def filter_grid(estimator, grid: dict) -> list[dict]:
    """过滤掉 estimator.get_params() 里不存在的键；为空→[{}]"""
    valid = set(estimator.get_params().keys())
    g = {k: v for k, v in grid.items() if k in valid}
    return [g] if g else [{}]

def build_spaces(args):
    spaces = []

    # ---- RandomForest (Randomized) ----
    rf = Pipeline([('clf', RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=args.random_state
    ))])
    rf_grid_raw = {
        "n_estimators": [i for i in range(5, 2001, 5)],
        "max_depth": [None] + list(range(5, 301, 5)),
        "min_samples_split": [2, 3, 4, 5, 6, 7,9,11,13,15],
        "max_features": ["sqrt", "log2", None],
        "min_samples_leaf": [1, 2, 3,4,5,6,7,8,9,10]
    }
    rf_grid = filter_grid(rf, add_prefix("clf", rf_grid_raw))
    spaces.append(("rf", rf, rf_grid, "random"))

    # ---- SVM (Grid) ----
    svm = Pipeline([('scaler', StandardScaler()),
                    ('clf', SVC(probability=True, class_weight='balanced', random_state=args.random_state))])
    svm_grid_raw = {
        "C": [10**p for p in [-2,-1.5,-1,-0.5,0,0.5,1,1.5]],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]+ [10**p for p in [-4,-3,-2,-1,0]]
    }
    svm_grid = filter_grid(svm, add_prefix("clf", svm_grid_raw))
    spaces.append(("svm", svm, svm_grid, "grid"))

    # ---- KNN (Grid) ----
    knn = Pipeline([('scaler', StandardScaler()),
                    ('clf', KNeighborsClassifier())])
    knn_grid_raw = {
        "n_neighbors": [i for i in range(3, 1001, 2)],
        "p": [1, 2],
        "weights": ["uniform", "distance"]
    }
    knn_grid = filter_grid(knn, add_prefix("clf", knn_grid_raw))
    spaces.append(("knn", knn, knn_grid, "grid"))

    # ---- Logistic Regression (Grid) ----
    lr = Pipeline([('scaler', StandardScaler()),
                   ('clf', LogisticRegression(max_iter=2000, solver="lbfgs",
                                              class_weight="balanced",
                                              multi_class="auto",
                                              random_state=args.random_state))])
    lr_grid_raw = {"C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                   "penalty": ["l2","l1"],
                  
                 }
    lr_grid = filter_grid(lr, add_prefix("clf", lr_grid_raw))
    spaces.append(("logreg", lr, lr_grid, "grid"))

    # ---- XGBoost (Randomized, 可选) ----
    xgb_ok = False
    try:
        from xgboost import XGBClassifier  # noqa
        xgb_ok = True
    except Exception:
        xgb_ok = False

    if xgb_ok:
        from xgboost import XGBClassifier
        xgb_params_fixed = dict(
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method=("gpu_hist" if args.xgb_gpu else "hist"),
            predictor=("gpu_predictor" if args.xgb_gpu else "auto"),
            n_jobs=-1,
            random_state=args.random_state
        )
        xgb = Pipeline([('clf', XGBClassifier(**xgb_params_fixed))])
        xgb_grid_raw = {
            "n_estimators": [i for i in range(10, 1501, 5)],
            "max_depth": [i for i in range(3, 400)],
            "learning_rate": [round(i * 0.01, 4) for i in range(1, 101)],
            "subsample": [0.6,0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1,2, 3,4, 5,6, 7],
            "reg_alpha": [0, 0.001, 0.01, 0.1],
            "reg_lambda": [0.1,0.5, 1,2, 5, 10]
        }
        xgb_grid = filter_grid(xgb, add_prefix("clf", xgb_grid_raw))
        spaces.append(("xgb", xgb, xgb_grid, "random"))

    return spaces

# ------------------------ main ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='results/hsi/image_features.tsv', help='image_features.tsv 路径')
    ap.add_argument('--sep', default='\t', help='表格分隔符')
    ap.add_argument('--target', default='session_id', help='标签列（如 session_id/phase 等）')
    ap.add_argument('--outdir', default='results/hsi/ml')
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--cv-folds', type=int, default=5)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--random-iter', type=int, default=500, help='大模型 RandomizedSearch 的迭代数 (默认500，按需调大/小)')
    ap.add_argument('--scoring', default='f1_weighted', choices=['f1_weighted','f1_macro','accuracy'])
    ap.add_argument('--xgb-gpu', action='store_true', help='若安装了xgboost且节点有GPU，可加此开关启用 gpu_hist')
    ap.add_argument('--shap', action='store_true', help='尝试计算SHAP（树模型），可能较慢')
    args = ap.parse_args()

    out = ensure_dir(args.outdir)

    # reproducibility
    np.random.seed(args.random_state)

    # ---- load data ----
    df = pd.read_csv(args.images, sep=args.sep)
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c: c.lower() for c in df.columns}
    df.rename(columns=lower_map, inplace=True)

    if args.target not in df.columns:
        raise RuntimeError(f"目标列 {args.target} 不存在。可用列：{list(df.columns)[:20]} ...")

    feat_cols = pick_feature_columns(df)
    if not feat_cols:
        raise RuntimeError("未找到任何 R_* 光谱列；请确认 preprocess 已产出平铺光谱列。")

    y_raw, mask = safe_y(df, args.target)
    X = df.loc[mask, feat_cols].to_numpy(dtype=float)
    y_raw = y_raw.to_numpy(str)

    ok = np.isfinite(X).all(axis=1)
    X = X[ok]; y_raw = y_raw[ok]

    # label encode
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_.tolist()
    pd.Series(classes).to_csv(out/'classes.txt', index=False, header=False)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # CV config
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    # build model spaces
    spaces = build_spaces(args)

    best_model = None
    best_name = None
    best_test_score = -1
    summary_rows = []

    for name, pipe, grid, mode in spaces:
        print(f"==> Training {name} | mode={mode} | scoring={args.scoring}")

        if mode == "random":
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=grid[0] if grid and isinstance(grid, list) else {},
                n_iter=args.random_iter,
                cv=cv,
                scoring=args.scoring,
                n_jobs=args.n_jobs,
                random_state=args.random_state,
                refit=True,
                verbose=0,
                return_train_score=True
            )
        else:
            search = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                cv=cv,
                scoring=args.scoring,
                n_jobs=args.n_jobs,
                refit=True,
                verbose=0,
                return_train_score=True
            )

        search.fit(X_train, y_train)

        # save cv results
        cv_df = pd.DataFrame(search.cv_results_)
        cv_df.to_csv(out/f'cv_results_{name}.csv', index=False)

        # test set
        y_pred = search.best_estimator_.predict(X_test)
        report = classification_report(y_test, y_pred, labels=list(range(len(classes))),
                                       target_names=classes, output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(out/f'{name}_classification_report.csv')

        test_f1w = f1_score(y_test, y_pred, average='weighted')
        test_mac = f1_score(y_test, y_pred, average='macro')
        test_acc = accuracy_score(y_test, y_pred)

        with open(out/f"{name}_best_params.txt", "w") as f:
            f.write(f"[{name}] best_params: {search.best_params_}\n")
            f.write(f"[{name}] cv_best_score({args.scoring}) = {search.best_score_:.6f}\n")
            f.write(f"[{name}] test_f1_weighted = {test_f1w:.6f}, test_f1_macro = {test_mac:.6f}, test_acc = {test_acc:.6f}\n")

        # optional: plot n_estimators curve if present
        df_log = cv_df.copy()
        ne_col = None
        for cand in ["param_clf__n_estimators", "param_n_estimators"]:
            if cand in df_log.columns:
                ne_col = cand; break
        if ne_col is not None:
            try:
                xs = pd.to_numeric(df_log[ne_col], errors='coerce')
                ys = df_log["mean_test_score"]
                ok = xs.notna() & ys.notna()
                plt.figure()
                plt.plot(xs[ok], ys[ok], 'o-', linewidth=1.2)
                plt.xlabel("n_estimators"); plt.ylabel(f"CV {args.scoring}")
                plt.title(f"{name}: {args.scoring} vs n_estimators"); plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout(); plt.savefig(out/f"{name}_F1_vs_n_estimators.png"); plt.close()
            except Exception:
                pass

        # feature importance
        feat_imp_done = False
        base_est = search.best_estimator_
        if hasattr(base_est, "steps"):
            # 取管道最后一步
            try:
                base_est = dict(base_est.steps).get("clf", base_est)
            except Exception:
                pass

        if hasattr(base_est, "feature_importances_"):
            imp = np.asarray(base_est.feature_importances_, float)
            pd.DataFrame({"feature": feat_cols, "importance": imp}).sort_values("importance", ascending=False)\
              .to_csv(out/f"{name}_feature_importance.csv", index=False)
            plot_top_features(feat_cols, imp, out/f"{name}_feature_importance_top30.png",
                              topk=30, title=f"Feature importance ({name})")
            feat_imp_done = True

        # SHAP（可选且仅树模型）
        if args.shap and (hasattr(base_est, "feature_importances_")):
            try:
                import shap
                explainer = shap.TreeExplainer(base_est)
                samp = min(100, X_test.shape[0])
                shap_values = explainer.shap_values(X_test[:samp])
                shap.summary_plot(shap_values, X_test[:samp], feature_names=feat_cols, show=False)
                plt.tight_layout(); plt.savefig(out/f"{name}_SHAP_summary.png"); plt.close()
            except Exception as e:
                with open(out/f"{name}_shap_error.txt","w") as f:
                    f.write(str(e))

        # plots: confusion, pca, prob curves
        labels_idx = list(range(len(classes)))
        plot_confusion_matrix(y_test, y_pred, labels=labels_idx, out_png=out/f'{name}_confusion_matrix_norm.png', normalize=True)
        plot_pca_scatter(X_test, np.array([classes[i] for i in y_test]), out/f'{name}_pca_scatter.png', title=f'PCA (test) - {name}')
        try_plot_roc_pr(search.best_estimator_, X_test, y_test, classes=labels_idx, out_prefix=str(out/f'{name}_prob'))

        # choose overall best by test f1_weighted（与你示例一致）
        choose_score = test_f1w
        summary_rows.append({"model": name,
                             "cv_best_score": search.best_score_,
                             "test_f1_weighted": test_f1w,
                             "test_f1_macro": test_mac,
                             "test_accuracy": test_acc})
        if choose_score > best_test_score:
            best_test_score = choose_score
            best_model = search.best_estimator_
            best_name = name

    # save best
    import joblib
    joblib.dump(best_model, out/'best_model.joblib')
    with open(out/'best_model.txt','w',encoding='utf-8') as f:
        f.write(f"best_model={best_name}, test_{args.scoring}={best_test_score:.6f}\n")

    # overall summary
    pd.DataFrame(summary_rows).sort_values("test_f1_weighted", ascending=False)\
      .to_csv(out/'model_summary.csv', index=False)

    print(f"[OK] ML pipeline finished. Best={best_name} ({args.scoring}={best_test_score:.4f}) -> {out.resolve()}")

if __name__ == '__main__':
    main()

