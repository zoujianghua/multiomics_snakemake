# scripts/hsi_session_classify.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict
from collections import defaultdict

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.model_selection import StratifiedKFold


# ---------- I/O utils ----------
def load_npz_spectrum(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return (w, spec_med, spec_ctrl, spec_delta, keys)."""
    with np.load(str(npz_path)) as z:
        keys = list(z.keys())
        if   "wavelength"  in z: w = z["wavelength"]
        elif "wavelengths" in z: w = z["wavelengths"]
        elif "wl"          in z: w = z["wl"]
        else: raise ValueError(f"No wavelength key in {npz_path}, keys={keys}")

        if   "spec_med" in z: s = z["spec_med"]
        elif "spec"     in z: s = z["spec"]
        elif "median"   in z: s = z["median"]
        else: raise ValueError(f"No spectrum (spec_med/spec/median) in {npz_path}, keys={keys}")

        sc = z["spec_ctrl"]  if "spec_ctrl"  in z else None
        sd = z["spec_delta"] if "spec_delta" in z else None

        w  = np.asarray(w, dtype=float).ravel()
        s  = np.asarray(s, dtype=float).ravel()
        if s.shape[0] != w.shape[0]:
            raise ValueError(f"Len mismatch: w={w.shape[0]} spec={s.shape[0]} in {npz_path}")

        if sc is not None:
            sc = np.asarray(sc, dtype=float).ravel()
            if sc.shape[0] != w.shape[0]:
                sc = None
        if sd is not None:
            sd = np.asarray(sd, dtype=float).ravel()
            if sd.shape[0] != w.shape[0]:
                sd = None

        return w, s, sc, sd, keys


def parse_session_tokens(stem: str) -> Dict[str, str]:
    """
    Expect stems like:  '10_2h_stress' or '25_8d_control' or '25_1d_recovery'
    Return dict with 'temp','time','phase', plus 'session' (original).
    """
    toks = stem.split("_")
    out = {"session": stem, "temp": "", "time": "", "phase": ""}
    if len(toks) >= 3 and toks[0].isdigit():
        out["temp"]  = f"{toks[0]}C"
        out["time"]  = toks[1]
        out["phase"] = toks[2].lower()
    else:
        # fallback: try regex
        m = re.match(r"(?P<temp>\d+)[-_](?P<time>[^-_]+)[-_](?P<phase>[^-_]+)", stem)
        if m:
            out["temp"]  = f"{m.group('temp')}C"
            out["time"]  = m.group("time")
            out["phase"] = m.group("phase").lower()
        else:
            out["session"] = stem  # keep at least session name
    return out


def read_meta(meta_path: Path) -> pd.DataFrame:
    if not meta_path or not meta_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(meta_path, sep="\t")
    df.columns = [c.lower() for c in df.columns]
    for c in ("temp","time","phase"):
        if c not in df.columns:
            df[c] = ""
    return df


def to_hours(x) -> float:
    if x is None: return np.nan
    s = str(x).strip().lower()
    # general fuzzy parse: '2h', '1d', 't2h', 'day7', '7 d'
    s = re.sub(r'^(t|time|day)\s*', '', s)
    m = re.search(r'(\d+(?:\.\d+)?)\s*([hd])', s)
    if m:
        v = float(m.group(1)); u = m.group(2)
        return v if u=='h' else v*24.0
    # defaults
    table = {"2h":2, "6h":6, "1d":24, "3d":72, "7d":168, "8d":192, "10d":240, "14d":336}
    return float(table.get(s, np.nan))


# ---------- preprocessing ----------
class SNVTransformer(BaseEstimator, TransformerMixin):
    """Standard Normal Variate per sample (row-wise z-normalization)."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        mu = X.mean(1, keepdims=True)
        sd = X.std(1, keepdims=True)
        sd[sd == 0] = 1.0
        return (X - mu) / sd


# ---------- core ----------
def build_dataset(npz_dir: Path, meta_path: Path=None, split_recovery: bool=True):
    """
    Returns:
      X: (n_samples, n_features)
      y: list[str] class labels
      wl: wavelength grid (n_features,)
      info_df: DataFrame with columns [session,temp,time,phase[,recovery_from]]
    """
    files = sorted(npz_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz under {npz_dir}")

    # read all
    entries = []
    wls = []
    for f in files:
        try:
            w, s, sc, sd, _ = load_npz_spectrum(f)
            entries.append({"path":f, "w":w, "s":s})
            wls.append(w)
        except Exception as e:
            print(f"[warn] skip {f}: {e}", file=sys.stderr)

    if not entries:
        raise RuntimeError("No valid session spectra")

    # choose common wavelength axis (use first file) and interpolate others if needed
    wl0 = entries[0]["w"]
    feats = []
    rows  = []
    for e in entries:
        s = e["s"]
        if len(e["w"]) != len(wl0) or np.max(np.abs(e["w"] - wl0)) > 1e-6:
            s = np.interp(wl0, e["w"], s, left=s[0], right=s[-1])
        stem = e["path"].stem
        meta = parse_session_tokens(stem)
        feats.append(s.astype(float))
        rows.append(meta)

    X = np.vstack(feats)
    info_df = pd.DataFrame(rows)

    # attach recovery_from from meta tsv (if available)
    if meta_path and meta_path.exists():
        meta_df = read_meta(meta_path)
        on_cols = ["temp","time","phase"]
        info_df = info_df.merge(meta_df[on_cols + ([c for c in ["recovery_from"] if c in meta_df.columns])],
                                on=on_cols, how="left")

    # generate labels
    def make_label(r):
        temp  = str(r.get("temp",""))
        phase = str(r.get("phase","")).lower()
        if "control" in phase or "ctrl" in phase:
            return f"{temp}-control"
        if "stress" in phase:
            return f"{temp}-stress"
        if "reco" in phase:
            if split_recovery and str(r.get("recovery_from","")):
                src = str(r["recovery_from"])
                return f"Rec-from-{src}"
            return "25C-recovery"
        # fallback
        return f"{temp}-{phase or 'unknown'}"

    y = [make_label(r) for r in info_df.to_dict(orient="records")]
    wl = wl0.copy()
    return X, y, wl, info_df


def kfold_scores(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1u = f1_score(y_true, y_pred, average="micro")
    return acc, f1m, f1u


def run_cv(models, X, y, n_splits=5, random_state=42):
    labels = np.array(y)
    classes, counts = np.unique(labels, return_counts=True)
    minc = counts.min()
    if minc < 2:
        raise RuntimeError(f"Some class has <2 samples (min={minc}). Not enough for CV.")
    n_splits = min(n_splits, minc) if minc < n_splits else n_splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = {name: {"fold": [], "cm": np.zeros((len(classes), len(classes)), dtype=float)} for name in models}
    all_preds = []

    for fold, (tr, te) in enumerate(skf.split(X, labels), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = labels[tr], labels[te]
        for name, pipe in models.items():
            mdl = pipe.fit(Xtr, ytr)
            yp  = mdl.predict(Xte)
            acc, f1m, f1u = kfold_scores(yte, yp)
            results[name]["fold"].append({"fold":fold, "acc":acc, "f1_macro":f1m, "f1_micro":f1u})
            # confusion
            cm = confusion_matrix(yte, yp, labels=classes)
            results[name]["cm"] += cm
            # collect predictions
            row = pd.DataFrame({"session_idx": te,
                                "y_true": yte,
                                f"y_pred_{name}": yp})
            # proba if available
            if hasattr(mdl, "predict_proba"):
                proba = mdl.predict_proba(Xte)
                for j, c in enumerate(mdl.classes_):
                    row[f"proba_{name}_{c}"] = proba[:,j]
            all_preds.append(row)

    # aggregate
    summary = {}
    for name, rec in results.items():
        df = pd.DataFrame(rec["fold"])
        summary[name] = {
            "n_splits": n_splits,
            "accuracy_mean": float(df["acc"].mean()),
            "f1_macro_mean": float(df["f1_macro"].mean()),
            "f1_micro_mean": float(df["f1_micro"].mean()),
            "per_fold": df.to_dict(orient="records")
        }
    preds = None
    if all_preds:
        preds = all_preds[0]
        for k in range(1, len(all_preds)):
            preds = preds.merge(all_preds[k], how="outer", on=["session_idx","y_true"])
        preds.sort_values("session_idx", inplace=True)

    return classes.tolist(), results, summary, preds


def pick_best_model(summary: dict) -> str:
    # choose by macro-F1, then accuracy
    best = None; best_key = None
    for name, s in summary.items():
        key = (s["f1_macro_mean"], s["accuracy_mean"])
        if best is None or key > best:
            best = key; best_key = name
    return best_key or list(summary.keys())[0]


def plot_confusion(cm, classes, title, out_png):
    if cm.sum() == 0:
        plt.figure(figsize=(5,4)); plt.axis("off")
        plt.text(0.1, 0.5, "empty confusion", fontsize=12)
        plt.savefig(out_png, dpi=150); plt.close(); return
    fig, ax = plt.subplots(figsize=(max(5, 0.5*len(classes)+2), max(4, 0.4*len(classes)+2)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    # annotate
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm[i,j]
            ax.text(j, i, int(v), ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close(fig)


def save_importances(models, best_name, best_model, wl, classes, out_tsv):
    """
    Save feature importances for interpretation:
      - For logistic regression: per-class coefficients (one-vs-rest).
      - For random forest: global feature_importances_.
    """
    df = pd.DataFrame({"wavelength_nm": wl})
    if isinstance(best_model, LogisticRegression):
        coef = best_model.coef_  # (n_classes, n_features)
        for i, c in enumerate(best_model.classes_):
            df[f"logreg_coef_{c}"] = coef[i]
    elif isinstance(best_model, RandomForestClassifier):
        imp = best_model.feature_importances_
        df["rf_importance"] = imp
    else:
        # try both if present
        lr = None; rf = None
        for m in models.values():
            if isinstance(m, LogisticRegression): lr = m
            if isinstance(m, RandomForestClassifier): rf = m
        if lr is not None:
            for i, c in enumerate(lr.classes_):
                df[f"logreg_coef_{c}"] = lr.coef_[i]
        if rf is not None:
            df["rf_importance"] = rf.feature_importances_
    df.to_csv(out_tsv, sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True, help="Directory of session_spectra *.npz")
    ap.add_argument("--meta", default="", help="Optional TSV (delta_hsi.tsv or session_features.tsv)")
    ap.add_argument("--outdir", required=True, help="Output directory for ML results")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--no-snv", action="store_true", help="Disable SNV per spectrum")
    ap.add_argument("--no-std", action="store_true", help="Disable StandardScaler (feature-wise)")
    ap.add_argument("--split-recovery", action="store_true", default=True,
                    help="Split recovery by 'recovery_from' if available")
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    outdir  = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    meta    = Path(args.meta) if args.meta else None

    X, y, wl, info = build_dataset(npz_dir, meta, split_recovery=args.split_recovery)
    n, p = X.shape
    classes = sorted(set(y))
    print(f"[data] samples={n} features={p} classes={classes}")

    # build pipelines
    steps = []
    if not args.no-snv:
        steps.append(("snv", SNVTransformer()))
    if not args.no_std:
        steps.append(("std", StandardScaler(with_mean=True, with_std=True)))

    # define two models for comparison
    logreg = LogisticRegression(max_iter=2000, n_jobs=1, multi_class="ovr", C=1.0)
    rf     = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=1, random_state=42)

    # Wrap manually (simple)
    models = {
        "logreg": logreg,
        "rf": rf
    }

    # CV loop with preproc each fold
    def fit_with_preproc(mdl, Xtr, ytr):
        Xt = Xtr.copy()
        for name, step in steps:
            Xt = step.fit_transform(Xt, ytr)
        model = mdl.fit(Xt, ytr)
        return model, [s for _, s in steps]

    def predict_with_preproc(model, preproc_steps, Xte):
        Xt = Xte.copy()
        for step in preproc_steps:
            Xt = step.transform(Xt)
        return model.predict(Xt), (model.predict_proba(Xt) if hasattr(model, "predict_proba") else None)

    # custom CV to apply our preproc pipeline
    labels = np.array(y)
    uniq, cnts = np.unique(labels, return_counts=True)
    minc = cnts.min()
    if minc < 2:
        raise RuntimeError("Not enough samples per class for CV.")
    splits = min(args.splits, minc) if minc < args.splits else args.splits
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    metrics = {}
    cms = {name: np.zeros((len(uniq), len(uniq)), dtype=float) for name in models}
    pred_rows = []

    for fold, (tr, te) in enumerate(skf.split(X, labels), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = labels[tr], labels[te]
        for name, mdl in models.items():
            # fit
            fitted, preproc_steps = fit_with_preproc(mdl, Xtr, ytr)
            # predict
            yp = predict_with_preproc(fitted, preproc_steps, Xte)[0]
            # scores
            acc = accuracy_score(yte, yp)
            f1m = f1_score(yte, yp, average="macro")
            f1u = f1_score(yte, yp, average="micro")
            metrics.setdefault(name, []).append({"fold":fold, "acc":acc, "f1_macro":f1m, "f1_micro":f1u})
            cms[name] += confusion_matrix(yte, yp, labels=uniq)
            # store preds
            pred = pd.DataFrame({"session_idx": te, "y_true": yte, f"y_pred_{name}": yp})
            pred_rows.append(pred)

    # summarize metrics
    summary = {}
    for name, recs in metrics.items():
        df = pd.DataFrame(recs)
        summary[name] = {
            "n_splits": splits,
            "accuracy_mean": float(df["acc"].mean()),
            "f1_macro_mean": float(df["f1_macro"].mean()),
            "f1_micro_mean": float(df["f1_micro"].mean()),
            "per_fold": df.to_dict(orient="records")
        }

    # choose best
    best = max(summary.items(), key=lambda kv: (kv[1]["f1_macro_mean"], kv[1]["accuracy_mean"]))[0]
    print(f"[best] {best}  f1_macro={summary[best]['f1_macro_mean']:.3f}  acc={summary[best]['accuracy_mean']:.3f}")

    # refit best on full data for importances
    # (fit preproc on full data as well)
    Xt = X.copy()
    fitted_steps = []
    for _, step in steps:
        Xt = step.fit_transform(Xt, labels)
        fitted_steps.append(step)
    if best == "logreg":
        best_model = LogisticRegression(max_iter=2000, n_jobs=1, multi_class="ovr", C=1.0).fit(Xt, labels)
    else:
        best_model = RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=42).fit(Xt, labels)

    # save outputs
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    with open(outdir / "metrics.json", "w") as f:
        json.dump({
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "classes": uniq.tolist(),
            "cv_summary": summary,
            "best_model": best
        }, f, indent=2)

    # confusion matrix of best
    plot_confusion(cms[best], uniq.tolist(), f"Confusion ({best})", outdir / "confusion_matrix.png")

    # feature importances / coefficients
    # Try to extract from best_model; also try both models if attribute missing
    try:
        save_importances(
            {"logreg": best_model if isinstance(best_model, LogisticRegression) else LogisticRegression().fit(Xt, labels),
             "rf": best_model if isinstance(best_model, RandomForestClassifier) else RandomForestClassifier(n_estimators=10).fit(Xt, labels)},
            best, best_model, wl, uniq.tolist(), outdir / "feature_importance.tsv"
        )
    except Exception as e:
        # fallback: just dump wavelengths
        pd.DataFrame({"wavelength_nm": wl}).to_csv(outdir / "feature_importance.tsv", sep="\t", index=False)

    # predictions table
    if pred_rows:
        preds = pred_rows[0]
        for k in range(1, len(pred_rows)):
            preds = preds.merge(pred_rows[k], how="outer", on=["session_idx","y_true"])
        preds = preds.sort_values("session_idx").reset_index(drop=True)
        # attach session name and meta
        info2 = info.reset_index().rename(columns={"index":"session_idx"})
        preds = preds.merge(info2, on="session_idx", how="left")
        preds.to_csv(outdir / "cv_predictions.tsv", sep="\t", index=False)

    print("[OK] hsi session classification done.")


if __name__ == "__main__":
    main()

