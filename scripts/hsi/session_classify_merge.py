#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
汇总多种 session 分类模型的结果，并画模型对比图。

输入：若干 *_summary.csv，每个通常是 1 行，列至少包含：
  - model
  - cv_best_score
  - test_f1_weighted
  - test_f1_macro
  - test_accuracy
  - best_params

输出：
  - all_models_summary.csv
  - all_models_f1_bar.png
  - all_models_accuracy_bar.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pretty(ax):
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summaries', nargs='+', required=True,
                    help='各模型的 *_summary.csv 列表')
    ap.add_argument('--outdir', default='results/hsi/ml')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    for p_str in args.summaries:
        p = Path(p_str)
        if not p.exists():
            print(f"[WARN] summary not found: {p}")
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}")
            continue

        if df.empty:
            continue

        # 尝试补全 model 列
        if 'model' not in df.columns:
            model_name = p.stem.replace('_summary', '')
            df['model'] = model_name

        df['source'] = str(p)
        frames.append(df)

    if not frames:
        print("[MERGE] no valid summary files, nothing to do.")
        # 仍然写空表，避免 Snakemake 报错
        pd.DataFrame().to_csv(outdir / 'all_models_summary.csv', index=False)
        return

    all_df = pd.concat(frames, ignore_index=True)

    # 如果有重复 model，取 test_f1_weighted 最好的那行
    if 'test_f1_weighted' in all_df.columns:
        all_df = (
            all_df.sort_values('test_f1_weighted', ascending=False)
                  .groupby('model', as_index=False)
                  .first()
        )
    else:
        all_df = all_df.groupby('model', as_index=False).first()

    all_df.to_csv(outdir / 'all_models_summary.csv', index=False)
    print(f"[MERGE] merged summary saved to {outdir / 'all_models_summary.csv'}")

    # ---- 画 F1_weighted 对比 ----
    if 'test_f1_weighted' in all_df.columns:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        pretty(ax)
        models = all_df['model'].astype(str).tolist()
        f1w = all_df['test_f1_weighted'].to_numpy(float)
        xs = np.arange(len(models))
        ax.bar(xs, f1w, width=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(models, rotation=20, ha='right')
        ax.set_ylabel('Test F1 (weighted)')
        ax.set_title('Model comparison — F1_weighted')
        for i, v in enumerate(f1w):
            ax.text(i, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / 'all_models_f1_bar.png', bbox_inches='tight')
        plt.close(fig)
    else:
        print("[MERGE] no column 'test_f1_weighted'; skip F1 plot.")

    # ---- 画 Accuracy 对比 ----
    if 'test_accuracy' in all_df.columns:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        pretty(ax)
        models = all_df['model'].astype(str).tolist()
        acc = all_df['test_accuracy'].to_numpy(float)
        xs = np.arange(len(models))
        ax.bar(xs, acc, width=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(models, rotation=20, ha='right')
        ax.set_ylabel('Test accuracy')
        ax.set_title('Model comparison — accuracy')
        for i, v in enumerate(acc):
            ax.text(i, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / 'all_models_accuracy_bar.png', bbox_inches='tight')
        plt.close(fig)
    else:
        print("[MERGE] no column 'test_accuracy'; skip accuracy plot.")

    print("[MERGE] done.")


if __name__ == '__main__':
    main()

