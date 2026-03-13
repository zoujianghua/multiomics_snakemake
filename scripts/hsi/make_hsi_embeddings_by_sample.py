#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 patch 级 CNN embedding 导出 sample 级 HSI 表征：

输入：
- patch 级 embedding 表（例如：
  - results/hsi/patch/2dcnn_phase_core/patch_embeddings_2d.tsv
  - results/hsi/patch/3dcnn_phase_core/patch_embeddings_3d.tsv
- image 级元数据表：
  - results/hsi/image_features.tsv

输出：
- results/hsi/hsi_embedding_by_sample.tsv
  每行一个 sample（source_sample_id / sample_id），包含：
    - source_sample_id / sample_id
    - phase / phase_core / temp / time / physiological_state（如有）
    - emb2d_* / emb3d_* 等聚合后的 embedding 特征
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_and_aggregate_patch_emb(emb_path: Path, prefix: str) -> pd.DataFrame:
    """
    读取 patch_embeddings.tsv，并按 source_sample_id 聚合为 sample 级 embedding。
    聚合方式：对每个 sample_id 的所有 patch embedding 取均值。
    """
    print(f"[embed] 读取 patch embedding: {emb_path}")
    df = pd.read_csv(emb_path, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]

    if "source_sample_id" not in df.columns:
        raise RuntimeError(
            f"{emb_path} 缺少 source_sample_id 列，当前列: {list(df.columns)[:10]}"
        )

    # 选择 embedding 列
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise RuntimeError(f"{emb_path} 中未找到 emb_* 列")

    # 按 source_sample_id 聚合（均值）
    grouped = (
        df.groupby("source_sample_id")[emb_cols]
        .mean()
        .reset_index()
        .rename(columns={"source_sample_id": "sample_id"})
    )

    # 加上前缀以区分 2D / 3D
    rename_map = {c: f"{prefix}{c.split('emb_')[-1]}" for c in emb_cols}
    grouped = grouped.rename(columns=rename_map)

    print(
        f"[embed] {emb_path.name}: n_sample={len(grouped)}, "
        f"n_emb={len(rename_map)} (prefix={prefix})"
    )
    return grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--emb-2d",
        default=None,
        help="2D-CNN patch embedding 表路径（可选），如 "
        "results/hsi/patch/2dcnn_phase_core/patch_embeddings_2d.tsv",
    )
    ap.add_argument(
        "--emb-3d",
        default=None,
        help="3D-CNN patch embedding 表路径（可选），如 "
        "results/hsi/patch/3dcnn_phase_core/patch_embeddings_3d.tsv",
    )
    ap.add_argument(
        "--image-meta",
        required=True,
        help="image 级元数据表，如 results/hsi/image_features.tsv "
        "（需至少包含 sample_id / phase / phase_core / temp / time）",
    )
    ap.add_argument(
        "--out",
        default="results/hsi/hsi_embedding_by_sample.tsv",
        help="输出文件路径（TSV），默认 results/hsi/hsi_embedding_by_sample.tsv",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取 image 级元数据
    img_df = pd.read_csv(args.image_meta, sep="\t")
    img_df.columns = [c.strip().lower() for c in img_df.columns]
    if "sample_id" not in img_df.columns:
        raise RuntimeError(
            f"image_meta 缺少 sample_id 列，当前列: {list(img_df.columns)[:10]}"
        )

    meta_cols = [
        c
        for c in ["sample_id", "phase", "phase_core", "temp", "time", "physiological_state"]
        if c in img_df.columns
    ]
    img_meta = img_df[meta_cols].drop_duplicates(subset=["sample_id"])
    print(f"[embed] image_meta: n_sample={len(img_meta)}")

    # 聚合 2D / 3D embedding
    dfs = [img_meta]

    if args.emb_2d:
        emb2d_df = load_and_aggregate_patch_emb(Path(args.emb_2d), prefix="emb2d_")
        dfs.append(emb2d_df)

    if args.emb_3d:
        emb3d_df = load_and_aggregate_patch_emb(Path(args.emb_3d), prefix="emb3d_")
        dfs.append(emb3d_df)

    if len(dfs) == 1:
        raise RuntimeError("未提供任何 embedding 表（--emb-2d / --emb-3d 均为空）")

    # 依次左连接，按 sample_id 对齐
    merged = dfs[0]
    for extra in dfs[1:]:
        merged = merged.merge(extra, on="sample_id", how="left")

    merged.to_csv(out_path, sep="\t", index=False)
    print(
        f"[embed] 已写出 sample 级 HSI embedding 表: {out_path} "
        f"(n_sample={len(merged)}, n_cols={len(merged.columns)})"
    )


if __name__ == "__main__":
    main()

