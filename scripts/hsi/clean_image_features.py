#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【HSI patch pipeline - 通用异常值清洗】

对 raw_image_features.tsv / raw_leaf_features.tsv / raw_patch_features_*.tsv
做类内异常样本剔除，输出 clean_*.tsv。

- 全局硬过滤：
  * use_raw_mode == 1 的样本剔除（反射率标定太差，只能用原始强度分割）
  * R800_med < min_r800 的样本剔除（NIR 太暗，类似黑图）

- 组内光谱离群过滤（可配置分组列）：
  * 在每个 group 内，取所有光谱列 R_<nm>，按波段计算 median + MAD
  * 对每个样本在每个波段计算 robust z-score：
      z = 0.6745 * (x - median) / MAD
  * 对每个样本取 z_abs_max = max_b |z_{b}|
  * 若 z_abs_max > max_z，则该样本标记为离群样本

- 新增：
  * group-col 支持：
      * 若显式指定 --group-col，则按该列分组；
      * 若未指定，则按 phase > phase_core 的优先级自动选择；
      * 若两者都不存在，则所有样本视为一个整体组。
  * 生成离群样本详情表（可选），记录每个样本：
      group_col/group_value, sample_id(若存在), 行索引, z_max,
      max_band, max_band_z, exceed_bands(>max_z 的波段列表)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images-raw",
        required=True,
        help="raw_*_features.tsv，包含 R_<nm> 光谱列",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="clean_*_features.tsv 输出路径",
    )
    ap.add_argument(
        "--min-r800",
        type=float,
        default=0.03,
        help="R800_med 低于该值直接剔除",
    )
    ap.add_argument(
        "--max-z",
        type=float,
        default=4.5,
        help="robust z-score 的阈值，建议 4~5",
    )
    ap.add_argument(
        "--outliers-report",
        default=None,
        help=(
            "可选：离群样本详情输出路径（TSV）。"
            "若不指定，则使用 out 路径改名为 *_outliers.tsv"
        ),
    )
    ap.add_argument(
        "--group-col",
        default=None,
        help=(
            "按哪个列做组内离群检测；"
            "若未指定，则按 phase > phase_core 的优先级自动选择；"
            "若都不存在，则所有样本视为一个整体组。"
        ),
    )
    args = ap.parse_args()

    path_in = Path(args.images_raw)
    df = pd.read_csv(path_in, sep="\t")

    # 找光谱列
    spec_cols = [c for c in df.columns if c.startswith("R_")]
    if not spec_cols:
        raise RuntimeError("未找到任何 R_<nm> 列，无法根据光谱做异常检测。")

    # -------- 决定 group-col --------
    group_col = args.group_col
    if group_col is not None:
        # 显式指定
        if group_col not in df.columns:
            raise RuntimeError(
                f"指定的 group-col='{group_col}' 在输入表中不存在。"
                f" 可选列包括：{', '.join(df.columns)}"
            )
    else:
        # 自动选择：phase > phase_core > 不分组
        if "phase" in df.columns:
            group_col = "phase"
        elif "phase_core" in df.columns:
            group_col = "phase_core"
        else:
            group_col = None

    # 初始全部保留
    keep = np.ones(len(df), dtype=bool)

    # 全局硬过滤：raw-mode 和 R800_med
    if "use_raw_mode" in df.columns:
        keep &= (df["use_raw_mode"].fillna(0).astype(int) == 0)
    if "R800_med" in df.columns:
        keep &= df["R800_med"].fillna(0) >= args.min_r800

    df["__keep__"] = keep

    # 用来记录所有离群样本的详细信息
    outlier_records = []

    # -------- 分组并做 robust z-score 离群检测 --------
    if group_col is None:
        # 不分组：所有样本作为一个整体
        groups = [("ALL", df.index)]
        print(
            "[clean] WARNING: 未找到 phase / phase_core / group-col，"
            "所有样本视为一个整体做离群检测。"
        )
    else:
        groups = list(df.groupby(group_col).groups.items())
        print(f"[clean] using group-col='{group_col}', n_groups={len(groups)}")

    for g_val, idx in groups:
        sub = df.loc[idx, spec_cols]
        arr = sub.to_numpy(dtype=float)

        # 样本太少时跳过（比如 < 5 行）
        if arr.shape[0] < 5:
            continue

        # 对每个波段计算 median + MAD（忽略 NaN）
        med = np.nanmedian(arr, axis=0)
        mad = np.nanmedian(np.abs(arr - med), axis=0)
        mad[mad < 1e-9] = 1e-9

        z = 0.6745 * (arr - med) / mad
        z_abs = np.abs(z)
        z_max = np.nanmax(z_abs, axis=1)

        bad = z_max > args.max_z
        if bad.any():
            bad_idx = sub.index[bad]
            n_bad = int(bad.sum())
            print(
                f"[clean] group {group_col or 'ALL'}={g_val}: "
                f"remove {n_bad} / {len(sub)} outliers"
            )

            # 逐个离群样本记录“是谁 + 哪些波段出问题”
            for pos, row_idx in enumerate(sub.index):
                if not bad[pos]:
                    continue

                row_z = z_abs[pos]  # 这一行各波段的 |z|
                # 最大 z 以及对应波段
                max_z_val = float(np.nanmax(row_z))
                max_band_idx = int(np.nanargmax(row_z))
                max_band_name = spec_cols[max_band_idx]
                max_band_z = float(row_z[max_band_idx])

                # 所有 |z| > max_z 的波段
                exceed_mask = row_z > args.max_z
                exceed_bands = [
                    spec_cols[i]
                    for i, flag in enumerate(exceed_mask)
                    if flag
                ]

                rec = {
                    "row_index": int(row_idx),
                    "z_max": max_z_val,
                    "max_band": max_band_name,
                    "max_band_z": max_band_z,
                    "exceed_bands": ",".join(exceed_bands),
                }

                # 记录分组信息
                if group_col is not None:
                    rec[group_col] = g_val

                # 如果有 sample_id / session_id 等，顺手带上
                if "sample_id" in df.columns:
                    rec["sample_id"] = df.at[row_idx, "sample_id"]
                if "session_id" in df.columns:
                    rec["session_id"] = df.at[row_idx, "session_id"]

                outlier_records.append(rec)

            # 标记这些行为不保留
            df.loc[bad_idx, "__keep__"] = False

    # 输出清洗后的表
    cleaned = df[df["__keep__"]].drop(columns=["__keep__"])
    path_out = Path(args.out)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(path_out, sep="\t", index=False)

    print(
        f"[clean] from {len(df)} -> {len(cleaned)} rows, "
        f"removed {len(df) - len(cleaned)} outliers -> {path_out}"
    )

    # 若有离群样本，写详情表
    if outlier_records:
        if args.outliers_report is None:
            # 默认在 out 同一目录下生成 *_outliers.tsv
            path_rep = path_out.with_name(path_out.stem + "_outliers.tsv")
        else:
            path_rep = Path(args.outliers_report)

        path_rep.parent.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame(outlier_records)

        # 为了好看，列顺序稍微调一下
        col_order = []
        # 分组列（如果有）
        if group_col is not None and group_col in out_df.columns:
            col_order.append(group_col)
        # 其它常见字段
        for c in [
            "sample_id",
            "session_id",
            "row_index",
            "z_max",
            "max_band",
            "max_band_z",
            "exceed_bands",
        ]:
            if c in out_df.columns and c not in col_order:
                col_order.append(c)
        # 剩余列
        other_cols = [c for c in out_df.columns if c not in col_order]
        out_df = out_df[col_order + other_cols]

        out_df.to_csv(path_rep, sep="\t", index=False)
        print(f"[clean] outlier details saved to {path_rep} ({len(out_df)} rows)")
    else:
        print("[clean] no outliers detected above max_z threshold.")


if __name__ == "__main__":
    main()

