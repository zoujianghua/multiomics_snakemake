#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【HSI patch pipeline - Patch 特征提取】

从 patch_index_{target}_seed{seed}.tsv 提取 patch 级光谱特征，生成 raw_patch_features_{target}.tsv。
用于传统 ML 模型（RF/SVM/XGB 等），深度学习模型直接使用 patch_cubes。

功能：
- 从 patch_index_*.tsv 生成 raw_patch_features_{target}.tsv：
- 输入：patch_index_{target}_seed42.tsv
  需包含列：
    patch_id, sample_id, cube_npz, target, y0, x0, size
- 输出：results/hsi/raw_patch_features_{target}.tsv
  每行一个 patch，包含：
    sample_id (= patch_id),
    source_sample_id,
    <target_col>（例如 phase_core 或 metabo_state），
    roi_area, cube_npz, spec_npz, R800_med,
    以及 SG 平滑后的 R_<nm> 光谱列

本版本支持按 cube_npz 分组的多进程处理：
- 每个 cube 只加载一次，再在内存里循环切 patch；
- 主进程统一合并各 worker 的结果后写出 TSV。
"""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def nearest_band(wls, target_nm: float) -> int:
    return int(np.argmin(np.abs(wls - target_nm)))


def extract_roi_spectrum(R, mask, wls, sg_window=11, sg_poly=2, use_snv=False):
    """
    对 mask 内像素求平均光谱；可选 SNV；最后做 SG 滤波。
    返回：spec（原始均值），spec_sg（平滑后）。
    """
    pix = R[mask]
    if pix.size == 0:
        return None, None

    spec = np.nanmean(pix, axis=0)

    spec_work = spec.copy()
    if use_snv:
        mu = spec_work.mean()
        sd = spec_work.std() + 1e-8
        spec_work = (spec_work - mu) / sd

    # 确保窗口合法且为奇数
    win = min(sg_window, len(wls) - (1 - len(wls) % 2))
    if win < 3:
        spec_sg = spec_work
    else:
        if win % 2 == 0:
            win -= 1
        spec_sg = savgol_filter(
            spec_work,
            win,
            sg_poly,
            mode="interp",
        )

    return spec, spec_sg


def _process_one_cube(payload):
    """
    多进程 worker：处理同一个 cube_npz 或 cube_patch_npz 下的一批 patch。

    payload = (
        cube_npz_path_str,  # 对于旧格式是 cube_npz，对于 patch_cubes 模式是 cube_patch_npz
        records,            # list[dict]，来自该 cube 的所有 patch 行
        spec_dir_str,
        target_col,
        sg_window,
        sg_poly,
        use_snv,
        use_patch_cubes,    # bool，是否使用 patch_cubes 模式
    )

    返回：list[dict]，每个 dict 对应一行输出。
    """
    (
        cube_npz_path_str,
        records,
        spec_dir_str,
        target_col,
        sg_window,
        sg_poly,
        use_snv,
        use_patch_cubes,
    ) = payload

    cube_path = Path(cube_npz_path_str)
    spec_dir = Path(spec_dir_str)

    rows_out = []

    if not cube_path.exists():
        print(f"[patch] skip cube: not found -> {cube_path}")
        return rows_out

    if use_patch_cubes:
        # patch_cubes 模式：从 cube_patch_npz 加载已切好的 patches
        try:
            patch_cubes_data = np.load(cube_path, allow_pickle=True, mmap_mode="r")
        except Exception as e:
            print(f"[patch] error cube {cube_path}: fail load cube_patch_npz -> {e}")
            return rows_out

        if "patches" not in patch_cubes_data:
            print(f"[patch] cube {cube_path}: cube_patch_npz 缺少 patches")
            return rows_out

        patches_array = patch_cubes_data["patches"]  # (N, B, H, W)
        coords_array = patch_cubes_data.get("coords", None)  # (N, 3) 或 None
        wls = patch_cubes_data.get("wavelength", None)

        if wls is None or len(wls) == 0:
            print(f"[patch] cube {cube_path}: 缺少 wavelength，尝试从原始 cube_npz 获取")
            # 尝试从第一个 record 中获取原始 cube_npz 路径
            if records and "cube_npz" in records[0]:
                try:
                    orig_cube = np.load(records[0]["cube_npz"], allow_pickle=True)
                    wls = orig_cube.get("wavelength", None)
                    if wls is None:
                        print(f"[patch] cube {cube_path}: 无法获取 wavelength")
                        return rows_out
                except Exception:
                    print(f"[patch] cube {cube_path}: 无法从原始 cube_npz 获取 wavelength")
                    return rows_out
            else:
                print(f"[patch] cube {cube_path}: 无法获取 wavelength")
                return rows_out

        wls = np.asarray(wls)
        N_patches, B, H_patch, W_patch = patches_array.shape

        # 预先找好 800 nm 波段索引（用于 R800_med）
        try:
            b800 = nearest_band(wls, 800.0)
        except Exception:
            b800 = None

        # 需要从原始 cube_npz 获取 mask（patch_cubes 文件不包含 mask）
        # 从第一个 record 获取原始 cube_npz 路径
        orig_cube_npz = None
        R_full = None
        mask_full = None
        if records and "cube_npz" in records[0]:
            orig_cube_npz = Path(records[0]["cube_npz"])
            if orig_cube_npz.exists():
                try:
                    orig_data = np.load(orig_cube_npz, allow_pickle=True)
                    R_full = orig_data.get("R", None)  # (H, W, B)
                    mask_full = orig_data.get("mask", None)
                    if mask_full is not None:
                        mask_full = mask_full.astype(bool)
                except Exception as e:
                    print(f"[patch] cube {cube_path}: 无法加载原始 cube_npz {orig_cube_npz} -> {e}")

        for r in records:
            patch_id = str(r["patch_id"])
            source_sample_id = str(r["sample_id"])
            target_val = str(r["target"])
            patch_idx = int(r["patch_idx"])

            if patch_idx >= N_patches:
                print(f"[patch] {patch_id}: patch_idx {patch_idx} >= {N_patches}")
                continue

            # 从 patches_array 中取出 patch: (B, H, W)
            patch_R_band_first = patches_array[patch_idx]  # (B, H, W)
            # 转换为 (H, W, B) 格式以匹配后续处理
            patch_R = np.moveaxis(patch_R_band_first, 0, -1)  # (H, W, B)

            # 获取坐标信息（用于 mask 裁剪）
            if coords_array is not None:
                y0, x0, size = coords_array[patch_idx]
            else:
                # 如果没有 coords，尝试从 record 中获取
                y0 = int(r.get("y0", 0))
                x0 = int(r.get("x0", 0))
                size = int(r.get("size", H_patch))

            # 从原始 cube_npz 的 mask 中裁剪对应的 patch mask
            if mask_full is not None and R_full is not None:
                H_full, W_full = mask_full.shape[:2]
                if 0 <= y0 < H_full and 0 <= x0 < W_full:
                    y_end = min(y0 + size, H_full)
                    x_end = min(x0 + size, W_full)
                    patch_mask = mask_full[y0:y_end, x0:x_end]
                    # 如果 patch_mask 尺寸与 patch_R 不匹配，调整
                    if patch_mask.shape[:2] != patch_R.shape[:2]:
                        patch_mask = patch_mask[:patch_R.shape[0], :patch_R.shape[1]]
                else:
                    # 坐标超出范围，使用全 True mask（表示整个 patch 都有效）
                    patch_mask = np.ones((patch_R.shape[0], patch_R.shape[1]), dtype=bool)
            else:
                # 无法获取 mask，使用全 True mask（表示整个 patch 都有效）
                patch_mask = np.ones((patch_R.shape[0], patch_R.shape[1]), dtype=bool)

            # 后续处理：提取光谱、保存等（与旧格式共享）
            if not patch_mask.any():
                print(f"[patch] {patch_id}: empty patch mask, skip")
                continue

            spec, spec_sg = extract_roi_spectrum(
                patch_R,
                patch_mask,
                wls,
                sg_window=sg_window,
                sg_poly=sg_poly,
                use_snv=use_snv,
            )
            if spec is None:
                print(f"[patch] {patch_id}: spec is None, skip")
                continue

            # 计算 patch 内 R800 中位数
            if b800 is not None:
                try:
                    R800_patch = patch_R[:, :, b800][patch_mask]
                    R800_med = float(np.nanmedian(R800_patch))
                except Exception:
                    R800_med = float("nan")
            else:
                R800_med = float("nan")

            # 保存 patch spec npz
            patch_npz = spec_dir / f"{patch_id}.npz"
            np.savez_compressed(
                patch_npz,
                wavelength=wls,
                spec=spec,
                spec_sg=spec_sg,
            )

            # 获取原始 cube_npz 路径（用于输出）
            orig_cube_npz_str = str(orig_cube_npz) if orig_cube_npz is not None else str(cube_path)

            row = {
                # 为了兼容 clean_image_features.py，这里直接把 patch_id 填到 sample_id
                "sample_id": patch_id,
                "source_sample_id": source_sample_id,
                target_col: target_val,   # 标签列名由参数控制
                "roi_area": int(patch_mask.sum()),
                "cube_npz": orig_cube_npz_str,
                "spec_npz": str(patch_npz),
                "patch_y0": y0,
                "patch_x0": x0,
                "patch_size": size,
                "R800_med": R800_med,
            }

            # 展开 spec_sg -> R_<nm>
            def wl_colname(w):
                return "R_" + np.format_float_positional(float(w), trim="-")

            for wl, val in zip(wls, spec_sg):
                col = wl_colname(wl)
                row[col] = float(val)

            rows_out.append(row)

    else:
        # 旧格式：从 cube_npz 中按 y0/x0/size 切 patch
        try:
            data = np.load(cube_path, allow_pickle=True)
        except Exception as e:
            print(f"[patch] error cube {cube_path}: fail load cube_npz -> {e}")
            return rows_out

        if "R" not in data or "wavelength" not in data or "mask" not in data:
            print(f"[patch] cube {cube_path}: cube_npz 缺少 R/wavelength/mask")
            return rows_out

        R = data["R"]             # (H, W, B)
        wls = data["wavelength"]  # (B,)
        mask = data["mask"].astype(bool)

        H, W, B = R.shape
        if mask.shape[:2] != (H, W):
            print(f"[patch] cube {cube_path}: mask shape mismatch, cube={R.shape}, mask={mask.shape}")
            return rows_out

        # 预先找好 800 nm 波段索引（用于 R800_med）
        try:
            b800 = nearest_band(wls, 800.0)
        except Exception:
            b800 = None

        for r in records:
            patch_id = str(r["patch_id"])
            source_sample_id = str(r["sample_id"])
            target_val = str(r["target"])
            y0 = int(r["y0"])
            x0 = int(r["x0"])
            size = int(r["size"])

            if not (0 <= y0 < H and 0 <= x0 < W):
                print(f"[patch] {patch_id}: patch origin out of range")
                continue
            if y0 + size > H or x0 + size > W:
                print(f"[patch] {patch_id}: patch exceeds boundary")
                continue

            patch_R = R[y0:y0 + size, x0:x0 + size, :]
            patch_mask = mask[y0:y0 + size, x0:x0 + size]

            # 后续处理：提取光谱、保存等（与 patch_cubes 模式共享）
            if not patch_mask.any():
                print(f"[patch] {patch_id}: empty patch mask, skip")
                continue

            spec, spec_sg = extract_roi_spectrum(
                patch_R,
                patch_mask,
                wls,
                sg_window=sg_window,
                sg_poly=sg_poly,
                use_snv=use_snv,
            )
            if spec is None:
                print(f"[patch] {patch_id}: spec is None, skip")
                continue

            # 计算 patch 内 R800 中位数
            if b800 is not None:
                try:
                    R800_patch = patch_R[:, :, b800][patch_mask]
                    R800_med = float(np.nanmedian(R800_patch))
                except Exception:
                    R800_med = float("nan")
            else:
                R800_med = float("nan")

            # 保存 patch spec npz
            patch_npz = spec_dir / f"{patch_id}.npz"
            np.savez_compressed(
                patch_npz,
                wavelength=wls,
                spec=spec,
                spec_sg=spec_sg,
            )

            row = {
                # 为了兼容 clean_image_features.py，这里直接把 patch_id 填到 sample_id
                "sample_id": patch_id,
                "source_sample_id": source_sample_id,
                target_col: target_val,   # 标签列名由参数控制
                "roi_area": int(patch_mask.sum()),
                "cube_npz": str(cube_path),
                "spec_npz": str(patch_npz),
                "patch_y0": y0,
                "patch_x0": x0,
                "patch_size": size,
                "R800_med": R800_med,
            }

            # 展开 spec_sg -> R_<nm>
            def wl_colname(w):
                return "R_" + np.format_float_positional(float(w), trim="-")

            for wl, val in zip(wls, spec_sg):
                col = wl_colname(wl)
                row[col] = float(val)

            rows_out.append(row)

    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True,
                    help="results/hsi/ml/patch_index_<target>_seed42.tsv")
    ap.add_argument("--outdir", required=True,
                    help="results/hsi")
    ap.add_argument("--out-tsv", default=None,
                    help="可选，指定 raw_patch_features 输出路径；未指定时用 outdir/raw_patch_features_{target_col}.tsv")
    ap.add_argument("--target-col", required=True,
                    help="输出特征表中用于分类的标签列名，例如 phase_core 或 metabo_state")
    ap.add_argument("--sg-window", type=int, default=9)
    ap.add_argument("--sg-poly", type=int, default=2)
    ap.add_argument("--use-snv", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="多进程 workers 数量（1 表示单进程顺序处理）")
    args = ap.parse_args()

    index_df = pd.read_csv(args.index, sep="\t")
    index_df.columns = [c.strip().lower() for c in index_df.columns]

    # 检查索引格式：优先使用 patch_cubes 模式（cube_patch_npz + patch_idx）
    # 兼容旧格式（cube_npz + y0/x0/size）
    use_patch_cubes = "cube_patch_npz" in index_df.columns and "patch_idx" in index_df.columns
    use_old_format = "cube_npz" in index_df.columns and "y0" in index_df.columns and "x0" in index_df.columns and "size" in index_df.columns
    
    if use_patch_cubes:
        print(f"[patch] 使用 patch_cubes 模式（推荐，I/O 效率更高）")
        required_cols = ["patch_id", "sample_id", "cube_patch_npz", "patch_idx", "target"]
        for c in required_cols:
            if c not in index_df.columns:
                raise RuntimeError(f"patch_index 缺少列: {c}（patch_cubes 模式）")
    elif use_old_format:
        print(f"[patch] 使用旧格式（cube_npz + y0/x0/size，兼容模式）")
        required_cols = ["patch_id", "sample_id", "cube_npz", "target", "y0", "x0", "size"]
        for c in required_cols:
            if c not in index_df.columns:
                raise RuntimeError(f"patch_index 缺少列: {c}（旧格式）")
    else:
        raise RuntimeError(
            "patch_index 必须包含以下列之一：\n"
            "  - cube_patch_npz + patch_idx（推荐，patch_cubes 模式）\n"
            "  - cube_npz + y0 + x0 + size（兼容旧格式）"
        )

    out_root = Path(args.outdir)
    # 不同 target 用不同的 spec 子目录，避免覆盖
    spec_dir = out_root / f"spec_patch_{args.target_col}"
    spec_dir.mkdir(parents=True, exist_ok=True)

    # 按 cube_patch_npz 或 cube_npz 分组
    if use_patch_cubes:
        group_col = "cube_patch_npz"
    else:
        group_col = "cube_npz"
    
    groups = []
    for cube_path, sub_df in index_df.groupby(group_col):
        records = sub_df.to_dict(orient="records")
        groups.append(
            (
                str(cube_path),
                records,
                str(spec_dir),
                args.target_col,
                args.sg_window,
                args.sg_poly,
                args.use_snv,
                use_patch_cubes,  # 添加标志位
            )
        )

    print(f"[patch] total cubes={len(groups)}, total patches={len(index_df)}, "
          f"workers={args.workers}")

    all_rows = []

    if args.workers <= 1:
        # 单进程顺序处理
        for payload in groups:
            rows = _process_one_cube(payload)
            if rows:
                all_rows.extend(rows)
    else:
        # 多进程
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for rows in ex.map(_process_one_cube, groups):
                if rows:
                    all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_tsv = Path(args.out_tsv) if args.out_tsv else (out_root / f"raw_patch_features_{args.target_col}.tsv")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"[patch] wrote {len(df)} rows -> {out_tsv}")


if __name__ == "__main__":
    main()

