#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSI preprocess（leaf 级，一连通域 ≈ 一片叶）

- 输入：
    * samples_hsi.csv（和 preprocess.py 一样）
    * preprocess.py 生成的 cube/{sample_id}.npz
      其中包含：wavelength, R（反射率 cube，裁边后）, mask（image 级 ROI）

- 对每个 sample_id：
    1) 读取 cube npz → R, wavelength, mask
    2) 在 mask 内做连通域分割（label）
    3) 对每个连通域（候选叶片）按：
         - leaf_min_area
         - leaf_aspect_ratio_max
         - leaf_ndvi_mean_min
       进行过滤
    4) 对合格叶片的 mask 提取 ROI 光谱 → spec/spec_sg
    5) 按 leaf_id 写 spec_leaf/{leaf_id}.npz + raw_leaf_features.tsv（一行一叶）

- 输出：
    * spec_leaf/*.npz
    * raw_leaf_features.tsv（后续可走 clean_* / add_indices 一模一样的流程）
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from skimage.measure import label, regionprops
from concurrent.futures import ProcessPoolExecutor  # 新增：多进程


def nearest_band(wls, target_nm: float) -> int:
    return int(np.argmin(np.abs(wls - target_nm)))


def extract_roi_spectrum(R, mask, wls, sg_window=11, sg_poly=2, use_snv=False):
    """和 preprocess.py 同名函数保持一致：对 mask 内像素求平均光谱。"""
    pix = R[mask]
    if pix.size == 0:
        return None, None

    spec = np.nanmean(pix, axis=0)

    spec_work = spec.copy()
    if use_snv:
        mu = spec_work.mean()
        sd = spec_work.std() + 1e-8
        spec_work = (spec_work - mu) / sd

    spec_sg = savgol_filter(
        spec_work,
        sg_window,
        sg_poly,
        mode='interp'
    ) if sg_window > 2 else spec_work

    return spec, spec_sg


def _process_one_sample(payload):
    """
    多进程 worker：处理单个 sample，返回多行 leaf 级 row（可能为 0~N 行）。
    payload = (row_dict, cube_dir_str, spec_dir_str, args)
    """
    row_dict, cube_dir_str, spec_dir_str, args = payload
    r = row_dict
    sid = r['sample_id']

    cube_dir = Path(cube_dir_str)
    spec_dir = Path(spec_dir_str)

    cube_npz = cube_dir / f"{sid}.npz"
    if not cube_npz.exists():
        print(f"[leaf] skip {sid}: cube npz not found -> {cube_npz}")
        return []

    try:
        data = np.load(cube_npz, allow_pickle=True)
    except Exception as e:
        print(f"[leaf] error {sid}: failed to load cube npz -> {e}")
        return []

    R = data['R']                # (H, W, B) 反射率
    wls = data['wavelength']     # (B,)
    mask = data['mask'].astype(bool)  # image 级 ROI mask

    # 基于 R 重新算一遍 NDVI（leaf 过滤用）
    try:
        b670 = nearest_band(wls, 670)
        b800 = nearest_band(wls, 800)
        R670 = R[:, :, b670]
        R800 = R[:, :, b800]
        ndvi_map = (R800 - R670) / (R800 + R670 + 1e-9)
    except Exception:
        print(f"[leaf] warn {sid}: NDVI bands not found, skip")
        return []

    # 连通域：一连通域 ≈ 一片叶
    lab = label(mask)
    props = list(regionprops(lab))
    if not props:
        print(f"[leaf] warn {sid}: empty mask")
        return []

    rows = []

    for reg in props:
        area = reg.area
        if area < args.leaf_min_area:
            continue

        h = reg.bbox[2] - reg.bbox[0]
        w = reg.bbox[3] - reg.bbox[1]
        ar = max(h, w) / (min(h, w) + 1e-6)
        if ar > args.leaf_aspect_ratio_max:
            # 过细长，更多是茎或异常形状，丢弃
            continue

        coords = reg.coords  # (Npix, 2)
        ndvi_leaf = ndvi_map[coords[:, 0], coords[:, 1]]
        ndvi_mean = float(np.nanmean(ndvi_leaf))
        if ndvi_mean < args.leaf_ndvi_mean_min:
            # 整片的 NDVI 偏低，更像老叶/暗茎
            continue

        leaf_mask = (lab == reg.label)

        spec, spec_sg = extract_roi_spectrum(
            R,
            leaf_mask,
            wls,
            sg_window=args.sg_window,
            sg_poly=args.sg_poly,
            use_snv=args.use_snv,
        )
        if spec is None:
            continue

        leaf_id = f"{sid}_leaf{reg.label}"
        leaf_npz = spec_dir / f"{leaf_id}.npz"
        np.savez_compressed(
            leaf_npz,
            wavelength=wls,
            spec=spec,
            spec_sg=spec_sg,
        )

        # 叶片级 R800 中位数（可用于后续 clean_leaf_features）
        try:
            R800_leaf = R800[leaf_mask]
            R800_med_leaf = float(np.nanmedian(R800_leaf))
        except Exception:
            R800_med_leaf = np.nan

        # 元数据沿用 sample 级
        phase_val = str(r.get('phase', '')).strip()
        phase_core = str(r.get('phase_core', '')).strip()
        temp_val = str(r.get('temp', '')).strip()
        time_val = str(r.get('time', '')).strip()
        time_h = r.get('time_h', np.nan)
        rep_val = str(r.get('replicate', '')).strip()

        row = {
            'sample_id': sid,
            'leaf_id': leaf_id,
            'leaf_label': int(reg.label),   # 在该图中的连通域编号
            'temp': temp_val,
            'time': time_val,
            'time_h': time_h,
            'phase': phase_val,
            'phase_core': phase_core,
            'replicate': rep_val,
            'roi_area': int(area),
            'leaf_bbox_y0': int(reg.bbox[0]),
            'leaf_bbox_x0': int(reg.bbox[1]),
            'leaf_bbox_y1': int(reg.bbox[2]),
            'leaf_bbox_x1': int(reg.bbox[3]),
            'leaf_ndvi_mean': ndvi_mean,
            'spec_npz': str(leaf_npz),
            'cube_npz': str(cube_npz),
            'R800_med_leaf': R800_med_leaf,
        }

        # 展开 spec_sg 为 R_<nm> 列
        def wl_colname(w):
            return "R_" + np.format_float_positional(float(w), trim='-')

        for wl, val in zip(wls, spec_sg):
            col = wl_colname(wl)
            row[col] = float(val)

        rows.append(row)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True, help='samples_hsi.csv')
    ap.add_argument('--cube-dir', required=True,
                    help='preprocess.py 生成的 cube 目录，例如 results/hsi/cube')
    ap.add_argument('--outdir', required=True,
                    help='leaf 级输出目录，例如 results/hsi_leaf')
    ap.add_argument('--sg-window', type=int, default=11)
    ap.add_argument('--sg-poly', type=int, default=2)
    ap.add_argument('--use-snv', action='store_true')

    # 叶片级过滤阈值
    ap.add_argument('--leaf-min-area', type=int, default=800,
                    help='叶片连通域的最小面积（像素）')
    ap.add_argument('--leaf-aspect-ratio-max', type=float, default=8.0,
                    help='叶片 bbox 的最大纵横比（>该值视为过细长，丢弃）')
    ap.add_argument('--leaf-ndvi-mean-min', type=float, default=0.2,
                    help='叶片内 NDVI 均值的最小阈值，用于去掉暗茎/老叶')

    # 新增：workers（多进程进程数）
    ap.add_argument('--workers', type=int, default=1,
                    help='并行进程数（<= 申请的 threads；1 表示不开并行）')

    args = ap.parse_args()

    out = Path(args.outdir)
    spec_dir = out / 'spec_leaf'
    spec_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.samples)
    cube_dir = Path(args.cube_dir)

    # 准备每个 sample 的 payload
    tasks = []
    for _, r in df.iterrows():
        tasks.append((r.to_dict(), str(cube_dir), str(spec_dir), args))

    all_rows = []

    if args.workers <= 1:
        print(f"[leaf] running sequentially on {len(tasks)} samples")
        for payload in tasks:
            rows = _process_one_sample(payload)
            if rows:
                all_rows.extend(rows)
    else:
        print(f"[leaf] running with ProcessPool: workers={args.workers}, nsamples={len(tasks)}")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for rows in ex.map(_process_one_sample, tasks):
                if rows:
                    all_rows.extend(rows)

    leaf_df = pd.DataFrame(all_rows)
    out_tsv = out / 'raw_leaf_features.tsv'
    leaf_df.to_csv(out_tsv, sep='\t', index=False)
    print(f"[leaf] wrote {len(leaf_df)} rows -> {out_tsv}")


if __name__ == '__main__':
    main()

