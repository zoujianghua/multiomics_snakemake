#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSI preprocess（精细 phase 版，仅做反射率 + 分割 + ROI 光谱）

- 反射率标定：优先固定参考（白/黑板中央带中位数），否则本地参考；极端时可 raw-mode 做分割
- 叶片分割：四方案 A/B/C/D（NDRE+NDVI / NIR+NDVI / KMeans / NDVI top-k），记录 seg_plan 与 roi_area
- ROI 光谱：对 mask 内像素求均值光谱，按需 SNV + SG 平滑；只展开为 R_<nm> 列，不在此计算任何植被指数
- phase 列：直接是精细标签，例如：
    - control_25_T2h
    - control_25_T8d
    - stress_35_T3d
    - recovery_from_35_T6h
  方便后续按 phase 直接做分类 / 清洗
- 额外字段：temp（数值字符串）、time（'2h'/'3d'）、time_h（小时）、replicate（重复号）
- 输出：
    - spec/*.npz：每样本 wavelength/spec/spec_sg（image 级）
    - cube/*.npz：每样本 wavelength/R/mask（后续 leaf / patch 使用）
    - raw_image_features.tsv：一行一个样本，含元数据 + 精细 phase + 光谱列 + R800_med 等质量指标
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import spectral as sp
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation

from concurrent.futures import ProcessPoolExecutor


# ======================== 基础工具 ========================

def load_cube(hdr_path: Path):
    """读取 ENVI 立方和 wavelength；若波长解析失败，返回空数组以便上游跳过。"""
    ds = sp.envi.open(str(hdr_path))
    img = ds.load().astype(np.float32)
    md = ds.metadata
    wls_raw = md.get('wavelength', [])
    try:
        wls = np.array([float(x) for x in wls_raw], dtype=np.float32)
    except Exception:
        wls = np.array([], dtype=np.float32)
    return img, wls


def nearest_band(wls, target_nm: float) -> int:
    """找到最接近 target_nm 的波段索引。"""
    return int(np.argmin(np.abs(wls - target_nm)))


def pseudo_rgb(R, wls):
    """用 ~700/550/450 nm 合成伪彩，做 2-98 分位缩放提升对比度。"""
    bR, bG, bB = nearest_band(wls, 700), nearest_band(wls, 550), nearest_band(wls, 450)
    rgb = np.stack([R[:, :, bR], R[:, :, bG], R[:, :, bB],], axis=-1)
    lo = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
    return rgb


def save_mask_and_overlay(mask, R, wls, out_mask_png: Path, out_overlay_png: Path):
    """保存二值 mask 以及叠加红色边缘的伪彩图，用于质检。"""
    out_mask_png.parent.mkdir(parents=True, exist_ok=True)
    out_overlay_png.parent.mkdir(parents=True, exist_ok=True)

    plt.imsave(out_mask_png, mask.astype(float), cmap='gray', vmin=0, vmax=1)

    rgb = pseudo_rgb(R, wls)
    edge = binary_dilation(mask, iterations=1) ^ mask
    overlay = rgb.copy()
    overlay[edge] = [1.0, 0.0, 0.0]
    overlay[~mask] = overlay[~mask] * 0.4
    plt.imsave(out_overlay_png, overlay)


# ======================== 参考谱 + 反射率 ========================

def _central_crop(img, frac=0.6):
    """取图像中央 frac 的矩形区域（避开边缘/遮挡）。"""
    H, W = img.shape[:2]
    fh = max(1, int(H * frac))
    fw = max(1, int(W * frac))
    top = (H - fh) // 2
    left = (W - fw) // 2
    return img[top:top + fh, left:left + fw, :]


def _bandwise_median(img):
    """把 3D 图像拉平成 (Npix, B) 后，对每个波段取中位数。"""
    x = img.reshape(-1, img.shape[-1]).astype(np.float32)
    return np.nanmedian(x, axis=0)


def _make_ref_spectrum(white_cube, dark_cube, center_frac=0.6):
    """对白/黑板中央带分别取每波段中位数，构造分母（加稳健下限防止除 0）。"""
    Wc = _central_crop(white_cube, center_frac)
    Dc = _central_crop(dark_cube, center_frac)
    W_spec = _bandwise_median(Wc)
    D_spec = _bandwise_median(Dc)
    denom = W_spec - D_spec
    eps = max(1e-6, float(np.nanpercentile(np.abs(denom), 10)))
    denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps, denom)
    return W_spec, D_spec, denom


def reflectance_broadcast(sample, W_spec, D_spec, denom):
    """用 1D 参考谱对整幅样本做广播校正：R = (S - D)/(W - D)。"""
    B = min(sample.shape[-1], len(denom))
    S = sample[..., :B].astype(np.float32)
    R = (S - D_spec[:B].reshape(1, 1, B)) / (denom[:B].reshape(1, 1, B))
    return np.clip(R, 0, 1.5)


def reflectance_pixel(sample, white, dark):
    """像素级校正：当固定参考不可用且本地白/黑尺寸匹配时使用。"""
    H = min(sample.shape[0], white.shape[0], dark.shape[0])
    W = min(sample.shape[1], white.shape[1], dark.shape[1])
    sample, white, dark = sample[:H, :W, :], white[:H, :W, :], dark[:H, :W, :]
    R = (sample - dark) / (white - dark + 1e-6)
    return np.clip(R, 0, 1.5)


# ======================== 分割（四方案） ========================

def _clean_mask(mask, min_area=700, aspect_ratio_max=15, rel_area_min_frac=0.0):
    """
    形态学开闭 + 去小连通域 + 去细长目标 + （可选）按相对面积进一步过滤。

    注意：aspect_ratio_max 是对“每个连通域”的 bbox 计算的，而不是整个 mask。
    rel_area_min_frac > 0 时：只保留面积 >= rel_area_min_frac * 最大连通域面积 的区域。
    """
    # 先做形态学平滑 + 绝对面积过滤
    mask = binary_opening(mask, footprint=disk(2))
    mask = binary_closing(mask, footprint=disk(3))
    mask = remove_small_objects(mask, min_size=min_area)

    lab = label(mask)
    props = regionprops(lab)
    if not props:
        return mask

    # 最大连通域面积
    max_area = max(r.area for r in props)

    keep = np.zeros_like(mask, dtype=bool)
    for r in props:
        area = r.area
        h = r.bbox[2] - r.bbox[0]
        w = r.bbox[3] - r.bbox[1]
        ar = max(h, w) / (min(h, w) + 1e-6)

        # 去掉太细长的
        if ar > aspect_ratio_max:
            continue

        # 按相对面积过滤：比如 rel_area_min_frac=0.3，则小于最大面积 30% 的都扔掉
        if rel_area_min_frac > 0.0 and area < rel_area_min_frac * max_area:
            continue

        keep[lab == r.label] = True

    return keep


def build_leaf_mask(
    R,
    wls,
    min_area=700,
    min_ndvi=0.10,
    min_ndre=0.02,
    aspect_ratio_max=15,
    rel_area_min_frac=0.0,
):
    """
    四方案分割：A(NDRE+NDVI+Otsu)；B(NIR+NDVI)；C(KMeans NIR)；D(NDVI top-k 兜底)。
    """
    def nb(nm): return int(np.argmin(np.abs(wls - nm)))

    b670, b705, b740, b800 = nb(670), nb(705), nb(740), nb(800)
    R670, R705, R740, R800 = R[:, :, b670], R[:, :, b705], R[:, :, b740], R[:, :, b800]

    ndvi = (R800 - R670) / (R800 + R670 + 1e-9)
    ndre = (R740 - R705) / (R740 + R705 + 1e-9)

    # 方案 A：NDRE + NDVI + Otsu
    try:
        t_ndre = threshold_otsu(ndre[np.isfinite(ndre)])
    except Exception:
        t_ndre = np.nanpercentile(ndre, 70)
    mA = (ndre >= max(t_ndre, min_ndre)) & (ndvi >= min_ndvi)
    mA = _clean_mask(
        mA,
        min_area=min_area,
        aspect_ratio_max=aspect_ratio_max,
        rel_area_min_frac=rel_area_min_frac,
    )
    if mA.sum() >= min_area:
        return mA, {
            'plan': 'A',
            't_ndre': float(t_ndre),
            'ndvi_med': float(np.nanmedian(ndvi)),
            'area': int(mA.sum()),
        }

    # 方案 B：NIR + NDVI 宽松门限
    nir_thr = max(0.08, float(np.nanpercentile(R800, 60)) * 0.5)
    red_thr = float(np.nanpercentile(R670, 95))
    mB = (R800 >= nir_thr) & (R670 <= red_thr) & (ndvi >= min(min_ndvi, 0.05))
    mB = _clean_mask(
        mB,
        min_area=min_area,
        aspect_ratio_max=aspect_ratio_max,
        rel_area_min_frac=rel_area_min_frac,
    )
    if mB.sum() >= min_area:
        return mB, {
            'plan': 'B',
            't_ndre': float(t_ndre),
            'ndvi_med': float(np.nanmedian(ndvi)),
            'area': int(mB.sum()),
        }

    # 方案 C：KMeans 基于 [700,750,800] 三通道
    try:
        b700, b750 = nb(700), nb(750)
        X = np.stack(
            [R[:, :, b700].ravel(), R[:, :, b750].ravel(), R[:, :, b800].ravel()],
            axis=1
        )
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        km = KMeans(n_clusters=2, n_init=5, random_state=0).fit(X)
        lab2 = km.labels_.reshape(R.shape[:2])
        m0 = np.nanmean(R800[lab2 == 0])
        m1 = np.nanmean(R800[lab2 == 1])
        mC = (lab2 == (0 if m0 >= m1 else 1)) & (ndvi >= 0.0)
        mC = _clean_mask(
            mC,
            min_area=min_area,
            aspect_ratio_max=aspect_ratio_max,
            rel_area_min_frac=rel_area_min_frac,
        )
        if mC.sum() >= min_area:
            return mC, {
                'plan': 'C',
                't_ndre': float(t_ndre),
                'ndvi_med': float(np.nanmedian(ndvi)),
                'area': int(mC.sum()),
            }
    except Exception:
        pass

    # 方案 D：NDVI top-k 兜底（取前 0.3% 或 min_area）
    ndvi_flat = ndvi.ravel()
    k = max(int(0.003 * ndvi_flat.size), min_area)
    thr = np.partition(ndvi_flat, -k)[-k]
    mD = ndvi >= max(thr, 0.0)
    mD = _clean_mask(
        mD,
        min_area=min_area,
        aspect_ratio_max=aspect_ratio_max,
        rel_area_min_frac=rel_area_min_frac,
    )
    return mD, {
        'plan': 'D',
        't_ndre': float(t_ndre),
        'ndvi_med': float(np.nanmedian(ndvi)),
        'area': int(mD.sum()),
    }


# ======================== ROI 光谱（不算指数） ========================

def extract_roi_spectrum(R, mask, wls, sg_window=11, sg_poly=2, use_snv=False):
    """
    对 mask 内像素求平均光谱；按需做 SNV；再 SG 平滑。
    这里只返回 spec / spec_sg，不计算任何指数。
    """
    pix = R[mask]
    if pix.size == 0:
        return None, None

    spec = np.nanmean(pix, axis=0)  # 原始 ROI 均值光谱

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
    多进程 worker：处理单个样本。
    payload = (row_dict, outdir_str, args, fixed_refs)
    返回：row 字典，或者 None（表示跳过）
    """
    row_dict, outdir_str, args, fixed_refs = payload
    out = Path(outdir_str)
    r = row_dict  # 保持原来的 r[...] / r.get(...) 写法

    sid = r['sample_id']
    sh = Path(str(r['sample_hdr']))
    wh = Path(str(r.get('white_hdr', ''))) if str(r.get('white_hdr', '')) else None
    dh = Path(str(r.get('dark_hdr', ''))) if str(r.get('dark_hdr', '')) else None

    if not sh.exists():
        print(f"[skip] {sid}: missing sample_hdr")
        return None

    # 读样本立方 & 波长
    sample, wls = load_cube(sh)
    if wls is None or len(wls) == 0:
        print(f"[warn] {sid}: empty wavelength list, skip")
        return None

    # 反射率：优先固定参考；否则本地白/黑；再不行用原始
    if fixed_refs is not None:
        W_spec, D_spec, denom = fixed_refs
        R = reflectance_broadcast(sample, W_spec, D_spec, denom)
    else:
        if (wh is not None and wh.exists()) and (dh is not None and dh.exists()):
            white, _ = load_cube(wh)
            dark, _ = load_cube(dh)
            try:
                W_spec, D_spec, denom = _make_ref_spectrum(
                    white, dark, center_frac=args.ref_center_frac
                )
                R = reflectance_broadcast(sample, W_spec, D_spec, denom)
            except Exception:
                R = reflectance_pixel(sample, white, dark)
        else:
            R = sample.copy()

    # 裁掉四周 5 像素边缘
    if R.shape[0] > 10 and R.shape[1] > 10:
        R = R[5:-5, 5:-5, :]
        sample_crop = sample[5:-5, 5:-5, :]
    else:
        sample_crop = sample

    # 判断 raw-mode
    use_raw = False
    try:
        med_R800 = float(np.nanmedian(R[:, :, nearest_band(wls, 800)]))
    except Exception:
        med_R800 = 0.0
    if (not np.isfinite(R).any()) or med_R800 < 0.03:
        use_raw = args.raw_mode

    # 选择用于分割的立方
    CubeForSeg = R.copy()
    if use_raw:
        S = sample_crop.copy()
        q1 = np.nanpercentile(S, 5, axis=(0, 1), keepdims=True)
        q9 = np.nanpercentile(S, 95, axis=(0, 1), keepdims=True)
        CubeForSeg = np.clip((S - q1) / (q9 - q1 + 1e-6), 0, 1)

    # 叶片分割
    mask, mstat = build_leaf_mask(
        CubeForSeg,
        wls,
        min_area=args.min_area,
        min_ndvi=args.min_ndvi,
        min_ndre=args.min_ndre,
        aspect_ratio_max=args.aspect_ratio_max,
        rel_area_min_frac=args.roi_min_area_frac,
    )

    # 叶片细化（NDVI 分位）
    try:
        b670 = nearest_band(wls, 670)
        b800 = nearest_band(wls, 800)
        R670 = CubeForSeg[:, :, b670]
        R800 = CubeForSeg[:, :, b800]
        ndvi_map = (R800 - R670) / (R800 + R670 + 1e-9)

        if (
            mask.any()
            and args.leaf_ndvi_quantile is not None
            and args.leaf_ndvi_quantile > 0.0
        ):
            q = float(np.nanpercentile(
                ndvi_map[mask],
                args.leaf_ndvi_quantile * 100.0,
            ))
            mask = mask & (ndvi_map >= q)
            mask = _clean_mask(
                mask,
                min_area=max(args.leaf_refine_min_area, args.min_area // 2),
                aspect_ratio_max=int(args.aspect_ratio_max),
                rel_area_min_frac=args.roi_min_area_frac,
            )
    except Exception:
        pass

    # 可选：再按 NDRE 分位细化
    try:
        b705 = nearest_band(wls, 705)
        b740 = nearest_band(wls, 740)
        R705 = CubeForSeg[:, :, b705]
        R740 = CubeForSeg[:, :, b740]
        ndre_map = (R740 - R705) / (R740 + R705 + 1e-9)
        if (
            mask.any()
            and args.leaf_ndre_quantile is not None
            and args.leaf_ndre_quantile > 0.0
        ):
            q_ndre = float(np.nanpercentile(
                ndre_map[mask],
                args.leaf_ndre_quantile * 100.0,
            ))
            mask = mask & (ndre_map >= q_ndre)
            mask = _clean_mask(
                mask,
                min_area=max(args.leaf_refine_min_area, args.min_area // 2),
                aspect_ratio_max=int(args.aspect_ratio_max),
                rel_area_min_frac=args.roi_min_area_frac,
            )
    except Exception:
        pass

    # 质检图
    mask_png = out / 'mask' / f'{sid}_mask.png'
    overlay_png = out / 'mask' / f'{sid}_overlay.png'
    save_mask_and_overlay(mask, CubeForSeg, wls, mask_png, overlay_png)

    # 保存 cube 中间文件
    cube_npz_path = out / 'cube' / f'{sid}.npz'
    np.savez_compressed(
        cube_npz_path,
        wavelength=wls,
        R=R,
        mask=mask.astype(np.uint8),
    )

    # 提取 image 级 ROI 光谱
    spec, spec_sg = extract_roi_spectrum(
        R,
        mask,
        wls,
        sg_window=args.sg_window,
        sg_poly=args.sg_poly,
        use_snv=args.use_snv,
    )
    if spec is None:
        print(f"[warn] {sid}: empty ROI, saved diagnostics -> {overlay_png}")
        return None

    # 光谱 npz（image 级）
    npz_path = out / 'spec' / f'{sid}.npz'
    np.savez_compressed(npz_path, wavelength=wls, spec=spec, spec_sg=spec_sg)

    # ===== 使用样本表中的 phase / temp / time / time_h / replicate =====
    phase_val = str(r.get('phase', '')).strip()
    phase_core = str(r.get('phase_core', '')).strip()
    temp_val = str(r.get('temp', '')).strip()
    time_val = str(r.get('time', '')).strip()
    time_h = r.get('time_h', np.nan)
    rep_val = str(r.get('replicate', '')).strip()

    row = {
        'sample_id': sid,
        'group_dir': r.get('group_dir', ''),
        'sample_dir': r.get('sample_dir', ''),
        'capture_dir': r.get('capture_dir', ''),
        'sample_hdr': str(sh),
        'white_hdr': str(wh) if wh else '',
        'dark_hdr': str(dh) if dh else '',
        'temp': temp_val,
        'time': time_val,
        'time_h': time_h,
        'phase': phase_val,
        'phase_core': phase_core,
        'replicate': rep_val,
        'roi_area': int(mstat['area']),
        'ndre_thr': float(mstat['t_ndre']),
        'seg_plan': str(mstat.get('plan', '')),
        'spec_npz': str(npz_path),
        'cube_npz': str(cube_npz_path),
        'mask_png': str(mask_png),
        'overlay_png': str(overlay_png),
        'use_raw_mode': int(use_raw),
        'R800_med': float(med_R800) if np.isfinite(med_R800) else np.nan,
    }

    def wl_colname(w):
        return "R_" + np.format_float_positional(float(w), trim='-')

    for wl, val in zip(wls, spec_sg):
        col = wl_colname(wl)
        row[col] = float(val)

    return row



# ======================== 主流程 ========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True, help='samples_hsi.csv')
    ap.add_argument('--outdir', required=True, help='results dir')
    ap.add_argument('--sg-window', type=int, default=11)
    ap.add_argument('--sg-poly', type=int, default=2)
    ap.add_argument('--min-area', type=int, default=800)
    ap.add_argument('--min-ndvi', type=float, default=0.2)
    ap.add_argument('--min-ndre', type=float, default=0.05)
    ap.add_argument('--aspect-ratio-max', type=float, default=8.0)
    # 新增：按相对面积过滤小连通域，比如 0.3 表示保留面积 ≥ 最大连通域 30% 的区域
    ap.add_argument(
        '--roi-min-area-frac',
        type=float,
        default=0.3,
        help='清理连通域时，仅保留面积 ≥ 该比例 * 最大连通域面积的区域（0 关闭）',
    )
    ap.add_argument(
        '--use-snv',
        action='store_true',
        help='对 ROI 均值先做 SNV 再 SG',
    )

    # 固定参考
    ap.add_argument('--fixed-white-hdr', type=str, default='', help='全局固定白板 hdr')
    ap.add_argument('--fixed-dark-hdr', type=str, default='', help='全局固定黑板 hdr')
    ap.add_argument('--ref-center-frac', type=float, default=0.6, help='参考图中央带比例')

    # 原始模式兜底
    ap.add_argument(
        '--raw-mode',
        action='store_true',
        help='当反射率极差时用原始稳健缩放来分割',
    )
    ap.add_argument(
        '--leaf-ndvi-quantile',
        type=float,
        default=0.0,
        help='在 mask 内仅保留 NDVI 高于此分位数的像素（0 关闭）',
    )
    ap.add_argument(
        '--leaf-refine-min-area',
        type=int,
        default=400,
        help='NDVI/NDRE 细化后的最小面积',
    )
    ap.add_argument(
        '--leaf-ndre-quantile',
        type=float,
        default=0.0,
        help='可选：再按 NDRE 分位细化（0 关闭）',
    )
    ap.add_argument(
        '--workers',
        type=int,
        default=1,
        help='并行进程数（<= 申请的 threads；1 表示不开并行）',
    )

    args = ap.parse_args()

    out = Path(args.outdir)
    (out / 'spec').mkdir(parents=True, exist_ok=True)
    (out / 'mask').mkdir(exist_ok=True)
    (out / 'cube').mkdir(exist_ok=True)  # 新增：保存 R cube + mask 的中间文件

    df = pd.read_csv(args.samples)

    # 预加载固定参考（原逻辑不变）
    fixed_refs = None
    if args.fixed_white_hdr and args.fixed_dark_hdr:
        wfix_img, _ = load_cube(Path(args.fixed_white_hdr))
        dfix_img, _ = load_cube(Path(args.fixed_dark_hdr))
        B = min(wfix_img.shape[-1], dfix_img.shape[-1])
        wfix_img = wfix_img[..., :B]
        dfix_img = dfix_img[..., :B]
        W_spec_fix, D_spec_fix, denom_fix = _make_ref_spectrum(
            wfix_img, dfix_img, center_frac=args.ref_center_frac
        )
        fixed_refs = (W_spec_fix, D_spec_fix, denom_fix)
        print(
            f"[info] using FIXED references: W={args.fixed_white_hdr}, "
            f"D={args.fixed_dark_hdr}, center_frac={args.ref_center_frac}"
        )

    # 准备 payload 列表
    outdir_str = str(out)
    tasks = []
    for _, r in df.iterrows():
        tasks.append((r.to_dict(), outdir_str, args, fixed_refs))

    rows = []

    # workers=1 时走原来顺序逻辑，方便调试
    if args.workers <= 1:
        print(f"[info] running sequentially on {len(tasks)} samples")
        for payload in tasks:
            row = _process_one_sample(payload)
            if row is not None:
                rows.append(row)
    else:
        print(f"[info] running with ProcessPool: workers={args.workers}, nsamples={len(tasks)}")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for row in ex.map(_process_one_sample, tasks):
                if row is not None:
                    rows.append(row)

    feat = pd.DataFrame(rows)
    out_tsv = out / 'raw_image_features.tsv'
    feat.to_csv(out_tsv, sep='\t', index=False)
    print(f"[OK] wrote {len(feat)} rows -> {out_tsv}")


if __name__ == '__main__':
    main()

