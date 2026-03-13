#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSI preprocess (ASCII logs, Chinese comments)
- 统一反射率标定：支持固定白/黑板；仅用参考图中央带做每波段中位数，按波段广播校正
- 多方案分割（A: NDRE+NDVI+Otsu; B: NIR+NDVI; C: KMeans on NIR; D: NDVI top-k 兜底）
- 每张图保存 mask.png / overlay.png、光谱 npz（wavelength/spec/spec_sg）
- 可生成 per-session 的光谱汇总图（可通过 --no-session-plots 关闭）
"""

import argparse
from pathlib import Path
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


# ---------------- 基础工具 ----------------

def load_cube(hdr_path: Path):
    """读取 ENVI 立方与 wavelength。若无法解析波长，返回空数组以便上游跳过。"""
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
    return int(np.argmin(np.abs(wls - target_nm)))

def pseudo_rgb(R, wls):
    """取 ~700/550/450 nm 伪彩，做 2-98 分位缩放显示。"""
    bR, bG, bB = nearest_band(wls, 700), nearest_band(wls, 550), nearest_band(wls, 450)
    rgb = np.stack([R[:, :, bR], R[:, :, bG], R[:, :, bB]], axis=-1)
    lo = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
    return rgb

def save_mask_and_overlay(mask, R, wls, out_mask_png: Path, out_overlay_png: Path):
    out_mask_png.parent.mkdir(parents=True, exist_ok=True)
    out_overlay_png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_mask_png, mask.astype(float), cmap='gray', vmin=0, vmax=1)
    rgb = pseudo_rgb(R, wls)
    edge = binary_dilation(mask, iterations=1) ^ mask
    overlay = rgb.copy()
    overlay[edge] = [1.0, 0.0, 0.0]
    overlay[~mask] = overlay[~mask] * 0.4
    plt.imsave(out_overlay_png, overlay)

# ---------------- 参考谱（中央带）与反射率 ----------------

def _central_crop(img, frac=0.6):
    """取参考图中央带。frac=0.6 表示取中心 60% 区域，避开边缘/遮挡。"""
    H, W = img.shape[:2]
    fh = max(1, int(H * frac))
    fw = max(1, int(W * frac))
    top = (H - fh) // 2
    left = (W - fw) // 2
    return img[top:top + fh, left:left + fw, :]

def _bandwise_median(img):
    """把参考图压成每波段中位数。"""
    x = img.reshape(-1, img.shape[-1]).astype(np.float32)
    return np.nanmedian(x, axis=0)

def _make_ref_spectrum(white_cube, dark_cube, center_frac=0.6):
    """对白/黑板取中央带后各自做每波段中位数，并构造安全分母。"""
    Wc = _central_crop(white_cube, center_frac)
    Dc = _central_crop(dark_cube, center_frac)
    W_spec = _bandwise_median(Wc)
    D_spec = _bandwise_median(Dc)
    denom = W_spec - D_spec
    eps = max(1e-6, float(np.nanpercentile(np.abs(denom), 10)))
    denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps, denom)
    return W_spec, D_spec, denom

def reflectance_broadcast(sample, W_spec, D_spec, denom):
    """用 1D 参考谱做全幅广播校正。"""
    B = min(sample.shape[-1], len(denom))
    S = sample[..., :B].astype(np.float32)
    R = (S - D_spec[:B].reshape(1, 1, B)) / (denom[:B].reshape(1, 1, B))
    return np.clip(R, 0, 1.5)

def reflectance_pixel(sample, white, dark):
    """像素级校正（尺寸基本一致时可用）。"""
    H = min(sample.shape[0], white.shape[0], dark.shape[0])
    W = min(sample.shape[1], white.shape[1], dark.shape[1])
    sample, white, dark = sample[:H, :W, :], white[:H, :W, :], dark[:H, :W, :]
    R = (sample - dark) / (white - dark + 1e-6)
    return np.clip(R, 0, 1.5)

# ---------------- 分割（四方案） ----------------

def _clean_mask(mask, min_area=200, aspect_ratio_max=15):
    """开/闭运算、去小连通域、去过细长区域。"""
    mask = binary_opening(mask, footprint=disk(2))
    mask = binary_closing(mask, footprint=disk(3))
    mask = remove_small_objects(mask, min_size=min_area)
    lab = label(mask)
    keep = np.zeros_like(mask, dtype=bool)
    for r in regionprops(lab):
        h, w = r.bbox[2] - r.bbox[0], r.bbox[3] - r.bbox[1]
        ar = max(h, w) / (min(h, w) + 1e-6)
        if ar <= aspect_ratio_max:
            keep[lab == r.label] = True
    return keep

def build_leaf_mask(R, wls, min_area=200, min_ndvi=0.10, min_ndre=0.02, aspect_ratio_max=15):
    """A/B/C/兜底 四方案分割。"""
    def nb(nm): return int(np.argmin(np.abs(wls - nm)))
    b670, b705, b740, b800 = nb(670), nb(705), nb(740), nb(800)
    R670, R705, R740, R800 = R[:, :, b670], R[:, :, b705], R[:, :, b740], R[:, :, b800]
    ndvi = (R800 - R670) / (R800 + R670 + 1e-9)
    ndre = (R740 - R705) / (R740 + R705 + 1e-9)

    # Plan A: NDRE+NDVI+Otsu
    try:
        t_ndre = threshold_otsu(ndre[np.isfinite(ndre)])
    except Exception:
        t_ndre = np.nanpercentile(ndre, 70)
    mA = (ndre >= max(t_ndre, min_ndre)) & (ndvi >= min_ndvi)
    mA = _clean_mask(mA, min_area=min_area, aspect_ratio_max=aspect_ratio_max)
    if mA.sum() >= min_area:
        return mA, {'t_ndre': float(t_ndre), 'ndvi_med': float(np.nanmedian(ndvi)), 'area': int(mA.sum())}

    # Plan B: NIR + NDVI 宽松门限
    nir_thr = max(0.08, float(np.nanpercentile(R800, 60)) * 0.5)
    red_thr = float(np.nanpercentile(R670, 95))
    mB = (R800 >= nir_thr) & (R670 <= red_thr) & (ndvi >= min(min_ndvi, 0.05))
    mB = _clean_mask(mB, min_area=min_area, aspect_ratio_max=aspect_ratio_max)
    if mB.sum() >= min_area:
        return mB, {'t_ndre': float(t_ndre), 'ndvi_med': float(np.nanmedian(ndvi)), 'area': int(mB.sum())}

    # Plan C: KMeans on [700, 750, 800]
    try:
        b700, b750 = nb(700), nb(750)
        X = np.stack([R[:, :, b700].ravel(), R[:, :, b750].ravel(), R[:, :, b800].ravel()], 1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        km = KMeans(n_clusters=2, n_init=5, random_state=0).fit(X)
        lab2 = km.labels_.reshape(R.shape[:2])
        m0 = np.nanmean(R800[lab2 == 0]); m1 = np.nanmean(R800[lab2 == 1])
        mC = (lab2 == (0 if m0 >= m1 else 1)) & (ndvi >= 0.0)
        mC = _clean_mask(mC, min_area=min_area, aspect_ratio_max=aspect_ratio_max)
        if mC.sum() >= min_area:
            return mC, {'t_ndre': float(t_ndre), 'ndvi_med': float(np.nanmedian(ndvi)), 'area': int(mC.sum())}
    except Exception:
        pass

    # Plan D: NDVI top-k 兜底
    ndvi_flat = ndvi.ravel()
    k = max(int(0.003 * ndvi_flat.size), min_area)  # top 0.3% 或 min_area
    thr = np.partition(ndvi_flat, -k)[-k]
    mD = ndvi >= max(thr, 0.0)
    mD = _clean_mask(mD, min_area=min_area, aspect_ratio_max=aspect_ratio_max)
    return mD, {'t_ndre': float(t_ndre), 'ndvi_med': float(np.nanmedian(ndvi)), 'area': int(mD.sum())}




def _interp_at(wl_nm, refl, target_nm):
    return float(np.interp(target_nm, wl_nm, refl))

def compute_rep_pair(spec_sg, wl_nm):
    """
    返回 (rep_4pt, rep_d1)；无效则 NaN。
    - 四点法：Guyot & Baret 1988，670/700/740/780 nm
    - 一阶导数法：680–750 nm 区间 dR/dλ 最大处
    """
    import numpy as np
    wl = np.asarray(wl_nm, float); R = np.asarray(spec_sg, float)
    if wl.size < 5 or R.size != wl.size: return np.nan, np.nan
    if wl.min() > 670 or wl.max() < 780: return np.nan, np.nan
    try:
        R670 = _interp_at(wl, R, 670.0)
        R700 = _interp_at(wl, R, 700.0)
        R740 = _interp_at(wl, R, 740.0)
        R780 = _interp_at(wl, R, 780.0)
        rep4 = 700.0 + 40.0 * (((R670 + R780)/2.0) - R700) / max(1e-12, (R740 - R700))
    except Exception:
        rep4 = np.nan
    # 导数法
    repd1 = np.nan
    try:
        m = (wl >= 680) & (wl <= 750) & np.isfinite(R)
        if m.sum() >= 5:
            dR = np.gradient(R[m], wl[m])
            repd1 = float(wl[m][np.argmax(dR)])
    except Exception:
        pass
    if not (680.0 <= (rep4 if np.isfinite(rep4) else 0) <= 750.0): rep4 = np.nan
    if not (680.0 <= (repd1 if np.isfinite(repd1) else 0) <= 750.0): repd1 = np.nan
    return rep4, repd1



# ---------------- 光谱提取 ----------------

def extract_roi_spectrum(R, mask, wls, sg_window=11, sg_poly=2, use_snv=False):
    """ROI 像素平均光谱 + SG 平滑；并基于原始 R 计算若干指数。"""
    pix = R[mask]
    if pix.size == 0:
        return None, None, {}
    spec = np.nanmean(pix, axis=0)

    spec_work = spec.copy()
    if use_snv:
        mu = spec_work.mean()
        sd = spec_work.std() + 1e-8
        spec_work = (spec_work - mu) / sd
    spec_sg = savgol_filter(spec_work, sg_window, sg_poly, mode='interp') if sg_window > 2 else spec_work

    def band(nm): return spec[nearest_band(wls, nm)]
    p800, p670, p550, p531, p570, p700 = band(800), band(670), band(550), band(531), band(570), band(700)
    NDVI = (p800 - p670) / (p800 + p670 + 1e-9)
    GNDVI = (p800 - p550) / (p800 + p550 + 1e-9)
    PRI = (p531 - p570) / (p531 + p570 + 1e-9)
    ARI = (1.0 / (p550 + 1e-9) - 1.0 / (p700 + 1e-9))
    idx = {'NDVI': float(NDVI), 'GNDVI': float(GNDVI), 'PRI': float(PRI), 'ARI': float(ARI)}
    return spec, spec_sg, idx

# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True, help='samples_hsi.csv')
    ap.add_argument('--outdir', required=True, help='results dir')
    ap.add_argument('--sg-window', type=int, default=11)
    ap.add_argument('--sg-poly', type=int, default=2)
    ap.add_argument('--min-area', type=int, default=500)
    ap.add_argument('--min-ndvi', type=float, default=0.2)
    ap.add_argument('--min-ndre', type=float, default=0.05)
    ap.add_argument('--aspect-ratio-max', type=float, default=8.0)
    ap.add_argument('--use-snv', action='store_true', help='apply SNV on per-ROI mean spectrum before SG (for shape)')
    ap.add_argument('--no-session-plots', action='store_true', help='do not generate per-session spectrum plots')
    # 固定参考与中央带
    ap.add_argument('--fixed-white-hdr', type=str, default='', help='use this white reference for ALL samples')
    ap.add_argument('--fixed-dark-hdr', type=str, default='', help='use this dark reference for ALL samples')
    ap.add_argument('--ref-center-frac', type=float, default=0.6, help='central fraction (0-1) used in reference frames')
    # 原始模式兜底（若反射率极差）
    ap.add_argument('--raw-mode', action='store_true', help='segment on robustly scaled raw cube if reflectance is degenerate')
    ap.add_argument('--leaf-ndvi-quantile', type=float, default=0.30,
                help='within initial mask, keep pixels with NDVI >= this quantile (0-1) to drop stems; e.g., 0.3')
    ap.add_argument('--leaf-refine-min-area', type=int, default=150,
                help='min area after leaf refinement')
    ap.add_argument('--leaf-ndre-quantile', type=float, default=0.20,
                help='optional: keep pixels with NDRE >= this quantile inside mask')

    args = ap.parse_args()

    out = Path(args.outdir)
    (out / 'spec').mkdir(parents=True, exist_ok=True)
    (out / 'mask').mkdir(exist_ok=True)
    (out / 'qc' / 'session_plots').mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.samples)
    rows = []
    session_spectra = {}

    # 预加载固定参考（如指定）
    fixed_refs = None
    if args.fixed_white_hdr and args.fixed_dark_hdr:
        wfix_img, wfix_wls = load_cube(Path(args.fixed_white_hdr))
        dfix_img, dfix_wls = load_cube(Path(args.fixed_dark_hdr))
        B = min(wfix_img.shape[-1], dfix_img.shape[-1])
        wfix_img = wfix_img[..., :B]
        dfix_img = dfix_img[..., :B]
        W_spec_fix, D_spec_fix, denom_fix = _make_ref_spectrum(wfix_img, dfix_img, center_frac=args.ref_center_frac)
        fixed_refs = (W_spec_fix, D_spec_fix, denom_fix)
        print(f"[info] using FIXED references: W={args.fixed_white_hdr}, D={args.fixed_dark_hdr}, center_frac={args.ref_center_frac}")

    for i, r in df.iterrows():
        sid = r['sample_id']
        sh = Path(str(r['sample_hdr']))
        wh = Path(str(r['white_hdr'])) if str(r.get('white_hdr', '')) else None
        dh = Path(str(r['dark_hdr'])) if str(r.get('dark_hdr', '')) else None
        if not sh.exists():
            print(f"[skip] {sid}: missing sample_hdr")
            continue

        # 读样本
        sample, wls = load_cube(sh)
        if wls is None or len(wls) == 0:
            print(f"[warn] {sid}: empty wavelength list, skip")
            continue

        # 反射率（优先用固定参考；否则尝试用本地白/黑板的中央带谱）
        if fixed_refs is not None:
            W_spec, D_spec, denom = fixed_refs
            R = reflectance_broadcast(sample, W_spec, D_spec, denom)
        else:
            if (wh is not None and wh.exists()) and (dh is not None and dh.exists()):
                white, _ = load_cube(wh)
                dark, _ = load_cube(dh)
                try:
                    W_spec, D_spec, denom = _make_ref_spectrum(white, dark, center_frac=args.ref_center_frac)
                    R = reflectance_broadcast(sample, W_spec, D_spec, denom)
                except Exception:
                    R = reflectance_pixel(sample, white, dark)
            else:
                # 没有参考就直接用 sample 做稳健缩放，仅用于分割（指数将不可用）
                R = sample.copy()

        # 裁边
        if R.shape[0] > 10 and R.shape[1] > 10:
            R = R[5:-5, 5:-5, :]

        # 如果反射率极差，且用户允许 raw-mode，用原始立方的稳健缩放做分割
        use_raw = False
        try:
            med_R800 = float(np.nanmedian(R[:, :, nearest_band(wls, 800)]))
        except Exception:
            med_R800 = 0.0
        if (not np.isfinite(R).any()) or med_R800 < 0.03:
            use_raw = args.raw_mode

        CubeForSeg = R.copy()
        if use_raw:
            S = sample.copy()
            if S.shape[0] > 10 and S.shape[1] > 10:
                S = S[5:-5, 5:-5, :]
            q1 = np.nanpercentile(S, 5, axis=(0, 1), keepdims=True)
            q9 = np.nanpercentile(S, 95, axis=(0, 1), keepdims=True)
            CubeForSeg = np.clip((S - q1) / (q9 - q1 + 1e-6), 0, 1)

        # 分割
        mask, mstat = build_leaf_mask(
            CubeForSeg, wls,
            min_area=args.min_area,
            min_ndvi=args.min_ndvi,
            min_ndre=args.min_ndre,
            aspect_ratio_max=args.aspect_ratio_max
        )
        # --- 叶片细化：在初始 mask 内按 NDVI 分位裁掉茎（NDVI 通常低于叶片） ---
        try:
            b670 = nearest_band(wls, 670)
            b800 = nearest_band(wls, 800)
            R670 = CubeForSeg[:, :, b670]
            R800 = CubeForSeg[:, :, b800]
            ndvi_map = (R800 - R670) / (R800 + R670 + 1e-9)

            if mask.any():
                q = float(np.nanpercentile(ndvi_map[mask], args.leaf_ndvi_quantile * 100.0))
                # 仅保留 NDVI 高于分位数阈值的像素（去茎）
                mask = mask & (ndvi_map >= q)

                # 细清：小区域/细长去除
                mask = _clean_mask(
                    mask,
                    min_area=max(args.leaf_refine_min_area, args.min_area // 2),
                    aspect_ratio_max=int(args.aspect_ratio_max)
                )
        except Exception:
    # 出现任何异常，不影响主流程
            pass
        
        # 可选：再按 NDRE 分位细化
        try:
            b705 = nearest_band(wls, 705)
            b740 = nearest_band(wls, 740)
            R705 = CubeForSeg[:, :, b705]
            R740 = CubeForSeg[:, :, b740]
            ndre_map = (R740 - R705) / (R740 + R705 + 1e-9)
            if mask.any() and args.leaf_ndre_quantile is not None:
                q_ndre = float(np.nanpercentile(ndre_map[mask], args.leaf_ndre_quantile * 100.0))
                mask = mask & (ndre_map >= q_ndre)
                mask = _clean_mask(mask, min_area=max(args.leaf_refine_min_area, args.min_area // 2),
                                   aspect_ratio_max=int(args.aspect_ratio_max))
        except Exception:
            pass


        # 保存诊断图（即便空 ROI）
        mask_png = out / 'mask' / f'{sid}_mask.png'
        overlay_png = out / 'mask' / f'{sid}_overlay.png'
        save_mask_and_overlay(mask, CubeForSeg, wls, mask_png, overlay_png)

        # 提取 ROI 光谱与指数
        spec, spec_sg, idx = extract_roi_spectrum(R, mask, wls,
                                                  sg_window=args.sg_window,
                                                  sg_poly=args.sg_poly,
                                                  use_snv=args.use_snv)
        if spec is None:
            print(f"[warn] {sid}: empty ROI, saved diagnostics -> {overlay_png}")
            continue

        # 保存光谱
        npz_path = out / 'spec' / f'{sid}.npz'
        np.savez_compressed(npz_path, wavelength=wls, spec=spec, spec_sg=spec_sg)

        # 记录行
        session_id = f"{r.get('temp', '')}_{r.get('time', '')}_{r.get('phase', '')}"
        
        rep4, repd1 = compute_rep_pair(spec_sg=spec_sg, wl_nm=wls)
        idx['REP']    = float(rep4)  if np.isfinite(rep4)  else np.nan
        idx['REP_d1'] = float(repd1) if np.isfinite(repd1) else np.nan


        
        rows.append({
            'sample_id': sid,
            'group_dir': r.get('group_dir', ''),
            'sample_dir': r.get('sample_dir', ''),
            'capture_dir': r.get('capture_dir', ''),
            'sample_hdr': str(sh),
            'white_hdr': str(wh) if wh else '',
            'dark_hdr': str(dh) if dh else '',
            'temp': r.get('temp', ''),
            'time': r.get('time', ''),
            'phase': r.get('phase', ''),
            'session_id': session_id,
            'roi_area': mstat['area'],
            'ndre_thr': mstat['t_ndre'],
            **idx,
            'spec_npz': str(npz_path),
            'mask_png': str(mask_png),
            'overlay_png': str(overlay_png),
            'use_raw_mode': int(use_raw),
            'R800_med': float(med_R800) if np.isfinite(med_R800) else np.nan
        })

        # 汇总会话光谱
        session_spectra.setdefault(session_id, []).append((wls, spec))
        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{len(df)}] processed")

    # 输出图像级表
    feat = pd.DataFrame(rows)
    feat.to_csv(out / 'image_features.tsv', sep='\t', index=False)
    print(f"[OK] wrote {len(feat)} rows -> {out/'image_features.tsv'}")

    # 会话汇总图
    if not args.no_session_plots:
        for sess_id, lst in session_spectra.items():
            w0 = lst[0][0]
            S = []
            for (w, s) in lst:
                if len(w) != len(w0) or np.max(np.abs(w - w0)) > 1e-6:
                    s = np.interp(w0, w, s, left=s[0], right=s[-1])
                S.append(s)
            specs = np.stack(S, axis=0)
            plt.figure(figsize=(7.5, 4.5))
            for s in specs:
                plt.plot(w0, s, linewidth=0.6, alpha=0.6)
            med = np.nanmedian(specs, axis=0)
            q25 = np.nanpercentile(specs, 25, axis=0)
            q75 = np.nanpercentile(specs, 75, axis=0)
            plt.fill_between(w0, q25, q75, alpha=0.25, label='IQR (25-75%)')
            plt.plot(w0, med, linewidth=2.0, label='Median')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Reflectance')
            plt.title(f'Session: {sess_id} (n={specs.shape[0]})')
            plt.legend(loc='best', fontsize=8)
            plt.tight_layout()
            out_png = out / 'qc' / 'session_plots' / f'{sess_id}.png'
            plt.savefig(out_png, dpi=160)
            plt.close()
        print(f"[OK] session spectra saved to {out/'qc'/'session_plots'}")


if __name__ == '__main__':
    main()

