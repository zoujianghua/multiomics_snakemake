#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 clean_image_features.tsv 的基础上计算 400–1000nm 范围内尽可能多的经典植被指数。

类别：
- 结构 / 叶绿素：
    ndvi, gndvi, rdvi, dvi, sr, msr, savi, osavi, msavi, msavi2, wdrvi
- red-edge / 氮素相关：
    ndre1/2/3, gi, gci, cigreen, cirededge, rep, rep_d1
- 色素 / 黄化：
    pri, ari1, ari2, npci, sipi, psri, cri1, cri2
- 综合指数：
    tcari, mcari, tcari_osavi, mcari2, mtvi2
- 水分 / 结构：
    wi (r900/r970), wbi(同 wi)

注意：每个指数只生成 **一列小写名字**，不会再有 “NDVI + ndvi” 这种重复。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_rep_pair(spec_sg, wl_nm):
    """计算 (四点 rep, 一阶导数 rep_d1)，无效返回 NaN。"""
    wl = np.asarray(wl_nm, float)
    R = np.asarray(spec_sg, float)
    if wl.size < 5 or R.size != wl.size:
        return np.nan, np.nan
    if wl.min() > 670 or wl.max() < 780:
        return np.nan, np.nan

    def _interp_at(target_nm):
        return float(np.interp(target_nm, wl, R))

    try:
        R670 = _interp_at(670.0)
        R700 = _interp_at(700.0)
        R740 = _interp_at(740.0)
        R780 = _interp_at(780.0)
        rep4 = 700.0 + 40.0 * (((R670 + R780) / 2.0) - R700) / max(1e-12, (R740 - R700))
    except Exception:
        rep4 = np.nan

    repd1 = np.nan
    try:
        m = (wl >= 680) & (wl <= 750) & np.isfinite(R)
        if m.sum() >= 5:
            dR = np.gradient(R[m], wl[m])
            repd1 = float(wl[m][np.argmax(dR)])
    except Exception:
        pass

    if not (680.0 <= (rep4 if np.isfinite(rep4) else 0) <= 750.0):
        rep4 = np.nan
    if not (680.0 <= (repd1 if np.isfinite(repd1) else 0) <= 750.0):
        repd1 = np.nan
    return rep4, repd1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True,
                    help="results/hsi/clean_image_features.tsv")
    ap.add_argument("--out", required=True,
                    help="results/hsi/image_features.tsv")
    args = ap.parse_args()

    df = pd.read_csv(args.images, sep="\t")

    # 找光谱列
    spec_cols = [c for c in df.columns if c.startswith("R_")]
    if not spec_cols:
        raise RuntimeError("未找到任何 R_<nm> 列，无法计算植被指数。")

    wl = np.array([float(c[2:]) for c in spec_cols], float)
    order = np.argsort(wl)
    wl_sorted = wl[order]
    spec_cols_sorted = [spec_cols[i] for i in order]

    def band_col(target_nm: float) -> str:
        idx = int(np.argmin(np.abs(wl_sorted - target_nm)))
        return spec_cols_sorted[idx]

    # 常用波段列
    col_BLUE = band_col(450.0)
    col_GREEN = band_col(550.0)
    col_RED = band_col(670.0)

    col_R430 = band_col(430.0)
    col_R445 = band_col(445.0)
    col_R500 = band_col(500.0)
    col_R510 = band_col(510.0)
    col_R531 = band_col(531.0)
    col_R550 = band_col(550.0)
    col_R570 = band_col(570.0)
    col_R680 = band_col(680.0)
    col_R700 = band_col(700.0)
    col_R705 = band_col(705.0)
    col_R720 = band_col(720.0)
    col_R740 = band_col(740.0)
    col_R750 = band_col(750.0)
    col_R800 = band_col(800.0)
    col_R900 = band_col(900.0)
    col_R970 = band_col(970.0)
    col_R670 = band_col(670.0)

    # 简写（Series）
    Rb = df[col_BLUE]
    Rg = df[col_GREEN]
    Rr = df[col_RED]
    R430 = df[col_R430]
    R445 = df[col_R445]
    R500 = df[col_R500]
    R510 = df[col_R510]
    R531 = df[col_R531]
    R550 = df[col_R550]
    R570 = df[col_R570]
    R680 = df[col_R680]
    R700 = df[col_R700]
    R705 = df[col_R705]
    R720 = df[col_R720]
    R740 = df[col_R740]
    R750 = df[col_R750]
    R800 = df[col_R800]
    R900 = df[col_R900]
    R970 = df[col_R970]
    R670 = df[col_R670]

    NIR = R800
    eps = 1e-9

    def mk(name: str, series):
        """只写一列，小写名字。"""
        df[name.lower()] = series

    # ===== 结构 / 叶绿素 =====
    ndvi = (NIR - Rr) / (NIR + Rr + eps)
    gndvi = (NIR - Rg) / (NIR + Rg + eps)
    rdvi = (NIR - Rr) / np.sqrt(np.clip(NIR + Rr, eps, None))
    dvi = NIR - Rr
    sr = NIR / (Rr + eps)
    msr = (NIR / (Rr + eps) - 1.0) / np.sqrt(
        np.clip(NIR / (Rr + eps) + 1.0, eps, None)
    )

    L_savi = 0.5
    savi = (NIR - Rr) * (1.0 + L_savi) / (NIR + Rr + L_savi + eps)
    osavi = (NIR - Rr) / (NIR + Rr + 0.16 + eps)

    msavi2 = 0.5 * (
        2.0 * NIR + 1.0 - np.sqrt(
            np.clip((2.0 * NIR + 1.0) ** 2 - 8.0 * (NIR - Rr), eps, None)
        )
    )

    a = 0.1
    wdrvi = (a * NIR - Rr) / (a * NIR + Rr + eps)

    mk("ndvi", ndvi)
    mk("gndvi", gndvi)
    mk("rdvi", rdvi)
    mk("dvi", dvi)
    mk("sr", sr)
    mk("msr", msr)
    mk("savi", savi)
    mk("osavi", osavi)
    mk("msavi", msavi2)
    mk("msavi2", msavi2)
    mk("wdrvi", wdrvi)

    # ===== Red-edge / 氮素相关 =====
    ndre1 = (NIR - R705) / (NIR + R705 + eps)
    ndre2 = (NIR - R720) / (NIR + R720 + eps)
    ndre3 = (NIR - R740) / (NIR + R740 + eps)
    gi = Rg / (Rr + eps)

    cigreen = NIR / (R550 + eps) - 1.0
    cirededge = NIR / (R700 + eps) - 1.0
    gci = NIR / (Rg + eps) - 1.0

    mk("ndre1", ndre1)
    mk("ndre2", ndre2)
    mk("ndre3", ndre3)
    mk("gi", gi)
    mk("cigreen", cigreen)
    mk("cirededge", cirededge)
    mk("gci", gci)

    # ===== 色素 / 黄化相关 =====
    pri = (R531 - R570) / (R531 + R570 + eps)
    ari1 = 1.0 / (R550 + eps) - 1.0 / (R700 + eps)
    ari2 = NIR * ari1
    npci = (R680 - R430) / (R680 + R430 + eps)
    sipi = (R800 - R445) / (R800 + R680 + eps)
    psri = (R680 - R500) / (R750 + eps)

    cri1 = 1.0 / (R510 + eps) - 1.0 / (R550 + eps)
    cri2 = 1.0 / (R510 + eps) - 1.0 / (R700 + eps)

    mk("pri", pri)
    mk("ari1", ari1)
    mk("ari2", ari2)
    mk("npci", npci)
    mk("sipi", sipi)
    mk("psri", psri)
    mk("cri1", cri1)
    mk("cri2", cri2)

    # ===== 综合结构指数（tcari / mcari / mtvi2 等） =====
    mcari = ((R700 - R670) - 0.2 * (R700 - R550)) * (R700 / (R670 + eps))
    tcari = 3.0 * ((R700 - R670) - 0.2 * (R700 - R550) * (R700 / (R670 + eps)))

    tcari_osavi = tcari / (osavi + eps)

    mcari2_num = 1.5 * (2.5 * (NIR - R670) - 1.3 * (NIR - R550))
    mtvi2_num = 1.5 * (1.2 * (NIR - R550) - 2.5 * (R670 - R550))
    denom = np.sqrt(
        np.clip(
            (2.0 * NIR + 1.0) ** 2
            - (6.0 * NIR - 5.0 * np.sqrt(np.abs(R670) + eps))
            - 0.5,
            eps,
            None,
        )
    )

    mcari2 = mcari2_num / (denom + eps)
    mtvi2 = mtvi2_num / (denom + eps)

    mk("mcari", mcari)
    mk("tcari", tcari)
    mk("tcari_osavi", tcari_osavi)
    mk("mcari2", mcari2)
    mk("mtvi2", mtvi2)

    # ===== 水分 / 近红外结构 =====
    wi = R900 / (R970 + eps)
    mk("wi", wi)
    mk("wbi", wi)   # Water Band Index，形式上等同 wi

    # ===== REP / REP_d1 =====
    reps4 = []
    repsd1 = []
    for row in df[spec_cols_sorted].itertuples(index=False):
        spec_row = np.asarray(row, float)
        r4, rd = compute_rep_pair(spec_row, wl_sorted)
        reps4.append(r4)
        repsd1.append(rd)

    mk("rep", np.array(reps4, float))
    mk("rep_d1", np.array(repsd1, float))

    # 输出最终 image_features.tsv
    path_out = Path(args.out)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_out, sep="\t", index=False)
    print(
        f"[VI] wrote image_features with {df.shape[0]} rows, "
        f"{df.shape[1]} columns -> {path_out}"
    )


if __name__ == "__main__":
    main()

