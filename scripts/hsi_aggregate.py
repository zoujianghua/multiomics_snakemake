#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSI aggregate (ASCII logs, Chinese comments)
- 会话聚合（中位数+SE）
- Δ 指标（相对 25C，对照缺点位线性插值）
- 动态指标（抗性/半恢复/韧性/迟滞）
- 会话中位数光谱 npz（含 wavelength/spec_med/spec_ctrl/spec_delta）
- REP（nm）、dREP（nm）、SAM（rad）、EUC（L2）
- 列名大小写稳健：内部统一为小写处理
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TIME_MAP_DEFAULT = {'2h': 2, '6h': 6, '1d': 24, '3d': 72, '7d': 168}
METRICS_CANON = ['ndvi', 'gndvi', 'pri', 'ari']  # 统一用小写

# ---------- 小工具 ----------

def time_to_hours(t):
    if pd.isna(t): return np.nan
    t = str(t).strip().lower()
    if t in TIME_MAP_DEFAULT: return float(TIME_MAP_DEFAULT[t])
    if t.endswith('h'):
        try: return float(t[:-1])
        except: return np.nan
    if t.endswith('d'):
        try: return float(t[:-1]) * 24.0
        except: return np.nan
    return np.nan

def auc_discrete(times, ys):
    if len(times) < 2: 
        return np.nan
    idx = np.argsort(times)
    t = np.asarray(times)[idx]; y = np.asarray(ys)[idx]
    # numpy >= 1.26 推荐 trapezoid
    return np.trapezoid(y, t)


def nearest_band_idx(wls, target_nm):
    return int(np.argmin(np.abs(wls - target_nm)))

def compute_rep(wls, spec):
    # 在 680–760 nm 范围找一阶导数最大点，并用三点二次细化
    wls = np.asarray(wls); spec = np.asarray(spec)
    if len(wls) < 5 or len(spec) != len(wls): return np.nan
    m = (wls >= 680) & (wls <= 760)
    if m.sum() < 5: return np.nan
    x = wls[m]; y = spec[m]
    dy = np.gradient(y, x)
    k = int(np.argmax(dy))
    rep = float(x[k])
    if 0 < k < len(x) - 1:
        x0, x1, x2 = x[k-1], x[k], x[k+1]
        y0, y1, y2 = dy[k-1], dy[k], dy[k+1]
        denom = (x0-x1)*(x0-x2)*(x1-x2)
        if denom != 0:
            a = (x2*(y1-y0) + x1*(y0-y2) + x0*(y2-y1)) / denom
            b = (x2*x2*(y0-y1) + x1*x1*(y2-y0) + x0*x0*(y1-y2)) / denom
            if a != 0:
                xr = -b/(2*a)
                if x0 <= xr <= x2: rep = float(xr)
    return rep

def spectral_angle(u, v):
    u = np.asarray(u); v = np.asarray(v)
    if len(u) != len(v) or len(u) == 0: return np.nan
    un = np.linalg.norm(u); vn = np.linalg.norm(v)
    if un == 0 or vn == 0: return np.nan
    cosv = np.clip(np.dot(u, v) / (un * vn), -1.0, 1.0)
    return float(np.arccos(cosv))

def l2_distance(u, v):
    u = np.asarray(u); v = np.asarray(v)
    if len(u) != len(v) or len(u) == 0: return np.nan
    return float(np.linalg.norm(u - v))

def interp_to(w_src, y_src, w_tgt):
    w_src = np.asarray(w_src); y_src = np.asarray(y_src); w_tgt = np.asarray(w_tgt)
    if len(w_src) == 0 or len(y_src) != len(w_src):
        return np.full_like(w_tgt, np.nan, dtype=float)
    xu, idx = np.unique(w_src, return_index=True)
    yu = y_src[idx]
    return np.interp(w_tgt, xu, yu, left=yu[0], right=yu[-1])

# ---------- 会话级处理 ----------

def load_session_median_spectra(image_df, out_dir_npz):
    out_dir_npz.mkdir(parents=True, exist_ok=True)
    groups = image_df.groupby(['temp', 'time', 'phase', 'session_id'], dropna=False)
    rows = []
    for (temp, time, phase, sid), g in groups:
        specs = []
        wls0 = None
        for p in g['spec_npz']:
            try:
                arr = np.load(str(p))
                w = arr['wavelength']; s = arr['spec_sg'] if 'spec_sg' in arr.files else arr['spec']
                if wls0 is None:
                    wls0 = w.copy()
                else:
                    if len(w) != len(wls0) or np.max(np.abs(w - wls0)) > 1e-6:
                        s = np.interp(wls0, w, s, left=s[0], right=s[-1])
                specs.append(s)
            except Exception:
                continue
        if len(specs) == 0 or wls0 is None:
            continue
        specs = np.stack(specs, axis=0)
        med = np.nanmedian(specs, axis=0)
        out_path = out_dir_npz / f"{sid}.npz"
        np.savez_compressed(out_path, wavelength=wls0, spec_med=med)
        rows.append({'temp': temp, 'time': time, 'phase': phase, 'session_id': sid, 'spec_npz': str(out_path)})
    return pd.DataFrame(rows)

def agg_session(image_df):
    g = image_df.groupby(['temp', 'time', 'phase', 'session_id'], dropna=False)
    med = g[METRICS_CANON].median().reset_index()
    counts = g.size().rename('n').reset_index()
    std = g[METRICS_CANON].std().reset_index().rename(columns={m: f'{m}_sd' for m in METRICS_CANON})
    ses = std.merge(counts, on=['temp', 'time', 'phase', 'session_id'], how='left')
    for m in METRICS_CANON:
        ses[f'{m}_se'] = ses[f'{m}_sd'] / np.sqrt(ses['n'].clip(lower=1))
    out = med.merge(ses[['temp', 'time', 'phase', 'session_id', 'n'] + [f'{m}_se' for m in METRICS_CANON]],
                    on=['temp', 'time', 'phase', 'session_id'], how='left')
    return out

def build_control_curves(sess_df, spectra_df, control_temp='25'):
    # 指数基线
    base = sess_df[(sess_df['temp'].astype(str) == str(control_temp)) & (sess_df['time'].notna())].copy()
    base['t_h'] = base['time'].apply(time_to_hours)
    base = base.dropna(subset=['t_h']).sort_values('t_h')
    base_idx = {}
    for m in METRICS_CANON:
        x = base['t_h'].to_numpy()
        y = base.groupby('t_h')[m].median().reindex(x).to_numpy()
        if len(x) == 0:
            base_idx[m] = (np.array([0.0]), np.array([0.0]))
        else:
            xu, idx = np.unique(x, return_index=True)
            yu = y[idx]
            base_idx[m] = (xu, yu)
    # 光谱基线（每个 time 一个 npz）
    base_spec = {}
    base_sess = spectra_df[spectra_df['temp'].astype(str) == str(control_temp)]
    for t, sub in base_sess.groupby('time'):
        wls0 = None; specs = []
        for p in sub['spec_npz']:
            try:
                z = np.load(str(p))
                w = z['wavelength']; s = z['spec_med']
                if wls0 is None:
                    wls0 = w.copy()
                else:
                    if len(w) != len(wls0) or np.max(np.abs(w - wls0)) > 1e-6:
                        s = interp_to(w, s, wls0)
                specs.append(s)
            except Exception:
                continue
        if wls0 is not None and len(specs) > 0:
            med = np.nanmedian(np.stack(specs, axis=0), axis=0)
            base_spec[str(t)] = (wls0, med)
    return base_idx, base_spec

def compute_delta_indices(sess_df, base_idx):
    out = sess_df.copy()
    out['t_h'] = out['time'].apply(time_to_hours)
    for m in METRICS_CANON:
        xu, yu = base_idx[m]
        base_vals = np.interp(out['t_h'].fillna(np.nan), xu, yu,
                              left=yu[0] if len(yu) > 0 else 0.0,
                              right=yu[-1] if len(yu) > 0 else 0.0)
        out[f'd{m}'] = out[m].to_numpy() - base_vals
    return out.drop(columns=['t_h'])

def enrich_with_rep_and_dist(sess_df, spectra_df, base_spec):
    sess = sess_df.merge(spectra_df[['session_id', 'spec_npz']], on='session_id', how='left')
    rep_list, drep_list, sam_list, euc_list = [], [], [], []
    for _, row in sess.iterrows():
        spec_path = row.get('spec_npz', None)
        time_key = str(row.get('time', ''))
        try:
            z = np.load(str(spec_path))
            w = z['wavelength']; s = z['spec_med']
        except Exception:
            rep_list.append(np.nan); drep_list.append(np.nan); sam_list.append(np.nan); euc_list.append(np.nan); continue
        rep = compute_rep(w, s)
        rep_list.append(rep)
        if time_key in base_spec:
            wc, sc = base_spec[time_key]
            sc_int = interp_to(wc, sc, w) if (len(w) != len(wc) or np.max(np.abs(w - wc)) > 1e-6) else sc
            drep = rep - compute_rep(w, sc_int)
            b_lo = nearest_band_idx(w, 500)
            b_hi = nearest_band_idx(w, 900)
            u = s[b_lo:b_hi+1]; v = sc_int[b_lo:b_hi+1]
            sam = spectral_angle(u, v)
            euc = l2_distance(u, v)
        else:
            drep = np.nan; sam = np.nan; euc = np.nan
        drep_list.append(float(drep) if pd.notna(drep) else np.nan)
        sam_list.append(float(sam) if pd.notna(sam) else np.nan)
        euc_list.append(float(euc) if pd.notna(euc) else np.nan)
    sess['rep_nm'] = rep_list
    sess['drep'] = drep_list
    sess['sam'] = sam_list
    sess['euc'] = euc_list
    return sess

def save_session_delta_npz(spectra_df, base_spec):
    for _, row in spectra_df.iterrows():
        sid = row['session_id']; tkey = str(row['time'])
        try:
            z = np.load(str(row['spec_npz']))
            w = z['wavelength']; s = z['spec_med']
        except Exception:
            continue
        sc = None
        if tkey in base_spec:
            wc, sctrl = base_spec[tkey]
            sc = interp_to(wc, sctrl, w) if (len(w) != len(wc) or np.max(np.abs(w - wc)) > 1e-6) else sctrl
        d = s - sc if sc is not None else np.full_like(s, np.nan)
        np.savez_compressed(row['spec_npz'], wavelength=w, spec_med=s,
                            spec_ctrl=(sc if sc is not None else np.full_like(s, np.nan)),
                            spec_delta=d)

def dynamic_metrics(delta_df):
    metrics_all = METRICS_CANON + ['drep', 'sam', 'euc']
    rows = []
    temps = sorted(set(delta_df['temp'].astype(str)) - set(['25', '25.0']))
    for temp in temps:
        for m in metrics_all:
            dS = delta_df[(delta_df['temp'].astype(str) == temp) & (delta_df['phase'] == 'stress')].copy()
            dR = delta_df[(delta_df['temp'].astype(str) == temp) & (delta_df['phase'] == 'recovery')].copy()
            if m in METRICS_CANON:
                col = f'd{m}'
                series_S = dS[col]; series_R = dR[col]
            else:
                col = m
                series_S = dS[col]; series_R = dR[col]
            d2h = series_S[dS['time'] == '2h'].median() if '2h' in set(dS['time']) else np.nan
            resistance = -abs(d2h) if pd.notna(d2h) else np.nan
            end = series_S[dS['time'] == '7d'].median() if '7d' in set(dS['time']) else np.nan
            t_half = np.nan
            if pd.notna(end) and not dR.empty:
                target = 0.5 * abs(end)
                order = {k: TIME_MAP_DEFAULT.get(k, 1e9) for k in set(dR['time'])}
                for t in sorted(order, key=lambda z: order[z]):
                    val = abs(series_R[dR['time'] == t].median())
                    if pd.notna(val) and val <= target:
                        t_half = TIME_MAP_DEFAULT.get(t, np.nan)
                        break
            def ser(vals, times):
                ts = [TIME_MAP_DEFAULT.get(x, np.nan) for x in times]
                ys = [abs(v) if pd.notna(v) else np.nan for v in vals]
                ts2, ys2 = [], []
                for t, y in zip(ts, ys):
                    if pd.notna(t) and pd.notna(y):
                        ts2.append(t); ys2.append(y)
                return ts2, ys2
            ts_s, ys_s = ser(series_S.tolist(), dS['time'].tolist())
            ts_r, ys_r = ser(series_R.tolist(), dR['time'].tolist())
            auc_s = auc_discrete(ts_s, ys_s); auc_r = auc_discrete(ts_r, ys_r)
            resilience = np.nan
            if pd.notna(auc_s) and auc_s > 0 and pd.notna(auc_r):
                resilience = 1.0 - (auc_r / auc_s)
            hysteresis = (auc_r - auc_s) if (pd.notna(auc_s) and pd.notna(auc_r)) else np.nan
            rows.append({'temp': temp, 'metric': m,
                         'resistance': resistance, 't_half_h': t_half,
                         'auc_stress': auc_s, 'auc_recovery': auc_r,
                         'resilience': resilience, 'hysteresis': hysteresis})
    return pd.DataFrame(rows)

# ---------- 入口 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image-features', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--control-temp', default='25')
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    out_specdir = out / 'session_spectra'; out_specdir.mkdir(exist_ok=True)

    feat = pd.read_csv(args.image_features, sep='\t')
    # 列名统一为小写，保证稳健
    feat.columns = [c.lower() for c in feat.columns]
    for m in METRICS_CANON:
        if m not in feat.columns:
            feat[m] = np.nan

    spectra = load_session_median_spectra(feat, out_specdir)

    sess_idx = agg_session(feat)
    sess_idx.to_csv(out / 'session_features.tsv', sep='\t', index=False)

    base_idx, base_spec = build_control_curves(sess_idx, spectra, control_temp=str(args.control_temp))

    delt_idx = compute_delta_indices(sess_idx, base_idx)

    sess_enriched = enrich_with_rep_and_dist(delt_idx, spectra, base_spec)

    save_session_delta_npz(spectra, base_spec)

    cols_keep = ['temp', 'time', 'phase', 'session_id', 'n'] + \
                METRICS_CANON + [f'd{m}' for m in METRICS_CANON] + \
                ['rep_nm', 'drep', 'sam', 'euc']
    for c in cols_keep:
        if c not in sess_enriched.columns:
            sess_enriched[c] = np.nan
    sess_enriched[cols_keep].to_csv(out / 'delta_hsi.tsv', sep='\t', index=False)

    dyn = dynamic_metrics(sess_enriched)
    dyn.to_csv(out / 'resilience_metrics.tsv', sep='\t', index=False)

    print("[OK] wrote:")
    print(" -", out / 'session_features.tsv')
    print(" -", out / 'delta_hsi.tsv')
    print(" -", out / 'resilience_metrics.tsv')
    print(" - session spectra npz:", out_specdir)

if __name__ == '__main__':
    main()

