#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TIME_MAP = {'2h': 2, '6h': 6, '1d': 24, '3d': 72, '7d': 168}
METRICS = ['ndvi', 'gndvi', 'pri', 'ari']
REP_PREF = 'rep_d1'
RECOVERY_OFFSET = 168.0
DELTA_METRICS = ['dndvi', 'dgndvi', 'dpri', 'dari', 'drep', 'sam', 'euc']


def to_hours(t):
    if pd.isna(t):
        return np.nan
    s = str(t).strip().lower()
    if s in TIME_MAP:
        return float(TIME_MAP[s])
    if s.endswith('h'):
        try:
            return float(s[:-1])
        except Exception:
            return np.nan
    if s.endswith('d'):
        try:
            return float(s[:-1]) * 24.0
        except Exception:
            return np.nan
    return np.nan


def eff_hours(phase, time_str):
    th = to_hours(time_str)
    if pd.isna(th):
        return np.nan
    return th + RECOVERY_OFFSET if str(phase).lower().startswith('recovery_from_') else th


def se(x):
    x = pd.to_numeric(x, errors='coerce')
    x = x[np.isfinite(x)]
    n = len(x)
    if n <= 1:
        return np.nan
    return float(x.std(ddof=1) / (n ** 0.5))


def spectral_angle(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if a.size == 0 or b.size == 0:
        return np.nan
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    cosv = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.arccos(cosv))


def euclidean(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if a.size == 0 or b.size == 0:
        return np.nan
    return float(np.linalg.norm(a - b))


def auc_xy(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return np.nan
    x = x[m]
    y = np.abs(y[m])
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)


def nearest_ctrl_time(sf, t):
    s = sf.loc[(sf['phase'].str.lower() == 'control') & sf['t_eff_h'].notna(), 't_eff_h']
    if s.empty or pd.isna(t):
        return np.nan
    return float(s.iloc[(s.sub(float(t))).abs().argmin()])


def pick_at(df, t, col):
    m = df.drop_duplicates(subset=['t_eff_h']).set_index('t_eff_h')
    try:
        return float(m.at[float(t), col])
    except KeyError:
        return np.nan


def compute_pair_metrics(sf, stress_phase, rec_phase, temp_label):
    """
    只在给定的一对相位中计算韧性：stress_X vs recovery_from_X，统一用 t_eff_h。
    temp_label: "10"/"35" 等，用于输出到 resilience_metrics.tsv 的 temp 列。
    """
    rows = []
    S = sf[sf['phase'].str.lower() == stress_phase].copy()
    R = sf[sf['phase'].str.lower() == rec_phase].copy()

    t_early = 2.0
    t_end = 168.0

    def auc_on(df, col):
        return auc_xy(df['t_eff_h'].to_numpy(dtype=float),
                      df[col].to_numpy(dtype=float))

    for m in DELTA_METRICS:
        d2h = pick_at(S, t_early, m)
        resistance = -abs(d2h) if pd.notna(d2h) else np.nan

        end = pick_at(S, t_end, m)

        t_half = np.nan
        if pd.notna(end) and not R.empty:
            target = 0.5 * abs(end)
            for _, rr in R.sort_values('t_eff_h').iterrows():
                val = rr[m]
                if pd.notna(val) and abs(val) <= target:
                    t_half = float(rr['t_eff_h']) - RECOVERY_OFFSET
                    break

        auc_s = auc_on(S, m)
        auc_r = auc_on(R, m)

        resilience = 1.0 - (auc_r / auc_s) if (pd.notna(auc_s) and auc_s > 0 and pd.notna(auc_r)) else np.nan
        hysteresis = (auc_r - auc_s) if (pd.notna(auc_s) and pd.notna(auc_r)) else np.nan

        rows.append({
            'temp': str(temp_label),
            'pair': f'{stress_phase} vs {rec_phase}',
            'metric': m,
            'resistance': resistance,
            't_half_h': t_half,
            'auc_stress': auc_s,
            'auc_recovery': auc_r,
            'resilience': resilience,
            'hysteresis': hysteresis
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image-features', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--control-temp', default='25')
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.image_features, sep='\t')
    df.columns = [c.lower() for c in df.columns]

    for m in METRICS:
        if m not in df.columns:
            df[m] = np.nan

    if REP_PREF in df.columns:
        df['rep_nm'] = pd.to_numeric(df[REP_PREF], errors='coerce')
    elif 'rep' in df.columns:
        df['rep_nm'] = pd.to_numeric(df['rep'], errors='coerce')
    elif 'rep_nm' in df.columns:
        df['rep_nm'] = pd.to_numeric(df['rep_nm'], errors='coerce')
    else:
        df['rep_nm'] = np.nan

    spec_cols = []
    for c in df.columns:
        if c.startswith('r_'):
            s = c[2:]
            try:
                float(s)
                spec_cols.append(c)
            except Exception:
                pass
    spec_cols.sort(key=lambda c: float(c[2:]))

    keys = ['temp', 'time', 'phase']
    mean_df = df.groupby(keys, dropna=False)[METRICS + ['rep_nm'] + spec_cols].mean().reset_index()
    n_df = df.groupby(keys, dropna=False).size().rename('n').reset_index()
    se_df = df.groupby(keys, dropna=False)[METRICS + spec_cols].agg(se).reset_index()
    se_df = se_df.rename(columns={c: f'{c}_se' for c in METRICS + spec_cols})

    sf = mean_df.merge(se_df, on=keys, how='left').merge(n_df, on=keys, how='left')
    sf['temp'] = sf['temp'].astype(str)
    sf['t_eff_h'] = [eff_hours(p, t) for p, t in zip(sf['phase'], sf['time'])]

    sf.to_csv(out / 'session_features.tsv', sep='\t', index=False)

    # Δ 指标
    for ph in sf['phase'].unique():
        if str(ph).lower() == 'control':
            continue
        ph_mask = (sf['phase'] == ph)
        times = sf.loc[ph_mask, 't_eff_h'].dropna().unique()
        for t in np.sort(times):
            base_rows = sf.loc[(sf['phase'].str.lower() == 'control') & (sf['t_eff_h'] == t)]
            if base_rows.empty:
                t0 = nearest_ctrl_time(sf, t)
                if pd.isna(t0):
                    continue
                base_rows = sf.loc[(sf['phase'].str.lower() == 'control') & (sf['t_eff_h'] == t0)]
                if base_rows.empty:
                    continue
            base_row = base_rows.iloc[0]

            idx = sf.index[ph_mask & (sf['t_eff_h'] == t)]
            if len(idx) == 0:
                continue

            for m in METRICS:
                sf.loc[idx, f'd{m}'] = sf.loc[idx, m].astype(float).values - float(base_row[m])

            sf.loc[idx, 'drep'] = sf.loc[idx, 'rep_nm'].astype(float).values - float(base_row['rep_nm'])

            if spec_cols:
                base_vec = base_row[spec_cols].to_numpy(dtype=float)
                for i in idx:
                    sess_vec = sf.loc[i, spec_cols].to_numpy(dtype=float)
                    sf.at[i, 'sam'] = spectral_angle(sess_vec, base_vec)
                    sf.at[i, 'euc'] = euclidean(sess_vec, base_vec)

    # delta_hsi.tsv：键列 + Δ列 + t_eff_h
    keep = ['temp', 'time', 'phase', 'session_id', 'n', 't_eff_h'] + \
           [f'd{x}' for x in METRICS] + ['drep', 'sam', 'euc']
    for k in keep:
        if k not in sf.columns:
            sf[k] = np.nan
    sf[keep].to_csv(out / 'delta_hsi.tsv', sep='\t', index=False)

    # 韧性指标：10℃ / 35℃ 各一对
    rows = []
    rows += compute_pair_metrics(sf, 'stress_10', 'recovery_from_10', temp_label='10')
    rows += compute_pair_metrics(sf, 'stress_35', 'recovery_from_35', temp_label='35')
    pd.DataFrame(rows).to_csv(out / 'resilience_metrics.tsv', sep='\t', index=False)

    print("[OK] wrote:")
    print(" -", out / 'session_features.tsv')
    print(" -", out / 'delta_hsi.tsv')
    print(" -", out / 'resilience_metrics.tsv')


if __name__ == '__main__':
    main()

