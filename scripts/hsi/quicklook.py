#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quicklook.py —— 论文级 HSI 可视化与多组学桥接

【HSI patch pipeline - 论文级可视化】

功能：
1. Image-level 快速总览（向后兼容原有输出）
2. 可选的 leaf / patch 级别聚合与可视化
3. 统一的配色和美术风格，适合论文正式图（300+ dpi，多子图布局）
4. 为后续与代谢组 / RNA-seq 联动准备好聚合表

输入：
  --images     image_features.tsv      （图像级，含 R_* 光谱、NDVI/GNDVI/PRI/ARI、roi_area、seg_plan、R800_med 等）
  --features   session_features.tsv    （会话级：均值与 *_se、rep_nm 等）
  --delta      delta_hsi.tsv           （Δ指标：dNDVI/dGNDVI/dPRI/dARI、dREP、SAM/EUC、t_eff_h）
  --resilience resilience_metrics.tsv  （resistance/t_half_h/auc/resilience/hysteresis）

可选输入：
  --leaf-features   leaf_features.tsv  （叶片级特征）
  --patch-features  patch_features.tsv  （patch级特征）
  --patch-target    phase_core         （patch_features 里的标签列名）

输出：
  论文级图像（统一前缀，300+ dpi）：
    {fig_prefix}_timeseries_ndvi_rep.png
    {fig_prefix}_heatmap_dndvi.png / {fig_prefix}_heatmap_drep.png
    {fig_prefix}_resilience_bar.png
    {fig_prefix}_patch_vs_leaf_dist.png（如果提供了 leaf/patch 数据）
    {fig_prefix}_patch_cnn_metrics.png（预留接口）
  
  向后兼容的原有输出（保持路径不变）：
    plot_ndvi_timeseries.png / plot_drep_timeseries.png / plot_dndvi_heatmap.png 等
  
  多组学桥接表：
    {omics_join_out}（默认 quicklook_omics_bridge.tsv）
"""

import argparse
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

# ==================== 全局常量 ====================
TIME_MAP = {'2h': 2, '6h': 6, '1d': 24, '3d': 72, '7d': 168}
CANON_PHASES = ['stress_10', 'stress_35', 'recovery_from_10', 'recovery_from_35']
IDX = ['ndvi', 'gndvi', 'pri', 'ari']
DMAP = {m: f'd{m}' for m in IDX}

# 统一配色方案（论文级）
PHASE_COLORS = {
    'control': '#808080',  # 灰色
    'stress_10': '#d62728',  # 红色（冷胁迫）
    'stress_35': '#1f77b4',  # 蓝色（热胁迫）
    'recovery_from_10': '#ff7f0e',  # 橙色（从冷恢复）
    'recovery_from_35': '#2ca02c',  # 绿色（从热恢复）
    'recovery': '#9467bd',  # 紫色（通用恢复）
}

# 默认 phase_core 到 phase 的映射（如果存在）
PHASE_CORE_TO_PHASE = {
    'control': 'control',
    'cold': 'stress_10',
    'heat': 'stress_35',
    'recovery_from_cold': 'recovery_from_10',
    'recovery_from_heat': 'recovery_from_35',
}

# ==================== 工具函数 ====================
def to_hours(t):
    """把 '2h'/'1d'/纯数字 转小时(float)；失败返回 NaN。"""
    if pd.isna(t):
        return np.nan
    s = str(t).strip().lower()
    if s in TIME_MAP:
        return float(TIME_MAP[s])
    if s.endswith('h'):
        try:
            return float(s[:-1])
        except:
            return np.nan
    if s.endswith('d'):
        try:
            return float(s[:-1]) * 24.0
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

def eff_hours(phase, t):
    """恢复阶段 +168h，否则原小时。"""
    th = to_hours(t)
    if pd.isna(th):
        return np.nan
    return th + 168.0 if str(phase).lower().startswith('recovery_from_') else th

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def safe_num(a):
    """安全转换为 numpy 数组。"""
    try:
        return np.asarray(a, float)
    except Exception:
        return np.full(len(a), np.nan, float)

def sort_by_teff(df):
    """按 phase 和 t_eff_h 排序。"""
    df = df.copy()
    if 't_eff_h' not in df.columns:
        df['t_eff_h'] = [eff_hours(p, t) for p, t in zip(df.get('phase', ''), df.get('time', ''))]
    return df.sort_values(['phase', 't_eff_h'])

def get_r_columns(df):
    """识别光谱列：以 'r_' 或 'R_' 开头，后缀为波长（可带小数）。"""
    cols = []
    for c in df.columns:
        lc = str(c).lower()
        if lc.startswith('r_'):
            try:
                float(str(c)[2:])
                cols.append(c)
            except:
                pass
    return cols

# ==================== 统一样式设置 ====================
def set_matplotlib_style(dpi=300, fontsize_base=11):
    """
    设置统一的 matplotlib 样式（论文级）。
    
    参数：
        dpi: 图像分辨率（默认 300）
        fontsize_base: 基础字体大小（默认 11）
    """
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.size": fontsize_base,
        "axes.titlesize": fontsize_base + 2,
        "axes.labelsize": fontsize_base + 1,
        "legend.fontsize": fontsize_base - 1,
        "xtick.labelsize": fontsize_base,
        "ytick.labelsize": fontsize_base,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.5",
        "legend.borderpad": 0.4,
        "legend.columnspacing": 0.8,
    })

def apply_style_to_ax(ax, grid=True):
    """对单个 axes 应用样式。"""
    if grid:
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def get_phase_color(phase, phase_core=None):
    """根据 phase 或 phase_core 获取颜色。"""
    # 优先使用 phase_core
    if phase_core is not None and str(phase_core) in PHASE_CORE_TO_PHASE:
        phase_key = PHASE_CORE_TO_PHASE[str(phase_core)]
    else:
        phase_key = str(phase).lower()
    
    # 尝试直接匹配
    if phase_key in PHASE_COLORS:
        return PHASE_COLORS[phase_key]
    
    # 模糊匹配
    if 'control' in phase_key or 'ck' in phase_key or '25' in phase_key:
        return PHASE_COLORS['control']
    elif '10' in phase_key or 'cold' in phase_key:
        return PHASE_COLORS['stress_10']
    elif '35' in phase_key or 'heat' in phase_key:
        return PHASE_COLORS['stress_35']
    elif 'recovery' in phase_key:
        if '10' in phase_key or 'cold' in phase_key:
            return PHASE_COLORS['recovery_from_10']
        elif '35' in phase_key or 'heat' in phase_key:
            return PHASE_COLORS['recovery_from_35']
        else:
            return PHASE_COLORS['recovery']
    
    # 默认灰色
    return PHASE_COLORS['control']

# ==================== 数据加载函数 ====================
def load_hsi_tables(images_path, features_path, delta_path, resilience_path=None):
    """
    加载 HSI 相关表格，整理成统一的 DataFrame。
    
    返回：
        dict，包含：
            'images': image_features DataFrame
            'sessions': session_features DataFrame
            'delta': delta_hsi DataFrame
            'resilience': resilience_metrics DataFrame（如果存在）
    """
    result = {}
    
    # 读取 image_features
    try:
        img = pd.read_csv(images_path, sep='\t')
        img.columns = [c.lower() for c in img.columns]
        for col in ['temp', 'time', 'phase', 'session_id']:
            if col not in img.columns:
                img[col] = np.nan
        if 'roi_area' not in img.columns:
            img['roi_area'] = np.nan
        if 'seg_plan' not in img.columns:
            img['seg_plan'] = ''
        if 'r800_med' not in img.columns:
            img['r800_med'] = np.nan
        img = sort_by_teff(img)
        result['images'] = img
        print(f"[quicklook] 加载 image_features: {len(img)} 行")
    except Exception as e:
        print(f"[quicklook][WARN] 无法加载 image_features: {e}")
        result['images'] = pd.DataFrame()
    
    # 读取 session_features
    try:
        sess = pd.read_csv(features_path, sep='\t')
        sess.columns = [c.lower() for c in sess.columns]
        for m in IDX:
            if m not in sess.columns:
                sess[m] = np.nan
            sec = f'{m}_se'
            if sec not in sess.columns:
                sess[sec] = np.nan
        if 'rep_nm' not in sess.columns:
            sess['rep_nm'] = np.nan
        if 'rep_nm_se' not in sess.columns:
            sess['rep_nm_se'] = np.nan
        sess = sort_by_teff(sess)
        result['sessions'] = sess
        print(f"[quicklook] 加载 session_features: {len(sess)} 行")
    except Exception as e:
        print(f"[quicklook][WARN] 无法加载 session_features: {e}")
        result['sessions'] = pd.DataFrame()
    
    # 读取 delta_hsi
    try:
        delt = pd.read_csv(delta_path, sep='\t')
        delt.columns = [c.lower() for c in delt.columns]
        for m in IDX:
            dm = f'd{m}'
            if dm not in delt.columns:
                delt[dm] = np.nan
        for c in ['drep', 'sam', 'euc']:
            if c not in delt.columns:
                delt[c] = np.nan
        delt = sort_by_teff(delt)
        result['delta'] = delt
        print(f"[quicklook] 加载 delta_hsi: {len(delt)} 行")
    except Exception as e:
        print(f"[quicklook][WARN] 无法加载 delta_hsi: {e}")
        result['delta'] = pd.DataFrame()
    
    # 读取 resilience_metrics（可选）
    if resilience_path and Path(resilience_path).exists():
        try:
            resf = pd.read_csv(resilience_path, sep='\t')
            resf.columns = [c.lower() for c in resf.columns]
            for c in ['temp', 'metric', 'resistance', 't_half_h', 'auc_stress', 'auc_recovery', 'resilience', 'hysteresis']:
                if c not in resf.columns:
                    resf[c] = np.nan
            result['resilience'] = resf
            print(f"[quicklook] 加载 resilience_metrics: {len(resf)} 行")
        except Exception as e:
            print(f"[quicklook][WARN] 无法加载 resilience_metrics: {e}")
            result['resilience'] = pd.DataFrame()
    else:
        result['resilience'] = pd.DataFrame()
        print(f"[quicklook] resilience_metrics 不存在或未指定，跳过")
    
    return result

def load_leaf_patch_tables(leaf_path=None, patch_path=None, patch_target='phase_core'):
    """
    加载 leaf 和 patch 级别的特征表，进行聚合。
    
    返回：
        dict，包含：
            'leaf_agg': 叶片级聚合 DataFrame（按 sample_id + phase/time/temp）
            'patch_agg': patch级聚合 DataFrame（按 source_sample_id + patch_target）
    """
    result = {'leaf_agg': None, 'patch_agg': None}
    
    # 加载 leaf_features
    if leaf_path and Path(leaf_path).exists():
        try:
            leaf_df = pd.read_csv(leaf_path, sep='\t')
            leaf_df.columns = [c.lower() for c in leaf_df.columns]
            
            # 确定 sample_id 列
            sample_col = 'sample_id' if 'sample_id' in leaf_df.columns else 'source_sample_id'
            if sample_col not in leaf_df.columns:
                print(f"[quicklook][WARN] leaf_features 缺少 sample_id/source_sample_id，跳过")
                return result
            
            # 聚合键
            agg_keys = [sample_col]
            for key in ['phase', 'time', 'temp', 'phase_core']:
                if key in leaf_df.columns:
                    agg_keys.append(key)
            
            # 聚合指标
            metrics = []
            for m in ['ndvi', 'gndvi', 'pri', 'ari', 'rep_nm']:
                if m in leaf_df.columns:
                    metrics.append(m)
            
            if metrics:
                # 计算分位数：使用更清晰的聚合方式
                agg_dict = {}
                for m in metrics:
                    agg_dict[m] = ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
                
                leaf_agg = leaf_df.groupby(agg_keys, dropna=False)[metrics].agg(agg_dict)
                
                # 重命名列：处理 MultiIndex columns
                new_cols = []
                for col in leaf_agg.columns:
                    if isinstance(col, tuple) and len(col) == 2:
                        metric_name, agg_func = col
                        if callable(agg_func):
                            # 检查是否是分位数函数
                            func_str = str(agg_func)
                            if 'quantile' in func_str and '0.25' in func_str:
                                new_cols.append(f'leaf_{metric_name}_q25')
                            elif 'quantile' in func_str and '0.75' in func_str:
                                new_cols.append(f'leaf_{metric_name}_q75')
                            else:
                                new_cols.append(f'leaf_{metric_name}_median')
                        elif agg_func == 'median':
                            new_cols.append(f'leaf_{metric_name}_median')
                        else:
                            new_cols.append(f'leaf_{metric_name}_{agg_func}')
                    else:
                        new_cols.append(f'leaf_{col}')
                
                leaf_agg.columns = new_cols
                leaf_agg = leaf_agg.reset_index()
                result['leaf_agg'] = leaf_agg
                print(f"[quicklook] 加载并聚合 leaf_features: {len(leaf_agg)} 组")
            else:
                print(f"[quicklook][WARN] leaf_features 中未找到可聚合的指标")
        except Exception as e:
            print(f"[quicklook][WARN] 无法加载 leaf_features: {e}")
    else:
        print(f"[quicklook] leaf_features 未指定或不存在，跳过")
    
    # 加载 patch_features
    if patch_path and Path(patch_path).exists():
        try:
            patch_df = pd.read_csv(patch_path, sep='\t')
            patch_df.columns = [c.lower() for c in patch_df.columns]
            
            # 确定 sample_id 列
            sample_col = 'source_sample_id' if 'source_sample_id' in patch_df.columns else 'sample_id'
            if sample_col not in patch_df.columns:
                print(f"[quicklook][WARN] patch_features 缺少 source_sample_id/sample_id，跳过")
                return result
            
            # 检查 patch_target 列
            if patch_target not in patch_df.columns:
                print(f"[quicklook][WARN] patch_features 缺少列 '{patch_target}'，跳过")
                return result
            
            # 聚合键
            agg_keys = [sample_col, patch_target]
            
            # 聚合指标
            metrics = []
            for m in ['ndvi', 'gndvi', 'pri', 'ari', 'rep_nm']:
                if m in patch_df.columns:
                    metrics.append(m)
            
            if metrics:
                # 计算分位数和计数：使用更清晰的聚合方式
                agg_dict = {}
                for m in metrics:
                    agg_dict[m] = ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
                
                # 添加计数（使用第一个 metric 列来计数）
                count_col = metrics[0] if metrics else sample_col
                
                patch_agg = patch_df.groupby(agg_keys, dropna=False)[metrics].agg(agg_dict)
                
                # 计算 patch 数量
                patch_counts = patch_df.groupby(agg_keys, dropna=False).size().reset_index(name='n_patches_total')
                
                # 重命名列：处理 MultiIndex columns
                new_cols = []
                for col in patch_agg.columns:
                    if isinstance(col, tuple) and len(col) == 2:
                        metric_name, agg_func = col
                        if callable(agg_func):
                            # 检查是否是分位数函数
                            func_str = str(agg_func)
                            if 'quantile' in func_str and '0.25' in func_str:
                                new_cols.append(f'patch_{metric_name}_q25')
                            elif 'quantile' in func_str and '0.75' in func_str:
                                new_cols.append(f'patch_{metric_name}_q75')
                            else:
                                new_cols.append(f'patch_{metric_name}_median')
                        elif agg_func == 'median':
                            new_cols.append(f'patch_{metric_name}_median')
                        else:
                            new_cols.append(f'patch_{metric_name}_{agg_func}')
                    else:
                        new_cols.append(f'patch_{col}')
                
                patch_agg.columns = new_cols
                patch_agg = patch_agg.reset_index()
                
                # 合并计数
                patch_agg = patch_agg.merge(patch_counts, on=agg_keys, how='left')
                
                # 计算高 NDVI patch 数量（如果存在 NDVI 列）
                if 'ndvi' in metrics and 'ndvi' in patch_df.columns:
                    high_ndvi_threshold = 0.7
                    high_ndvi_counts = patch_df[patch_df['ndvi'] > high_ndvi_threshold].groupby(
                        agg_keys, dropna=False
                    ).size().reset_index(name='n_patches_high_ndvi')
                    patch_agg = patch_agg.merge(high_ndvi_counts, on=agg_keys, how='left')
                    patch_agg['n_patches_high_ndvi'] = patch_agg['n_patches_high_ndvi'].fillna(0).astype(int)
                
                result['patch_agg'] = patch_agg
                print(f"[quicklook] 加载并聚合 patch_features: {len(patch_agg)} 组")
            else:
                print(f"[quicklook][WARN] patch_features 中未找到可聚合的指标")
        except Exception as e:
            print(f"[quicklook][WARN] 无法加载 patch_features: {e}")
    else:
        print(f"[quicklook] patch_features 未指定或不存在，跳过")
    
    return result

# ==================== 论文级图像函数 ====================
def make_figure_timeseries_ndvi_rep(sess, out_png, fig_prefix='quicklook', dpi=300):
    """
    Figure 1: NDVI / REP 时序总览
    
    2 行 × 2 列，上面 NDVI，下面 REP
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # NDVI
    ax = axes[0]
    apply_style_to_ax(ax)
    any_data = False
    for ph in CANON_PHASES:
        sub = sess[sess['phase'].str.fullmatch(ph, na=False)].copy()
        if sub.empty:
            continue
        x = safe_num(sub['t_eff_h'])
        y = safe_num(sub['ndvi'])
        se = safe_num(sub.get('ndvi_se', np.nan))
        idx = np.argsort(x)
        x, y, se = x[idx], y[idx], se[idx]
        color = get_phase_color(ph)
        ax.plot(x, y, marker='o', linewidth=2.0, label=ph.replace('_', ' '), color=color, markersize=5)
        if np.isfinite(se).any():
            ax.fill_between(x, y - se, y + se, alpha=0.2, color=color)
        if np.isfinite(y).any():
            any_data = True
    ax.set_ylabel('NDVI (mean ± SE)', fontsize=12)
    ax.set_title('NDVI time series', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, ncol=2, fontsize=9)
    if not any_data:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes, alpha=0.6, fontsize=12)
    
    # REP
    ax = axes[1]
    apply_style_to_ax(ax)
    any_data = False
    for ph in CANON_PHASES:
        sub = sess[sess['phase'].str.fullmatch(ph, na=False)].copy()
        if sub.empty:
            continue
        x = safe_num(sub['t_eff_h'])
        y = safe_num(sub.get('rep_nm', np.nan))
        se = safe_num(sub.get('rep_nm_se', np.nan))
        idx = np.argsort(x)
        x, y, se = x[idx], y[idx], se[idx]
        color = get_phase_color(ph)
        ax.plot(x, y, marker='o', linewidth=2.0, label=ph.replace('_', ' '), color=color, markersize=5)
        if np.isfinite(se).any():
            ax.fill_between(x, y - se, y + se, alpha=0.2, color=color)
        if np.isfinite(y).any():
            any_data = True
    ax.set_xlabel('Effective time (h) [Recovery +168 h]', fontsize=12)
    ax.set_ylabel('REP (nm, mean ± SE)', fontsize=12)
    ax.set_title('REP time series', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, ncol=2, fontsize=9)
    if not any_data:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes, alpha=0.6, fontsize=12)
    
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[quicklook] 保存: {out_png}")

def make_figure_heatmap_delta(delt, metric='dndvi', out_png=None, fig_prefix='quicklook', dpi=300):
    """
    Figure 2: ΔNDVI / ΔREP 热图
    
    使用 diverging colormap（coolwarm），0 在中间
    """
    dcol = metric.lower() if metric.startswith('d') else f'd{metric.lower()}'
    
    sub = delt.copy()
    sub = sub[sub['phase'].isin(CANON_PHASES)]
    if sub.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes, alpha=0.6, fontsize=12)
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return
    
    piv = sub.pivot_table(index='phase', columns='t_eff_h', values=dcol, aggfunc='mean')
    piv = piv.reindex(CANON_PHASES).sort_index(axis=1)
    
    vals = piv.values[np.isfinite(piv.values)]
    if vals.size == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes, alpha=0.6, fontsize=12)
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return
    
    vlim = float(max(np.nanpercentile(np.abs(vals), 90), 1e-6))
    
    fig, ax = plt.subplots(figsize=(12, 4))
    apply_style_to_ax(ax, grid=False)
    
    # 使用 TwoSlopeNorm 确保 0 在中间
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    im = ax.imshow(piv, aspect='auto', cmap='coolwarm', norm=norm, origin='upper', interpolation='nearest')
    
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([p.replace('_', ' ') for p in piv.index], fontsize=10)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([int(x) if float(x).is_integer() else f"{x:.1f}" for x in piv.columns], 
                       rotation=0, fontsize=9)
    ax.set_xlabel('Effective time (h)', fontsize=12)
    ax.set_title(f'Δ{metric.upper().replace("D", "")} heatmap (vs. 25°C)', fontsize=13, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.06)
    cbar.set_label(dcol.upper(), fontsize=10)
    
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[quicklook] 保存: {out_png}")

def make_figure_resilience_bar(resf, out_png, fig_prefix='quicklook', dpi=300):
    """
    Figure 3: 韧性指标条形图
    
    2×2 子图分别展示 resistance, resilience, t_half, hysteresis
    """
    if resf.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'no resilience data', ha='center', va='center', transform=ax.transAxes, alpha=0.6, fontsize=12)
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    metrics_info = [
        ('resistance', 'Resistance\n(-|Δ at 2h|)', 'higher is better'),
        ('resilience', 'Resilience\n(1 - AUC_rec/AUC_str)', 'higher is better'),
        ('t_half_h', 'Half-recovery time\n(t_half, h)', 'lower is better'),
        ('hysteresis', 'Hysteresis\n(AUC_rec - AUC_str)', 'absolute value'),
    ]
    
    for ax, (metric_field, ylabel, note) in zip(axes, metrics_info):
        apply_style_to_ax(ax)
        
        # 按 metric 和 temp 分组
        piv = resf.pivot_table(index='metric', columns='temp', values=metric_field, aggfunc='mean')
        if piv.empty:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes, alpha=0.6)
            continue
        
        piv = piv.loc[[*sorted(piv.index, key=lambda x: str(x))]]
        
        x_pos = np.arange(len(piv.index))
        width = 0.35
        temps = sorted(piv.columns.astype(str).unique())
        
        for i, tp in enumerate(temps):
            offset = (i - (len(temps) - 1) / 2) * width
            vals = piv[tp].values
            bars = ax.bar(x_pos + offset, vals, width, label=f'{tp}°C', alpha=0.8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.upper() for m in piv.index], rotation=0, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'{metric_field.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        ax.legend(title='Temp', frameon=True, fontsize=8, title_fontsize=9)
    
    fig.suptitle('Resilience metrics by phase and temperature', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[quicklook] 保存: {out_png}")

def make_figure_patch_vs_leaf(leaf_agg, patch_agg, out_png, fig_prefix='quicklook', dpi=300, patch_target='phase_core'):
    """
    Figure 4: Patch vs Leaf 级别分布
    
    上面一行：leaf-level NDVI/REP 分布（violin/boxplot）
    下面一行：patch-level NDVI/REP 分布
    """
    if leaf_agg is None and patch_agg is None:
        print(f"[quicklook][WARN] leaf_agg 和 patch_agg 都为空，跳过 patch vs leaf 图")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Leaf-level NDVI
    ax = axes[0]
    apply_style_to_ax(ax)
    if leaf_agg is not None and 'leaf_ndvi_median' in leaf_agg.columns:
        phase_col = 'phase_core' if 'phase_core' in leaf_agg.columns else 'phase'
        if phase_col in leaf_agg.columns:
            phases = leaf_agg[phase_col].unique()
            data_list = []
            labels = []
            for ph in phases:
                sub = leaf_agg[leaf_agg[phase_col] == ph]['leaf_ndvi_median'].dropna()
                if len(sub) > 0:
                    data_list.append(sub.values)
                    labels.append(str(ph))
            
            if data_list:
                bp = ax.boxplot(data_list, labels=labels, patch_artist=True, showfliers=False)
                for patch, ph in zip(bp['boxes'], labels):
                    patch.set_facecolor(get_phase_color(ph))
                    patch.set_alpha(0.7)
    ax.set_ylabel('NDVI (median)', fontsize=11)
    ax.set_title('Leaf-level NDVI distribution', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    # Leaf-level REP
    ax = axes[1]
    apply_style_to_ax(ax)
    if leaf_agg is not None and 'leaf_rep_nm_median' in leaf_agg.columns:
        phase_col = 'phase_core' if 'phase_core' in leaf_agg.columns else 'phase'
        if phase_col in leaf_agg.columns:
            phases = leaf_agg[phase_col].unique()
            data_list = []
            labels = []
            for ph in phases:
                sub = leaf_agg[leaf_agg[phase_col] == ph]['leaf_rep_nm_median'].dropna()
                if len(sub) > 0:
                    data_list.append(sub.values)
                    labels.append(str(ph))
            
            if data_list:
                bp = ax.boxplot(data_list, labels=labels, patch_artist=True, showfliers=False)
                for patch, ph in zip(bp['boxes'], labels):
                    patch.set_facecolor(get_phase_color(ph))
                    patch.set_alpha(0.7)
    ax.set_ylabel('REP (nm, median)', fontsize=11)
    ax.set_title('Leaf-level REP distribution', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    # Patch-level NDVI
    ax = axes[2]
    apply_style_to_ax(ax)
    if patch_agg is not None and 'patch_ndvi_median' in patch_agg.columns and patch_target in patch_agg.columns:
        phases = patch_agg[patch_target].unique()
        data_list = []
        labels = []
        for ph in phases:
            sub = patch_agg[patch_agg[patch_target] == ph]['patch_ndvi_median'].dropna()
            if len(sub) > 0:
                data_list.append(sub.values)
                labels.append(str(ph))
        
        if data_list:
            bp = ax.boxplot(data_list, labels=labels, patch_artist=True, showfliers=False)
            for patch, ph in zip(bp['boxes'], labels):
                patch.set_facecolor(get_phase_color(ph))
                patch.set_alpha(0.7)
    ax.set_xlabel(f'{patch_target.replace("_", " ").title()}', fontsize=11)
    ax.set_ylabel('NDVI (median)', fontsize=11)
    ax.set_title('Patch-level NDVI distribution', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    # Patch-level REP
    ax = axes[3]
    apply_style_to_ax(ax)
    if patch_agg is not None and 'patch_rep_nm_median' in patch_agg.columns and patch_target in patch_agg.columns:
        phases = patch_agg[patch_target].unique()
        data_list = []
        labels = []
        for ph in phases:
            sub = patch_agg[patch_agg[patch_target] == ph]['patch_rep_nm_median'].dropna()
            if len(sub) > 0:
                data_list.append(sub.values)
                labels.append(str(ph))
        
        if data_list:
            bp = ax.boxplot(data_list, labels=labels, patch_artist=True, showfliers=False)
            for patch, ph in zip(bp['boxes'], labels):
                patch.set_facecolor(get_phase_color(ph))
                patch.set_alpha(0.7)
    ax.set_xlabel(f'{patch_target.replace("_", " ").title()}', fontsize=11)
    ax.set_ylabel('REP (nm, median)', fontsize=11)
    ax.set_title('Patch-level REP distribution', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    fig.suptitle('Patch vs Leaf level distribution comparison', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[quicklook] 保存: {out_png}")

def make_figure_patch_cnn_metrics(out_png, fig_prefix='quicklook', dpi=300):
    """
    Figure 5: Patch CNN 分类结果 quicklook（预留接口）
    
    支持加载 results/hsi/patch/2dcnn_{target}/metrics.tsv 与 3dcnn 对应 metrics
    """
    # 预留接口，暂时生成占位图
    fig, ax = plt.subplots(figsize=(8, 6))
    apply_style_to_ax(ax)
    ax.text(0.5, 0.5, 'Patch CNN metrics\n(interface reserved for future extension)', 
            ha='center', va='center', transform=ax.transAxes, alpha=0.6, fontsize=12)
    ax.set_title('Patch CNN classification metrics', fontsize=13, fontweight='bold')
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[quicklook] 保存（占位）: {out_png}")

# ==================== 多组学桥接表生成 ====================
def make_omics_bridge_table(hsi_data, leaf_agg=None, patch_agg=None, patch_target='phase_core', out_path=None):
    """
    生成多组学联动桥接表。
    
    输出字段：
        session_id, phase, phase_core, time, temp
        HSI summary: ndvi_med, ndvi_se, rep_med, rep_se, dndvi_vs25, drep_vs25
        resilience: resistance, resilience, t_half, hysteresis
        Leaf 级聚合（如果有）: leaf_ndvi_q25/q50/q75, leaf_rep_q25/q50/q75
        Patch 级聚合（如果有）: patch_ndvi_q25/q50/q75, n_patches_total
    """
    bridge_rows = []
    
    # 从 session_features 构建基础行
    sess = hsi_data.get('sessions', pd.DataFrame())
    if not sess.empty:
        for _, row in sess.iterrows():
            rec = {
                'session_id': row.get('session_id', ''),
                'phase': row.get('phase', ''),
                'phase_core': row.get('phase_core', row.get('phase', '')),
                'time': row.get('time', ''),
                'temp': row.get('temp', ''),
                'ndvi_med': row.get('ndvi', np.nan),
                'ndvi_se': row.get('ndvi_se', np.nan),
                'rep_med': row.get('rep_nm', np.nan),
                'rep_se': row.get('rep_nm_se', np.nan),
            }
            bridge_rows.append(rec)
    
    # 从 delta_hsi 添加 dNDVI 和 dREP
    delt = hsi_data.get('delta', pd.DataFrame())
    if not delt.empty:
        for rec in bridge_rows:
            sess_id = rec.get('session_id', '')
            phase = rec.get('phase', '')
            time = rec.get('time', '')
            
            # 匹配 delta 行
            match = delt[
                (delt.get('session_id', '') == sess_id) &
                (delt.get('phase', '') == phase) &
                (delt.get('time', '') == time)
            ]
            if not match.empty:
                rec['dndvi_vs25'] = match.iloc[0].get('dndvi', np.nan)
                rec['drep_vs25'] = match.iloc[0].get('drep', np.nan)
            else:
                rec['dndvi_vs25'] = np.nan
                rec['drep_vs25'] = np.nan
    
    # 从 resilience_metrics 添加韧性指标
    resf = hsi_data.get('resilience', pd.DataFrame())
    if not resf.empty:
        for rec in bridge_rows:
            phase = rec.get('phase', '')
            temp = rec.get('temp', '')
            
            # 按 metric 聚合（取第一个 metric 或平均）
            match = resf[
                (resf.get('phase', '') == phase) &
                (resf.get('temp', '').astype(str) == str(temp))
            ]
            if not match.empty:
                rec['resistance'] = match['resistance'].mean() if 'resistance' in match.columns else np.nan
                rec['resilience'] = match['resilience'].mean() if 'resilience' in match.columns else np.nan
                rec['t_half'] = match['t_half_h'].mean() if 't_half_h' in match.columns else np.nan
                rec['hysteresis'] = match['hysteresis'].mean() if 'hysteresis' in match.columns else np.nan
            else:
                rec['resistance'] = np.nan
                rec['resilience'] = np.nan
                rec['t_half'] = np.nan
                rec['hysteresis'] = np.nan
    
    # 构建基础 DataFrame
    if not bridge_rows:
        print(f"[quicklook][WARN] 没有 session 数据，无法生成桥接表")
        bridge_df = pd.DataFrame()
    else:
        bridge_df = pd.DataFrame(bridge_rows)
    
    # 如果 bridge_df 为空，直接返回
    if bridge_df.empty:
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            bridge_df.to_csv(out_path, sep='\t', index=False)
            print(f"[quicklook] 保存多组学桥接表（空）: {out_path}")
        return bridge_df
    
    # 合并 leaf 级聚合
    if leaf_agg is not None and not leaf_agg.empty:
        # 确定合并键（优先使用 session_id，否则使用 phase/time/temp 组合）
        merge_keys = []
        if 'session_id' in bridge_df.columns and 'session_id' in leaf_agg.columns:
            merge_keys.append('session_id')
        for key in ['phase', 'time', 'temp', 'phase_core']:
            if key in bridge_df.columns and key in leaf_agg.columns:
                merge_keys.append(key)
        
        if merge_keys:
            bridge_df = bridge_df.merge(leaf_agg, on=merge_keys, how='left', suffixes=('', '_leaf'))
        else:
            print(f"[quicklook][WARN] 无法确定 leaf_agg 的合并键，跳过合并")
    
    # 合并 patch 级聚合
    if patch_agg is not None and not patch_agg.empty:
        # 确定合并键
        merge_keys = []
        
        # 尝试匹配 session_id 或 source_sample_id
        if 'session_id' in bridge_df.columns:
            if 'source_sample_id' in patch_agg.columns:
                # 创建临时列用于合并
                bridge_df_temp = bridge_df.copy()
                bridge_df_temp['source_sample_id'] = bridge_df_temp['session_id']
                merge_keys.append('source_sample_id')
            elif 'sample_id' in patch_agg.columns:
                bridge_df_temp = bridge_df.copy()
                bridge_df_temp['sample_id'] = bridge_df_temp['session_id']
                merge_keys.append('sample_id')
        elif 'source_sample_id' in bridge_df.columns:
            if 'source_sample_id' in patch_agg.columns:
                merge_keys.append('source_sample_id')
            elif 'sample_id' in patch_agg.columns:
                bridge_df_temp = bridge_df.copy()
                bridge_df_temp['sample_id'] = bridge_df_temp['source_sample_id']
                merge_keys.append('sample_id')
        
        # 添加 patch_target 作为合并键
        if patch_target in bridge_df.columns and patch_target in patch_agg.columns:
            merge_keys.append(patch_target)
        elif 'phase_core' in bridge_df.columns and patch_target in patch_agg.columns:
            # 尝试用 phase_core 匹配 patch_target
            bridge_df_temp = bridge_df.copy()
            bridge_df_temp[patch_target] = bridge_df_temp['phase_core']
            merge_keys.append(patch_target)
        
        if merge_keys:
            if 'bridge_df_temp' in locals():
                bridge_df = bridge_df_temp.merge(patch_agg, on=merge_keys, how='left', suffixes=('', '_patch'))
            else:
                bridge_df = bridge_df.merge(patch_agg, on=merge_keys, how='left', suffixes=('', '_patch'))
        else:
            print(f"[quicklook][WARN] 无法确定 patch_agg 的合并键，跳过合并")
    
    # 构建最终 DataFrame
    bridge_df = pd.DataFrame(bridge_rows)
    
    # 保存
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        bridge_df.to_csv(out_path, sep='\t', index=False)
        print(f"[quicklook] 保存多组学桥接表: {out_path} ({len(bridge_df)} 行)")
    
    return bridge_df

# ==================== 向后兼容的原有绘图函数 ====================
# 保留原有的绘图函数，确保向后兼容
def plot_metric_timeseries(sess, metric, out_png, dpi=300):
    """原有函数：单个指标的时序图"""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    apply_style_to_ax(ax)
    title = metric.upper()
    for ph in CANON_PHASES:
        sub = sess[sess['phase'].str.fullmatch(ph, na=False)].copy()
        if sub.empty:
            continue
        x = safe_num(sub['t_eff_h'])
        y = safe_num(sub[metric])
        se = safe_num(sub.get(f'{metric}_se', np.nan))
        idx = np.argsort(x)
        x, y, se = x[idx], y[idx], se[idx]
        color = get_phase_color(ph)
        ax.plot(x, y, marker='o', linewidth=2.0, label=ph.replace('_', ' '), color=color)
        if np.isfinite(se).any():
            ax.fill_between(x, y - se, y + se, alpha=0.20, color=color)
    ax.set_xlabel('Effective time (h) [Recovery +168 h]')
    ax.set_ylabel(f'{title} (mean ± SE)')
    ax.set_title(f'{title} time series (session means)')
    ax.legend(ncols=2, frameon=True)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def plot_delta_heatmap_single(delt, dcol, title, out_png, dpi=300):
    """原有函数：单个 Δ 指标的热图"""
    make_figure_heatmap_delta(delt, dcol.replace('d', ''), out_png, dpi=dpi)

def plot_drep_timeseries(delt, out_png, dpi=300):
    """原有函数：dREP 时序图"""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    apply_style_to_ax(ax)
    any_data = False
    for ph in CANON_PHASES:
        sub = delt[delt['phase'].str.fullmatch(ph, na=False)]
        if sub.empty:
            continue
        x = safe_num(sub['t_eff_h'])
        y = safe_num(sub['drep'])
        if np.isfinite(y).any():
            any_data = True
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        color = get_phase_color(ph)
        ax.plot(x, y, marker='o', linewidth=2.0, label=ph.replace('_', ' '), color=color)
    ax.axhline(0.0, color='k', linewidth=1.0, alpha=0.6)
    ax.set_xlabel('Effective time (h)')
    ax.set_ylabel('Δ REP (nm)')
    ax.set_title('dREP time series (vs. 25°C control)')
    ax.legend(ncols=2, frameon=True)
    if not any_data:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes, alpha=0.6)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

# ==================== 主程序 ====================
def main():
    ap = argparse.ArgumentParser(description='HSI quicklook - 论文级可视化与多组学桥接')
    
    # 原有必需参数
    ap.add_argument('--images', required=True, help='results/hsi/image_features.tsv')
    ap.add_argument('--features', required=True, help='results/hsi/session_features.tsv')
    ap.add_argument('--delta', required=True, help='results/hsi/delta_hsi.tsv')
    ap.add_argument('--resilience', required=False, default='results/hsi/resilience_metrics.tsv',
                    help='results/hsi/resilience_metrics.tsv（可选）')
    
    # 新增可选参数
    ap.add_argument('--leaf-features', default=None,
                    help='results/hsi/leaf_features.tsv（可选）')
    ap.add_argument('--patch-features', default=None,
                    help='results/hsi/patch_features.tsv（可选）')
    ap.add_argument('--patch-target', default='phase_core',
                    help='patch_features 里的标签列名（默认: phase_core）')
    
    # 输出控制
    ap.add_argument('--outdir', default='results/hsi', help='输出目录')
    ap.add_argument('--fig-prefix', default='quicklook', help='图像文件名前缀（默认: quicklook）')
    ap.add_argument('--dpi', type=int, default=300, help='图像分辨率（默认: 300）')
    ap.add_argument('--omics-join-out', default='results/hsi/quicklook_omics_bridge.tsv',
                    help='多组学桥接表输出路径（默认: quicklook_omics_bridge.tsv）')
    
    # 向后兼容的"摆设"参数
    ap.add_argument('--specdir', default='', help='（向后兼容，未使用）')
    ap.add_argument('--delta-mode', default='', help='（向后兼容，未使用）')
    
    args = ap.parse_args()

    # 设置样式
    set_matplotlib_style(dpi=args.dpi)
    outdir = ensure_dir(args.outdir)

    print(f"[quicklook] 开始处理...")
    print(f"[quicklook] 输出目录: {outdir}")
    print(f"[quicklook] 图像前缀: {args.fig_prefix}, DPI: {args.dpi}")
    
    # 1. 加载 HSI 表格
    hsi_data = load_hsi_tables(
        args.images,
        args.features,
        args.delta,
        args.resilience if args.resilience else None
    )
    
    # 2. 加载 leaf/patch 表格（可选）
    leaf_patch_data = load_leaf_patch_tables(
        args.leaf_features,
        args.patch_features,
        args.patch_target
    )
    
    # 3. 生成论文级图像
    sess = hsi_data.get('sessions', pd.DataFrame())
    delt = hsi_data.get('delta', pd.DataFrame())
    resf = hsi_data.get('resilience', pd.DataFrame())
    
    # Figure 1: NDVI/REP 时序总览
    if not sess.empty:
        make_figure_timeseries_ndvi_rep(
            sess,
            outdir / f'{args.fig_prefix}_timeseries_ndvi_rep.png',
            fig_prefix=args.fig_prefix,
            dpi=args.dpi
        )
    
    # Figure 2: Δ 热图
    if not delt.empty:
        make_figure_heatmap_delta(
            delt, 'dndvi',
            outdir / f'{args.fig_prefix}_heatmap_dndvi.png',
            fig_prefix=args.fig_prefix,
            dpi=args.dpi
        )
        make_figure_heatmap_delta(
            delt, 'drep',
            outdir / f'{args.fig_prefix}_heatmap_drep.png',
            fig_prefix=args.fig_prefix,
            dpi=args.dpi
        )
    
    # Figure 3: 韧性指标条形图
    if not resf.empty:
        make_figure_resilience_bar(
            resf,
            outdir / f'{args.fig_prefix}_resilience_bar.png',
            fig_prefix=args.fig_prefix,
            dpi=args.dpi
        )
    
    # Figure 4: Patch vs Leaf 分布（如果提供了数据）
    if leaf_patch_data['leaf_agg'] is not None or leaf_patch_data['patch_agg'] is not None:
        make_figure_patch_vs_leaf(
            leaf_patch_data['leaf_agg'],
            leaf_patch_data['patch_agg'],
            outdir / f'{args.fig_prefix}_patch_vs_leaf_dist.png',
            fig_prefix=args.fig_prefix,
            dpi=args.dpi,
            patch_target=args.patch_target
        )
    
    # Figure 5: Patch CNN metrics（预留接口）
    make_figure_patch_cnn_metrics(
        outdir / f'{args.fig_prefix}_patch_cnn_metrics.png',
        fig_prefix=args.fig_prefix,
        dpi=args.dpi
    )
    
    # 4. 生成多组学桥接表
    bridge_df = make_omics_bridge_table(
        hsi_data,
        leaf_patch_data.get('leaf_agg'),
        leaf_patch_data.get('patch_agg'),
        args.patch_target,
        args.omics_join_out
    )
    
    # 5. 向后兼容：生成原有的输出（保持路径不变）
    print(f"[quicklook] 生成向后兼容的输出...")
    if not sess.empty:
        for m in IDX:
            plot_metric_timeseries(sess, m, outdir / f'plot_{m}_timeseries.png', dpi=args.dpi)
    
    if not delt.empty:
        plot_drep_timeseries(delt, outdir / 'plot_drep_timeseries.png', dpi=args.dpi)
        for m in IDX:
            plot_delta_heatmap_single(
                delt, f'd{m}',
                f'Δ{m.upper()} heatmap (vs. 25°C)',
                outdir / f'plot_d{m}_heatmap.png',
                dpi=args.dpi
            )
    
    print(f"[quicklook] 完成！所有输出已保存到: {outdir.resolve()}")
    if not bridge_df.empty:
        print(f"[quicklook] 多组学桥接表: {args.omics_join_out} ({len(bridge_df)} 行)")


if __name__ == '__main__':
    main()
