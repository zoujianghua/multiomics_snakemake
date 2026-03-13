#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSI quicklook (fixed): distinct colors for 10/35 stress & recovery; robust dNDVI baseline; optional index pack
Inputs:
- results/hsi/delta_hsi.tsv (columns: session?, temp, phase, time, ndvi[, dndvi, rep_nm, drep, ...])
- results/hsi/session_features.tsv (to infer recovery origin if needed)
Outputs:
- results/hsi/plot_ndvi_timeseries.png
- results/hsi/plot_drep_timeseries.png
- results/hsi/plot_dndvi_heatmap.png
- results/hsi/plot_all_sessions_overlay.png
- optional: results/hsi/indices/{index}.png & indices.tsv
"""
import argparse, re
"35C_stress": ("#1f77b4","o","-"),
"10C_recovery": ("#ff9896","^","--"),
"35C_recovery": ("#98c1ff","^","--"),
}


IDX_DEF = {
# name: (lambda R: expr, required wavelengths (nm) or tuple keys)
"NDVI": lambda R: (R(860)-R(670))/(R(860)+R(670)+1e-9),
"GNDVI": lambda R: (R(860)-R(550))/(R(860)+R(550)+1e-9),
"NDRE": lambda R: (R(780)-R(705))/(R(780)+R(705)+1e-9),
"EVI2": lambda R: 2.5*(R(860)-R(670))/(R(860)+2.4*R(670)+1e-9),
"SAVI": lambda R: 1.5*(R(860)-R(670))/(R(860)+R(670)+0.5),
"OSAVI": lambda R: 1.16*(R(860)-R(670))/(R(860)+R(670)+0.16),
"MSR": lambda R: (R(860)/R(670)-1)/np.sqrt(R(860)/R(670)+1e-9),
"RDVI": lambda R: (R(860)-R(670))/np.sqrt(R(860)+R(670)+1e-9),
"CIrededge": lambda R: R(860)/R(705) - 1,
"MTCI": lambda R: (R(754)-R(709))/((R(709)-R(681))+1e-9),
"PRI": lambda R: (R(531)-R(570))/(R(531)+R(570)+1e-9),
"MCARI2": lambda R: 1.5*(2.5*(R(750)-R(705)) - 1.3*(R(750)-R(550))) / np.sqrt((2*R(750)+1)**2 - (6*R(750)-5*np.sqrt(R(705))) - 0.5),
}




def to_hours(x: str) -> float:
s = str(x).strip().lower()
return TIME_MAP.get(s, np.nan)




def nearest(wl: np.ndarray, target: float) -> int:
return int(np.argmin(np.abs(wl - target)))




def load_tables(delta_path: Path, feat_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
df = pd.read_csv(delta_path, sep=" " if delta_path.suffix==".tsv" else ",")
df.columns = [c.strip().lower() for c in df.columns]
for col in ("temp","phase","time"):
if col not in df.columns:
raise SystemExit(f"[ERR] missing column '{col}' in {delta_path}")
# Normalize phase
df["phase"] = df["phase"].astype(str).str.lower().map(lambda s: PHASE_KEYS.get(s, s))
# Uniform temp
df["temp"] = df["temp"].astype(str).str.upper().str.replace("℃","C")
# numeric time for plotting
df["time_h"] = df["time"].apply(to_hours)


# session features (optional; used to infer recovery origin)
sf = None
if feat_path.exists():
sf = pd.read_csv(feat_path, sep=" " if feat_path.suffix==".tsv" else ",")
sf.columns = [c.strip().lower() for c in sf.columns]
return df, sf

def infer_recovery_origin(df: pd.DataFrame, sf: pd.DataFrame|None) -> pd.Series:
if cand:
key = cand[0]
# guess stress temperature per session from sf
temp_col = "temp" if "temp" in sf.columns else None
phase_col = "phase" if "phase" in sf.columns else None
if temp_col and phase_col:
stress_temp = sf.loc[sf[phase_col].astype(str).str.lower().str.contains("stress"), [key,temp_col]].drop_duplicates()
stress_temp[temp_col] = stress_temp[temp_col].astype(str).str.upper().str.replace("℃","C")
st_map = dict(zip(stress_temp[key], stress_temp[temp_col]))
mask = (df["phase"]=="recovery") & df[key].notna()
origin.loc[mask] = df.loc[mask, key].map(st_map).fillna("")


# 3) Any leftovers: leave empty → will be labeled as '25→25 recovery (unknown)' but plotted separately by session if present
return origin




def recompute_deltas(df: pd.DataFrame, delta_mode: str) -> pd.DataFrame:
df = df.copy()
if "ndvi" not in df.columns:
return df
# Build control (25C_control) profile by time
ctrl_mask = (df["phase"]=="control") & (df["temp"].isin(["25C","25°","25"]))
ctrl = df.loc[ctrl_mask, ["time_h","ndvi"]].dropna()
ctrl = ctrl.groupby("time_h", as_index=False).median()
ctrl_map = dict(zip(ctrl["time_h"], ctrl["ndvi"]))


def dndvi_vs25(row):
base = ctrl_map.get(row["time_h"], np.nan)
return row["ndvi"] - base if np.isfinite(base) else np.nan


if delta_mode == "vs25":
df["dndvi_recalc"] = df.apply(dndvi_vs25, axis=1)
elif delta_mode == "vs_baseline":
# baseline = first stress time for that temperature (e.g., 2h at 10C/35C); control rows get 0
base_by_temp = {}
for t in ["10C","35C","25C"]:
sub = df[(df["temp"]==t) & (df["phase"]==("stress" if t!="25C" else "control")) & df["ndvi"].notna()]
if not sub.empty:
base_by_temp[t] = sub.sort_values("time_h").iloc[0]["ndvi"]
def f(row):
base = base_by_temp.get(row["temp"], np.nan)
return row["ndvi"] - base if np.isfinite(base) else np.nan
df["dndvi_recalc"] = df.apply(f, axis=1)
else:
# auto: if existing dndvi present, keep; else fallback to vs25
if "dndvi" in df.columns:
df["dndvi_recalc"] = df["dndvi"]
else:
df["dndvi_recalc"] = df.apply(dndvi_vs25, axis=1)
return df

def plot_timeseries(df: pd.DataFrame, ycol: str, out_png: Path, title: str):
xticks = [2,24,72,168, 168+6,168+24,168+72,168+168]
xlabels = ["2h","1d","3d","7d","R-6h","R-1d","R-3d","R-7d"]
ax.set_xticks(xticks, xlabels, rotation=0)


ax.set_title(title)
ax.set_xlabel("Time")
ax.set_ylabel(ycol)
ax.legend(loc="best", fontsize=8, ncol=2, frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png)
plt.close(fig)




def plot_dndvi_heatmap(df: pd.DataFrame, out_png: Path):
# Two panels: stress & recovery; shared color scale symmetric by robust quantile
import matplotlib.pyplot as plt
import matplotlib as mpl


stress = df[df["phase"]=="stress"][["temp","time_h","dndvi_recalc"]]
reco = df[df["phase"]=="recovery"][ ["recovery_from","time_h","dndvi_recalc"]].rename(columns={"recovery_from":"temp"})


def pivot_Z(dd):
if dd.empty:
return None, None, None
piv = dd.pivot_table(index="temp", columns="time_h", values="dndvi_recalc", aggfunc=np.nanmedian)
temps = list(piv.index)
times = sorted(piv.columns)
Z = piv.loc[temps, times].values
return temps, times, Z


rows = 1; cols = 2
fig, axes = plt.subplots(rows, cols, figsize=(10,4.2), dpi=160, constrained_layout=True)
v = df["dndvi_recalc"].replace([np.inf,-np.inf], np.nan).dropna()
vmax = np.nanpercentile(np.abs(v), 95)
norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
cmap = mpl.cm.RdBu_r


for ax, dd, title in zip(axes, (stress, reco), ("dNDVI — stress", "dNDVI — recovery")):
temps, times, Z = pivot_Z(dd)
if Z is None:
ax.axis("off"); continue
im = ax.imshow(Z, aspect="auto", cmap=cmap, norm=norm, origin="lower")
ax.set_yticks(range(len(temps)), temps)
# label recovery times as R-*
if "recovery" in title:
labels = [f"R-{int(t) if t<24 else (str(int(t/24))+ 'd')}" if t in (6,24,72,168) else str(t) for t in times]
else:
labels = ["2h" if t==2 else ("1d" if t==24 else ("3d" if t==72 else ("7d" if t==168 else str(t)))) for t in times]
ax.set_xticks(range(len(times)), labels, rotation=0)
ax.set_title(title)
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), shrink=0.9)
cbar.set_label("ΔNDVI (relative to 25C control)")
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png)
plt.close(fig)

def compute_indices(session_spec_dir: Path, outdir: Path):
Rarr = np.asarray(data["spec_med"]).astype(float)
def R(lam):
i = nearest(wl, float(lam))
return Rarr[i]
vals = {name: float(func(R)) for name, func in IDX_DEF.items()}
# Try to parse meta from filename (fallbacks)
sid = npz.stem
m = re.search(r"(?i)(10c|35c|25c).*?(stress|recovery|control)", sid)
temp = m.group(1).upper() if m else ""
phase = m.group(2).lower() if m else ""
idx_rows.append({"session":sid, "temp":temp, "phase":phase, **vals})
if not idx_rows:
return
df = pd.DataFrame(idx_rows)
out_tsv = outdir/"indices"/"indices.tsv"
out_tsv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_tsv, sep=" ", index=False)
# Plot a panel of selected indices
pick = ["NDVI","NDRE","GNDVI","EVI2","SAVI","CIrededge","PRI","MCARI2"]
for k in pick:
fig, ax = plt.subplots(figsize=(7,4), dpi=150)
for lab, g in df.groupby(["temp","phase"]):
lab_str = f"{lab[0]}_{lab[1]}".strip("_")
color, marker, ls = PALETTE.get(lab_str, ("#7f7f7f","o","-"))
ax.plot(range(len(g)), g[k], marker=marker, ls=ls, label=lab_str, lw=1.4)
ax.set_title(k)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(outdir/"indices"/f"plot_{k}.png"); plt.close(fig)




def main():
ap = argparse.ArgumentParser()
ap.add_argument("--delta", dest="delta_path", default="results/hsi/delta_hsi.tsv")
ap.add_argument("--features", dest="feat_path", default="results/hsi/session_features.tsv")
ap.add_argument("--specdir", dest="spec_dir", default="results/hsi/session_spectra")
ap.add_argument("--delta-mode", choices=["vs25","vs_baseline","auto"], default="vs25")
ap.add_argument("--with-indices", action="store_true")
args = ap.parse_args()


delta_path = Path(args.delta_path)
feat_path = Path(args.feat_path)
spec_dir = Path(args.spec_dir)


df, sf = load_tables(delta_path, feat_path)


# infer recovery origin per row
df["recovery_from"] = infer_recovery_origin(df, sf)


# recompute or adopt dNDVI
df = recompute_deltas(df, args.delta_mode)


# Subsets for timeseries
for ycol, outname, title in (
("ndvi", "plot_ndvi_timeseries.png", "NDVI timeseries"),
("rep_nm" if "rep_nm" in df.columns else "rep", "plot_drep_timeseries.png", "Red‑edge position (REP) timeseries"),
):
if ycol in df.columns:
plot_timeseries(df[df[ycol].notna()], ycol, Path("results/hsi")/outname, title)


# Heatmap for dNDVI
if "dndvi_recalc" in df.columns:
plot_dndvi_heatmap(df, Path("results/hsi")/"plot_dndvi_heatmap.png")


# Optional index pack
if args.with_indices:
compute_indices(spec_dir, Path("results/hsi"))


if __name__ == "__main__":
main()
