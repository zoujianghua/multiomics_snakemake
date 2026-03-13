#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 config/samples_rnaseq.csv 生成 config/rnaseq_design.tsv
——与代谢组设计表对齐：使用“精确 phase”，并让 group == phase

命名样例及解析：
  CK25_T2h_R1  -> phase=control_25_T2h,         temp=25,  time=2h,  t_eff_h=2
  HT35_T3d_R2  -> phase=stress_35_T3d,          temp=35,  time=3d,  t_eff_h=72
  LT10_T7d_R3  -> phase=stress_10_T7d,          temp=10,  time=7d,  t_eff_h=168
  HTR_T6h_R1   -> phase=recovery_from_35_T6h,   temp=25,  time=6h,  t_eff_h=174(+168)
  LTR_T3d_R2   -> phase=recovery_from_10_T3d,   temp=25,  time=3d,  t_eff_h=240(+168)

输出列（顺序固定）：
  sample, phase(精确), group(=phase), time, time_h, t_eff_h, temp, replicate, region, batch, rin
"""
from pathlib import Path
import re, sys
import pandas as pd


def _io_paths():
    if "snakemake" in globals():
        in_csv = Path(snakemake.input[0])     # type: ignore
        out_tsv = Path(snakemake.output[0])   # type: ignore
    else:
        if len(sys.argv) < 3:
            print("Usage: python make_rnaseq_design.py <in.csv> <out.tsv>")
            sys.exit(2)
        in_csv, out_tsv = Path(sys.argv[1]), Path(sys.argv[2])
    if (not in_csv.exists()) or in_csv.stat().st_size == 0:
        raise FileNotFoundError(f"样本表不存在或为空: {in_csv}")
    return in_csv, out_tsv


def to_hours(time_str: str) -> float:
    s = str(time_str).strip().lower()
    m = re.match(r'^(\d+)([hd])$', s)
    if not m:
        return float('nan')
    v = float(m.group(1))
    return v if m.group(2) == 'h' else v * 24.0


def parse_precise_phase(sample_id: str):
    """
    解析 sample_id -> 精确 phase、temp、time、time_h、t_eff_h、replicate
    允许无 R 后缀（replicate 置空）
    """
    up = str(sample_id).strip().upper()
    m = re.match(
        r'^(?P<prefix>CK|HTR|LTR|HT|LT)'
        r'(?P<deg>\d+)?'
        r'[_-]T(?P<tval>\d+)(?P<tunit>[HD])'
        r'(?:[_-]R(?P<rep>\d+))?$', up
    )
    if not m:
        # 尽量给出可用字段，默认当作 control_25_T0h
        return {
            "phase": "control_25_T0h",
            "temp": 25,
            "time": "0h",
            "time_h": 0.0,
            "t_eff_h": 0.0,
            "replicate": ""
        }

    prefix = m.group("prefix")
    deg    = (m.group("deg") or "").strip()
    tval   = m.group("tval")
    tunit  = m.group("tunit").lower()
    time   = f"{tval}{tunit}"
    time_h = to_hours(time)
    rep    = ("R" + m.group("rep")) if m.group("rep") else ""

    # 映射 prefix & 生成精确 phase
    if prefix == "CK":
        temp = int(deg) if deg else 25
        phase_core = f"control_{temp}"
    elif prefix == "HTR":
        temp = 25
        phase_core = "recovery_from_35"
    elif prefix == "LTR":
        temp = 25
        phase_core = "recovery_from_10"
    elif prefix == "HT":
        temp = int(deg) if deg else 35
        phase_core = f"stress_{temp}"
    elif prefix == "LT":
        temp = int(deg) if deg else 10
        phase_core = f"stress_{temp}"
    else:
        temp = 25
        phase_core = f"control_{temp}"

    phase = f"{phase_core}_T{time}"
    is_recovery = phase_core.startswith("recovery_from_")
    t_eff_h = (time_h + 168.0) if is_recovery else time_h

    return {
        "phase": phase,
        "temp": temp,
        "time": time,
        "time_h": time_h,
        "t_eff_h": t_eff_h,
        "replicate": rep
    }


def main():
    in_csv, out_tsv = _io_paths()
    df = pd.read_csv(in_csv)

    # 兜底列
    for c in ["sample_id", "region", "batch", "rin"]:
        if c not in df.columns:
            df[c] = ""

    df["sample"] = df["sample_id"].astype(str).str.strip()

    rows = []
    for _, r in df.iterrows():
        sid = str(r["sample"])
        meta = parse_precise_phase(sid)
        rows.append({
            "sample": sid,
            "phase": meta["phase"],          # 精确标签（含温度与时间）
            "group": meta["phase"],          # == phase（对齐代谢组设计）
            "time": meta["time"],
            "time_h": meta["time_h"],
            "t_eff_h": meta["t_eff_h"],      # 恢复段 +168h
            "temp": meta["temp"],            # 数字（25/35/10/…）
            "replicate": meta["replicate"],  # 可空
            "region": str(r.get("region", "")),
            "batch": str(r.get("batch", "")),
            "rin": str(r.get("rin", "")),
        })

    design = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    if design.empty:
        raise RuntimeError("生成的 design 为空，请检查样本命名是否符合规则。")

    # 固定列顺序，尽可能与代谢组设计表一致（无 file/is_qc/number_id）
    cols = ["sample","phase","group","time","time_h","t_eff_h","temp",
            "replicate","region","batch","rin"]
    design = design.reindex(columns=cols)

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(out_tsv, sep="\t", index=False)
    print(f"[design] wrote {out_tsv} rows: {len(design)}")
    print(design.head().to_string(index=False))


if __name__ == "__main__":
    main()

