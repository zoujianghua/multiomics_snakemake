#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 number_sample_name.xlsx 生成：
- config/samples_metabolomics.csv
  * phase = 精细分组（control_25_T2h / stress_35_T3d / recovery_from_35_T6h 等）
  * time  = 小写且不带 T（如 2h/3d）
  * time_h = 自然小时
  * t_eff_h = 有效小时（恢复类样本在 time_h 基础上 +168）
- config/metabo_design_pos.tsv / config/metabo_design_neg.tsv
  * 设计表用列名 group，但取值==对应样本的 phase
  * 同时给出 time, time_h, t_eff_h, temp, batch, is_qc 等列
支持样本名：CK25_T2h_R1 / HT35_T1d_R2 / LT10_T7d_R3 / HTR_T6h_R1 / LTR_T3d_R2
"""

import argparse, re, sys
from pathlib import Path
import pandas as pd

def extract_core_digits(s: str) -> str:
    m = re.search(r'(\d{6,12})', str(s))
    return m.group(1) if m else ""

def to_hours(time_str: str) -> float:
    """'2h' -> 2 ; '3d' -> 72；其他返回 NaN"""
    s = str(time_str).strip().lower()
    m = re.match(r'^(\d+)([hd])$', s)
    if not m:
        return float('nan')
    v = float(m.group(1))
    return v if m.group(2) == 'h' else v * 24.0

def parse_phase_full(sample_id: str):
    """
    直接从 sample_id 推出精细 phase / temp / time / time_h / replicate

      CK25_T2h_R1 -> control_25_T2h,  temp=25, time=2h
      HT35_T3d_R2 -> stress_35_T3d,   temp=35, time=3d
      HTR_T6h_R1  -> recovery_from_35_T6h, temp=25
      LTR_T6h_R2  -> recovery_from_10_T6h, temp=25
      LT10_T7d_R1 -> stress_10_T7d,   temp=10
    """
    sid_up = str(sample_id).strip().upper()
    m = re.match(r'^(?P<prefix>CK|HTR|LTR|HT|LT)(?P<deg>\d+)?_T(?P<tval>\d+)(?P<tunit>[HD])_R(?P<rep>\d+)$', sid_up)
    if not m:
        # 兜底：尽量给出可用字段
        return {"phase":"control_25_T0h","temp":"25","time":"0h","time_h":0.0,"replicate":""}

    prefix = m.group("prefix")
    deg    = (m.group("deg") or "").strip()
    tval   = m.group("tval")
    tunit  = m.group("tunit").lower()
    time   = f"{tval}{tunit}"      # 例如 2h / 3d（小写，不带 T）
    time_h = to_hours(time)
    rep    = "R" + m.group("rep")

    # 一步到位的 phase_core + 附上 _T{time}
    if prefix == "CK":
        temp = deg if deg else "25"
        phase_core = f"control_{temp}"
    elif prefix == "HTR":
        temp = "25"
        phase_core = "recovery_from_35"
    elif prefix == "LTR":
        temp = "25"
        phase_core = "recovery_from_10"
    elif prefix == "HT":
        temp = deg
        phase_core = "stress_35" if deg == "35" else f"stress_{deg}"
    elif prefix == "LT":
        temp = deg
        phase_core = "stress_10" if deg == "10" else f"stress_{deg}"
    else:
        temp = "25"
        phase_core = "control_25"

    phase_full = f"{phase_core}_T{time}"
    return {"phase": phase_full, "temp": temp, "time": time, "time_h": time_h, "replicate": rep}

def find_best_mzxml(raw_dir: Path, core: str):
    """匹配编号 core，并优先 2A 其次 1A。"""
    if not raw_dir:
        return "", ""
    hits = []
    for p in raw_dir.rglob(f"*{core}-[12]A.mzXML"):
        mm = re.search(rf"{core}-([12]A)\.mzXML$", p.name, flags=re.I)
        inj = mm.group(1).upper() if mm else ""
        hits.append((inj, str(p.resolve())))
    if not hits:
        return "", ""
    hits.sort(key=lambda x: (x[0] != "2A", x[0]))  # 2A 优先
    inj, path = hits[0]
    return path, inj

def pick_col(df, want):
    """在 xlsx 中挑列名（避免花哨语法）"""
    want = want.lower()
    for c in df.columns:
        s = str(c).lower()
        if (want == "number" and s.startswith("number")) or (want == "sample" and "sample" in s):
            return c
    raise ValueError(f"在 xlsx 中找不到列: {want}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-xlsx", required=True, help="number_sample_name.xlsx")
    ap.add_argument("--filelist-csv", default="", help="3004946.csv (optional)")
    ap.add_argument("--pos-dir", default="", help="原始 POS 目录（用于定位 mzXML）")
    ap.add_argument("--neg-dir", default="", help="原始 NEG 目录（用于定位 mzXML）")
    ap.add_argument("--out-sheet", default="config/samples_metabolomics.csv")
    ap.add_argument("--out-runlist", default="config/metabo_runlist.csv")
    ap.add_argument("--emit-design-pos", default="config/metabo_design_pos.tsv")
    ap.add_argument("--emit-design-neg", default="config/metabo_design_neg.tsv")
    ap.add_argument("--default-batch", default="b1")
    args = ap.parse_args()

    pos_dir = Path(args.pos_dir).resolve() if args.pos_dir else None
    neg_dir = Path(args.neg_dir).resolve() if args.neg_dir else None

    # 1) 读 number -> sample 映射
    xdf = pd.read_excel(args.map_xlsx)
    col_num = pick_col(xdf, "number")
    col_smp = pick_col(xdf, "sample")
    xdf["core"] = xdf[col_num].astype(str).map(extract_core_digits)
    id2name = dict(zip(xdf["core"], xdf[col_smp].astype(str)))

    # 2) samples 主表
    rows = []
    for core, sample_name in sorted(id2name.items()):
        meta = parse_phase_full(sample_name)
        time_h = meta["time_h"]
        is_rec = str(meta["phase"]).startswith("recovery_from_")
        t_eff_h = (time_h + 168.0) if is_rec else time_h

        pos_file, pos_inj = find_best_mzxml(pos_dir, core) if pos_dir else ("","")
        neg_file, neg_inj = find_best_mzxml(neg_dir, core) if neg_dir else ("","")

        rows.append({
            "sample": sample_name,
            "number_id": core,
            "temp": meta["temp"],
            "phase": meta["phase"],    # 精细分组（含温度与时间）
            "time": meta["time"],      # 小写、不带 T（例：2h/3d）
            "time_h": time_h,          # 自然小时
            "t_eff_h": t_eff_h,        # 恢复类 +168
            "replicate": meta["replicate"],
            "pos_file": pos_file,
            "pos_inj": pos_inj,
            "neg_file": neg_file,
            "neg_inj": neg_inj,
            "sample_type": "biological",
        })

    out_sheet = Path(args.out_sheet); out_sheet.parent.mkdir(parents=True, exist_ok=True)
    cols = ["sample","number_id","temp","phase","time","time_h","t_eff_h","replicate",
            "pos_file","pos_inj","neg_file","neg_inj","sample_type"]
    pd.DataFrame(rows, columns=cols).to_csv(out_sheet, index=False)
    print(f"[OK] wrote {out_sheet} rows={len(rows)}")

    # 3) 可选 runlist
    if args.filelist_csv:
        rr = []
        try:
            df = pd.read_csv(args.filelist_csv)
            col = None
            for c in df.columns:
                if str(c).lower().replace(" ","") in ("filename","file_name"):
                    col = c; break
            if col:
                for v in df[col].astype(str):
                    if v.lower() != "file name":
                        rr.append({"file_name": v})
        except Exception:
            with open(args.filelist_csv, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s and s.lower() != "file name":
                        rr.append({"file_name": s})
        out_run = Path(args.out_runlist); out_run.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rr, columns=["file_name"]).to_csv(out_run, index=False)
        print(f"[OK] wrote {out_run} rows={len(rr)}")
    else:
        print("[INFO] no filelist provided; skip runlist output")

    # 4) 每个模式输出 design：列名 group，但值==phase；恢复类 time_h 已经 +168 放在 t_eff_h
    def dump_design(mode: str, out_tsv: str):
        use_col = "pos_file" if mode == "pos" else "neg_file"
        recs = []
        for r in rows:
            f = r[use_col]
            if not f:
                continue
            recs.append({
                "sample": r["sample"],
                "file": f,
                "group": r["phase"],        # 关键：design 用 group，但值==phase（精细分组）
                "time": r["time"],
                "time_h": r["time_h"],
                "t_eff_h": r["t_eff_h"],    # 恢复类已 +168
                "temp": r["temp"],
                "batch": args.default_batch,
                "is_qc": "FALSE",
                "number_id": r["number_id"],
                "replicate": r["replicate"],
            })
        outp = Path(out_tsv); outp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(recs).to_csv(outp, sep="\t", index=False)
        print(f"[OK] wrote design ({mode}) -> {outp} (n={len(recs)})")

    dump_design("pos", args.emit_design_pos)
    dump_design("neg", args.emit_design_neg)

if __name__ == "__main__":
    main()

