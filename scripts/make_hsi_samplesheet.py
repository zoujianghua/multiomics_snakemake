#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_hsi_samplesheet_v5.py

根据目录结构 <group_dir>/<sample_dir>/capture/*.hdr 生成 HSI samplesheet：

- 精细 phase 直接在这里确定，比如：
    control_25_T2h
    control_25_T8d
    stress_35_T3d
    recovery_from_35_T6h
    recovery_from_10_T6h

- 输出列：
    sample_id, group_dir, sample_dir, capture_dir,
    sample_hdr, white_hdr, dark_hdr,
    temp, time, time_h, phase, replicate
"""

import sys, csv, re
from pathlib import Path
from collections import Counter

# === 时间映射 ===
TIME_MAP = {
    '2h': 2.0,
    '6h': 6.0,
    '1d': 24.0,
    '3d': 72.0,
    '7d': 168.0,
}

def to_hours(t: str) -> float:
    t = str(t).strip().lower()
    if t in TIME_MAP:
        return TIME_MAP[t]
    m = re.match(r'(\d+)\s*h', t)
    if m:
        return float(m.group(1))
    m = re.match(r'(\d+)\s*d', t)
    if m:
        return float(m.group(1)) * 24.0
    return float('nan')

def parse_time(src: str) -> str:
    """
    从 group_dir 或 sample_id 中解析时间字符串：
    - 识别 T2h/T6h/T1d/T3d/T7d
    - 或 rec_2h/rec-3d 这类形式
    """
    m = re.search(r'(?i)(?:^|[_-])T(\d+)([hd])', src, re.I)
    if not m:
        m = re.search(r'(?i)(?:^|[_-])rec[_-]?(\d+)([hd])', src, re.I)
    return f"{m.group(1)}{m.group(2).lower()}" if m else ""

def parse_phase_full(group_name: str, sample_id: str, time: str):
    """
    结合 group_dir 和 sample_id 推出精细 phase / temp / time_h / replicate。

    目标：
      CK25_T2h_R1 -> control_25_T2h,  temp=25, time=2h
      HT35_T3d_R2 -> stress_35_T3d,   temp=35, time=3d
      HTR_T6h_R1  -> recovery_from_35_T6h, temp=25
      LTR_T6h_R2  -> recovery_from_10_T6h, temp=25
      LT10_T7d_R1 -> stress_10_T7d,   temp=10
    """
    sid = str(sample_id)
    sid_up = sid.upper()
    grp_up = group_name.upper()

    # 优先用 sample_id 的前缀模式
    m = re.match(r'^(CK|HTR|LTR|HT|LT)(\d+)?_T([0-9]+[hd])(?:_R(\d+))?', sid_up)
    if m:
        prefix = m.group(1)
        deg = m.group(2) or ""
        time_token = m.group(3)     # 例如 2h/3d
        rep = m.group(4) or ""
        time_val = time_token
    else:
        # fallback：从 group_dir 补充前缀/温度信息
        m2 = re.match(r'^(CK|HTR|LTR|HT|LT)(\d+)?', grp_up)
        if m2:
            prefix = m2.group(1)
            deg = m2.group(2) or ""
        else:
            prefix = "CK"
            deg = ""
        rep = ""
        time_val = time or ""

    time_h = to_hours(time_val)

    # 按前缀构造 phase / temp
    if prefix == "CK":
        temp = deg if deg else "25"
        phase_core = f"control_{temp}"
        phase_full = f"{phase_core}_T{time_val}"
    elif prefix == "HT":
        temp = deg
        phase_core = "stress_35" if deg == "35" else f"stress_{deg}"
        phase_full = f"{phase_core}_T{time_val}"
    elif prefix == "LT":
        temp = deg
        phase_core = "stress_10" if deg == "10" else f"stress_{deg}"
        phase_full = f"{phase_core}_T{time_val}"
    elif prefix == "HTR":
        temp = "25"
        phase_core = "recovery_from_35"
        phase_full = f"{phase_core}_T{time_val}"
    elif prefix == "LTR":
        temp = "25"
        phase_core = "recovery_from_10"
        phase_full = f"{phase_core}_T{time_val}"
    else:
        temp = "25"
        phase_core = "control_25"
        phase_full = f"{phase_core}_T{time_val}"

    return {
        "phase": phase_full,
        "phase_core":phase_core,
        "temp": temp,
        "time": time_val,
        "time_h": time_h,
        "replicate": rep,
    }

def main():
    if len(sys.argv) < 3:
        print("用法: make_hsi_samplesheet_v5.py <hsi_root> <out_csv>", file=sys.stderr)
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    out_csv = sys.argv[2]

    rows = []
    for group_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for sample_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()]):
            cap = sample_dir / "capture"
            if not cap.exists():
                continue
            hdrs = list(cap.glob("*.hdr"))
            if not hdrs:
                continue

            whites = [h for h in hdrs if "WHITEREF" in h.name.upper()]
            darks  = [h for h in hdrs if "DARKREF"  in h.name.upper()]
            samples= [h for h in hdrs
                      if ("WHITEREF" not in h.name.upper()
                          and "DARKREF" not in h.name.upper())]

            for sh in sorted(samples):
                sid   = sh.stem
                # time 可以从 group_dir + sample_id 综合判断
                time  = parse_time(f"{group_dir.name}_{sid}")

                meta = parse_phase_full(group_dir.name, sid, time)
                phase = meta["phase"]
                phase_core = meta["phase_core"]
                temp  = meta["temp"]
                time  = meta["time"]
                time_h= meta["time_h"]
                rep   = meta["replicate"]

                rows.append({
                    "sample_id": sid,
                    "group_dir": group_dir.name,
                    "sample_dir": str(sample_dir.resolve()),
                    "capture_dir": str(cap.resolve()),
                    "sample_hdr": str(sh.resolve()),
                    "white_hdr": str(whites[0].resolve()) if whites else "",
                    "dark_hdr":  str(darks[0].resolve())  if darks  else "",
                    "temp": temp,
                    "time": time,
                    "time_h": time_h,
                    "phase": phase,
                    "phase_core":phase_core,
                    "replicate": rep,
                })

    if not rows:
        print("[!] 未找到任何 .hdr 样本；请检查根目录结构。", file=sys.stderr)
        sys.exit(2)

    fields = [
        "sample_id","group_dir","sample_dir","capture_dir",
        "sample_hdr","white_hdr","dark_hdr",
        "temp","time","time_h","phase","phase_core","replicate"
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    print(f"[OK] 写出 {len(rows)} 行 -> {out_csv}")

    print("[摘要] phase:", dict(Counter(r['phase'] or 'NA' for r in rows)))
    print("[摘要] time :", dict(Counter(str(r['time'] or 'NA') for r in rows)))
    print("[摘要] temp :", dict(Counter(str(r['temp'] or 'NA') for r in rows)))

if __name__ == "__main__":
    main()

