#!/usr/bin/env python3
import sys, re, csv, argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("raw_dirs", nargs="+", help="FASTQ 根目录（可一个或多个）")
p.add_argument("--default-batch", default="b1")
p.add_argument("--default-rin", default="")
args = p.parse_args()

# 适配：CK25_T1d_R1_1.fq.gz / HT35_T1d_R1_2.fq.gz / LTR_T1d_R5_2.fq.gz
pat = re.compile(r'(?P<group>[A-Za-z]+[0-9]*)_(?P<time>T\d+(?:h|d))_R(?P<rep>\d+)_(?P<mate>[12])\.(?:f(?:ast)?q\.gz)$', re.I)

def infer_temp_phase(group:str):
    g = group.upper()
    digits = "".join(ch for ch in g if ch.isdigit())
    if g.startswith("CK"): return (digits or "25", "stress")
    if g.startswith("LT") and g != "LTR": return (digits, "stress")
    if g.startswith("HT") and g != "HTR": return (digits, "stress")
    if g in ("HTR","LTR") or g.endswith("TR"): return ("25", "recovery")
    return ("", "stress")

rows = {}
for base in args.raw_dirs:
    for pth in Path(base).rglob("*.f*q.gz"):
        m = pat.search(pth.name)
        if not m: 
            continue
        group, time, rep, mate = m["group"].upper(), m["time"], m["rep"], m["mate"]
        sid = f"{group}_{time}_R{rep}"
        r = rows.setdefault(sid, {
            "sample_id": sid, "assay":"RNAseq", "region":"A", "group":group,
            "temperature":"", "phase":"", "time":time.replace("T",""),
            "replicate":rep, "batch":args.default_batch, "rin":args.default_rin,
            "fastq1":"", "fastq2":""
        })
        if mate == "1": r["fastq1"] = str(pth.resolve())
        else: r["fastq2"] = str(pth.resolve())

for r in rows.values():
    t, ph = infer_temp_phase(r["group"])
    r["temperature"], r["phase"] = t, ph

fieldnames = ["sample_id","assay","region","group","temperature","phase","time","replicate","batch","rin","fastq1","fastq2"]
w = csv.DictWriter(sys.stdout, fieldnames=fieldnames); w.writeheader()

missing = []
for sid in sorted(rows):
    d = rows[sid]
    if not (d["fastq1"] and d["fastq2"]):
        missing.append(sid)
    w.writerow(d)

if missing:
    print(f"[WARN] 有 {len(missing)} 个样本缺少配对端（请检查是否少了 _1 或 _2）：", file=sys.stderr)
    for sid in missing:
        print("  -", sid, file=sys.stderr)

