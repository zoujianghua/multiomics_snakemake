#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 WGS 样本表（每个小地区目录=一个样本；合并所有 R1 与所有 R2）。
目录结构：<wgs_root>/<大地区>/<小地区>/*.fq.gz 或 *.fastq.gz
输出列：sample_id,big_region,sub_region,fastq1,fastq2
  - fastq1/fastq2 为分号分隔的绝对路径列表（全体 R1 / 全体 R2）
用法：
  python make_wgs_samplesheet.py <wgs_root> [out_csv]
"""
import sys, csv
from pathlib import Path

def is_fastq_gz(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".fastq.gz") or n.endswith(".fq.gz")

def strip_ext(name: str) -> str:
    if name.lower().endswith(".fastq.gz"): return name[:-9]
    if name.lower().endswith(".fq.gz"):    return name[:-6]
    return name

SUF_R1 = ("_R1","_1",".1","-1")
SUF_R2 = ("_R2","_2",".2","-2")
def detect_mate(stem: str):
    if stem.endswith(SUF_R1): return "1"
    if stem.endswith(SUF_R2): return "2"
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: make_wgs_samplesheet.py <wgs_root> [out_csv]", file=sys.stderr)
        sys.exit(1)

    wgs_root = Path(sys.argv[1]).resolve()
    if not wgs_root.exists():
        print(f"[ERR] not found: {wgs_root}", file=sys.stderr); sys.exit(2)

    out_csv = sys.argv[2] if len(sys.argv) > 2 else "config/samples_wgs.csv"
    to_stdout = (out_csv == "-")
    if not to_stdout: Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    rows, warns = [], []

    # 扫两级：大地区/小地区
    for big_dir in sorted([p for p in wgs_root.iterdir() if p.is_dir()]):
        big = big_dir.name
        for sub_dir in sorted([p for p in big_dir.iterdir() if p.is_dir()]):
            sub = sub_dir.name
            files = sorted([p for p in sub_dir.iterdir() if p.is_file() and is_fastq_gz(p)])
            if not files: 
                continue

            R1 = []
            R2 = []
            for fq in files:
                stem = strip_ext(fq.name)
                mate = detect_mate(stem)
                if mate == "1": R1.append(str(fq.resolve()))
                elif mate == "2": R2.append(str(fq.resolve()))

            if len(R1) == 0 or len(R2) == 0:
                warns.append(f"{big}/{sub} (R1={len(R1)}, R2={len(R2)})")
                continue

            rows.append({
                "sample_id": f"{big}_{sub}",
                "big_region": big,
                "sub_region": sub,
                "fastq1": ";".join(sorted(R1)),
                "fastq2": ";".join(sorted(R2)),
            })

    header = ["sample_id","big_region","sub_region","fastq1","fastq2"]
    if to_stdout:
        w = csv.DictWriter(sys.stdout, fieldnames=header); w.writeheader()
        for r in rows: w.writerow(r)
    else:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows: w.writerow(r)

    print(f"[✓] wrote {out_csv if not to_stdout else 'stdout'} ({len(rows)} rows)", file=sys.stderr)
    if warns:
        print("[WARN] skipped (missing R1 or R2):", *warns, sep="\n  ", file=sys.stderr)

if __name__ == "__main__":
    main()

