# -*- coding: utf-8 -*-
import csv, glob, os


def expand_list(spec: str):
    """Split by ';'. If a token ends with '/', expand '*.fq.gz' under that dir;
    otherwise glob the token; if no hits, keep the token as-is.
    """
    spec = (spec or "").strip()
    if not spec:
        return []
    out = []
    for token in spec.split(";"):
        token = token.strip()
        if not token:
            continue
        if token.endswith("/"):
            out.extend(sorted(glob.glob(os.path.join(token, "*.fq.gz"))))
        else:
            hits = glob.glob(token)
            out.extend(sorted(hits if hits else [token]))
    return out


WGS = {}
WGS_IDS = []
with open("config/samples_wgs.csv") as f:
    for r in csv.DictReader(f):
        r1 = expand_list(r.get("fastq1", ""))
        r2 = expand_list(r.get("fastq2", ""))
        assert (
            len(r1) == len(r2) and len(r1) > 0
        ), f"{r.get('sample_id')}: R1/R2 list mismatch or empty"
        sid = r["sample_id"]
        WGS[sid] = {
            "r1": r1,
            "r2": r2,
            "big_region": r.get("big_region", ""),
            "sub_region": r.get("sub_region", ""),
        }
        WGS_IDS.append(sid)
WGS_IDS = sorted(set(WGS_IDS))
