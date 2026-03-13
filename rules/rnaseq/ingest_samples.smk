# rules/rnaseq/ingest_samples.smk
import csv

RNASEQ = {}
RNASEQ_IDS = []

with open("config/samples_rnaseq.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        # keep only rows with both fastq1 and fastq2 present
        if r.get("fastq1") and r.get("fastq2"):
            sid = r["sample_id"]
            RNASEQ[sid] = {
                # keep raw string paths (no split by ';')
                "r1": r["fastq1"].strip(),
                "r2": r["fastq2"].strip(),
                # optional metadata columns
                "temp": str(r.get("temperature", "")).strip(),
                "phase": str(r.get("phase", "")).strip(),
                "time": str(r.get("time", "")).strip(),
                "group": str(r.get("group", "")).strip(),
            }
            RNASEQ_IDS.append(sid)

# deduplicate and sort IDs
RNASEQ_IDS = sorted(set(RNASEQ_IDS))
