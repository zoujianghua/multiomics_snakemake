#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【HSI patch pipeline - Patch 特征清洗】

对 raw_patch_features_{target}.tsv 进行异常值清洗的轻量 wrapper。

功能：
- 复用 clean_image_features.py 的逻辑
- 支持按 patch_target（phase_core / metabo_state / rnaseq_state）分组进行 patch 级 outlier 过滤

输入：
- --images-raw : raw_patch_features_{target}.tsv
- --out        : clean_patch_features_{target}.tsv
- --group-col  : 分组列（默认使用 patch_target，可通过参数指定）

说明：
- 可以用 phase_core / metabo_state / rnaseq_state 作为分组进行 patch 级 outlier 过滤
- 对于 patch 级数据，建议使用较小的 max-z 阈值（例如 3.0-4.0），因为 patch 数量巨大
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="对 raw_patch_features 进行异常值清洗（wrapper）"
    )
    ap.add_argument(
        "--images-raw",
        required=True,
        help="raw_patch_features_{target}.tsv"
    )
    ap.add_argument(
        "--out",
        required=True,
        help="clean_patch_features_{target}.tsv"
    )
    ap.add_argument(
        "--group-col",
        default=None,
        help="分组列（默认自动选择：phase > phase_core > 不分组）"
    )
    ap.add_argument(
        "--min-r800",
        type=float,
        default=0.04,
        help="R800_med 阈值（默认 0.04）"
    )
    ap.add_argument(
        "--max-z",
        type=float,
        default=3.0,
        help="robust z-score 阈值（默认 3.0，patch 级建议较小值）"
    )
    ap.add_argument(
        "--outliers-report",
        default=None,
        help="可选：离群样本详情输出路径"
    )
    args = ap.parse_args()
    
    # 构建 clean_image_features.py 的命令
    script_path = Path(__file__).parent / "clean_image_features.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--images-raw", args.images_raw,
        "--out", args.out,
        "--min-r800", str(args.min_r800),
        "--max-z", str(args.max_z),
    ]
    
    if args.group_col:
        cmd.extend(["--group-col", args.group_col])
    
    if args.outliers_report:
        cmd.extend(["--outliers-report", args.outliers_report])
    
    # 执行
    print(f"[clean_patch_features] 调用 clean_image_features.py...")
    print(f"[clean_patch_features] 命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, check=True)
    
    print(f"[clean_patch_features] 完成！")
    print(f"[clean_patch_features] 输出: {args.out}")


if __name__ == "__main__":
    main()

