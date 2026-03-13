#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备 StringTie merge 所需的 GTF 列表文件

功能：
- 接收多个 GTF 文件路径（从 Snakemake expand 传入）
- 生成一个文本文件，每行一个 GTF 路径
- 供 StringTie --merge 使用
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="准备 StringTie merge 的 GTF 列表文件"
    )
    parser.add_argument(
        "--gtf-list",
        nargs="+",
        required=True,
        help="GTF 文件路径列表（从 Snakemake expand 传入）"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出的列表文件路径"
    )
    
    args = parser.parse_args()
    
    # 检查所有 GTF 文件是否存在
    valid_gtfs = []
    for gtf_path in args.gtf_list:
        gtf = Path(gtf_path)
        if gtf.exists():
            valid_gtfs.append(str(gtf.absolute()))
        else:
            print(f"[WARN] GTF 文件不存在，跳过: {gtf_path}", file=__import__('sys').stderr)
    
    if not valid_gtfs:
        raise RuntimeError("没有有效的 GTF 文件")
    
    # 写入列表文件
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for gtf in valid_gtfs:
            f.write(f"{gtf}\n")
    
    print(f"[prepare_stringtie_merge] 已生成列表文件: {output_path}")
    print(f"[prepare_stringtie_merge] 包含 {len(valid_gtfs)} 个 GTF 文件")


if __name__ == "__main__":
    main()

