#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【HSI patch pipeline - DEPRECATED】

从 patch_index.tsv 预裁剪所有 patch 并保存为单独的 npz 文件

**此脚本已被 build_patch_cubes.py 取代，不再推荐使用。**

新流程使用「一 cube 一个 patch 集合文件」的存储结构，大幅减少 I/O 开销。
每个 patch 不再单独保存为一个 npz 文件，而是按 cube 分组保存在一个集合文件中。

保留此脚本仅用于兼容旧流程，新流程请使用：
- build_patch_index_base.py - 生成通用几何索引
- build_patch_cubes.py - 生成 patch 集合文件（一 cube 一个文件）
- make_patch_index_for_target.py - 生成任务特定索引

功能：
1. 读取 patch_index.tsv（包含 patch_id, sample_id, cube_npz, y0, x0, size 等）
2. 按 cube_npz 分组，避免重复加载大文件
3. 对每个 patch，从 cube_npz 中裁剪并保存为独立的 patch npz 文件
4. 更新索引表，添加 patch_npz 列

使用多进程加速处理（HPC 环境优化）
"""

import argparse
import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd


def process_single_patch(row_dict, cube_data, outdir, dtype):
    """
    处理单个 patch
    
    参数：
        row_dict: patch 行数据（字典）
        cube_data: 已加载的 cube 数据（dict-like）
        outdir: 输出目录
        dtype: 数据类型
    
    返回：
        (patch_id, patch_npz_path, success)
    """
    patch_id = row_dict["patch_id"]
    y0 = int(row_dict["y0"])
    x0 = int(row_dict["x0"])
    size = int(row_dict["size"])
    
    try:
        # 从 cube 数据中提取 R 和 wavelength
        R = cube_data.get("R")
        if R is None:
            # 如果没有 R 键，尝试第一个键
            first_key = list(cube_data.keys())[0]
            R = cube_data[first_key]
        
        # R 形状应该是 [H, W, B]
        if R.ndim != 3:
            return (patch_id, None, False)
        
        # 裁剪 patch
        patch_R = R[y0:y0+size, x0:x0+size, :]  # [H, W, B]
        
        # 转换为 [B, H, W]
        patch_cube = np.moveaxis(patch_R, -1, 0).astype(dtype)
        
        # 生成输出路径
        out_npz = Path(outdir) / f"{patch_id}.npz"
        
        # 准备保存的数据
        save_dict = {"R": patch_cube}
        
        # 如果有 wavelength，也保存
        if "wavelength" in cube_data:
            save_dict["wavelength"] = cube_data["wavelength"]
        
        # 保存为压缩的 npz
        np.savez_compressed(out_npz, **save_dict)
        
        # 返回相对路径（POSIX 风格）
        patch_npz_path = str(out_npz)
        
        return (patch_id, patch_npz_path, True)
        
    except Exception as e:
        print(f"[ERROR] 处理 patch {patch_id} 失败: {e}", file=sys.stderr)
        return (patch_id, None, False)


def process_cube_group(cube_path, sub_df, outdir, dtype):
    """
    处理一个 cube 组的所有 patch（单进程，因为 cube 数据较大，多进程序列化开销大）
    
    参数：
        cube_path: cube_npz 文件路径
        sub_df: 该 cube 对应的所有 patch 行
        outdir: 输出目录
        dtype: 数据类型
    
    返回：
        list of (patch_id, patch_npz_path, success)
    """
    try:
        # 加载 cube 数据（只加载一次）
        cube_data = np.load(cube_path, allow_pickle=True)
        
        # 处理该 cube 下的所有 patch
        results = []
        for _, row in sub_df.iterrows():
            result = process_single_patch(row.to_dict(), cube_data, outdir, dtype)
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"[ERROR] 加载 cube {cube_path} 失败: {e}", file=sys.stderr)
        # 返回失败结果
        return [
            (row["patch_id"], None, False)
            for _, row in sub_df.iterrows()
        ]
    finally:
        # 清理内存
        if 'cube_data' in locals():
            del cube_data


def main():
    parser = argparse.ArgumentParser(
        description="从 patch_index.tsv 预裁剪所有 patch 并保存为 npz 文件"
    )
    parser.add_argument(
        "--index",
        required=True,
        help="输入的 patch 索引 tsv 文件"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="输出 patch npz 的目录"
    )
    parser.add_argument(
        "--overwrite-index",
        action="store_true",
        help="是否覆盖原索引文件（默认 False，生成新文件）"
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64", "uint16", "uint8"],
        help="保存 patch 时的数据类型（默认 float32）"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="多进程数（默认使用 CPU 核心数）"
    )
    
    args = parser.parse_args()
    
    # 解析 dtype
    dtype_map = {
        "float32": np.float32,
        "float64": np.float64,
        "uint16": np.uint16,
        "uint8": np.uint8,
    }
    dtype = dtype_map[args.dtype]
    
    # 确定进程数
    n_workers = args.n_workers if args.n_workers is not None else cpu_count()
    print(f"[patch_npz] 使用 {n_workers} 个进程")
    
    # 读取索引文件
    index_path = Path(args.index)
    print(f"[patch_npz] 读取索引文件: {index_path}")
    df = pd.read_csv(index_path, sep="\t")
    
    # 列名统一小写
    df.columns = [c.strip().lower() for c in df.columns]
    
    # 检查必需的列
    required_cols = ["patch_id", "sample_id", "cube_npz", "y0", "x0", "size", "target", "split"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise RuntimeError(f"索引文件缺少必需的列: {missing_cols}")
    
    # 创建输出目录
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[patch_npz] 输出目录: {outdir}")
    
    # 按 cube_npz 分组处理
    print(f"[patch_npz] 开始处理 {len(df)} 个 patch...")
    cube_groups = df.groupby("cube_npz")
    n_cubes = len(cube_groups)
    print(f"[patch_npz] 共 {n_cubes} 个 cube 文件")
    
    # 处理每个 cube 组（使用多进程处理不同的 cube，但每个 cube 内部单进程）
    all_results = []
    
    # 准备参数列表（每个 cube 一组）
    cube_args = [
        (cube_path, sub_df, outdir, dtype)
        for cube_path, sub_df in cube_groups
    ]
    
    # 使用多进程处理不同的 cube（每个进程处理一个 cube 的所有 patch）
    if n_workers > 1 and len(cube_args) > 1:
        print(f"[patch_npz] 使用 {min(n_workers, len(cube_args))} 个进程并行处理 {len(cube_args)} 个 cube")
        with Pool(min(n_workers, len(cube_args))) as pool:
            results_list = pool.starmap(process_cube_group, cube_args)
        # 展平结果
        for results in results_list:
            all_results.extend(results)
    else:
        # 单进程处理
        for i, (cube_path, sub_df) in enumerate(cube_groups, 1):
            print(f"[patch_npz] 处理 cube {i}/{n_cubes}: {Path(cube_path).name} ({len(sub_df)} patches)")
            
            if not Path(cube_path).is_file():
                print(f"[WARN] cube 文件不存在: {cube_path}")
                # 为所有 patch 标记为失败
                for _, row in sub_df.iterrows():
                    all_results.append((row["patch_id"], None, False))
                continue
            
            results = process_cube_group(cube_path, sub_df, outdir, dtype)
            all_results.extend(results)
    
    # 创建结果字典
    result_dict = {patch_id: path for patch_id, path, success in all_results if success}
    
    # 更新 DataFrame
    df["patch_npz"] = df["patch_id"].map(result_dict)
    df["patch_npz"] = df["patch_npz"].fillna("")
    
    # 统计成功数量
    n_success = sum(1 for _, _, success in all_results if success)
    n_failed = len(all_results) - n_success
    print(f"[patch_npz] 成功: {n_success}, 失败: {n_failed}")
    
    # 确定输出索引文件路径
    if args.overwrite_index:
        output_index = index_path
        temp_index = index_path.with_suffix(".tmp")
        # 先写到临时文件
        df.to_csv(temp_index, sep="\t", index=False)
        # 安全地重命名
        temp_index.replace(output_index)
        print(f"[patch_npz] 已覆盖索引文件: {output_index}")
    else:
        # 生成新文件名：patch_index_xxx.tsv -> patch_index_xxx_npz.tsv
        stem = index_path.stem
        output_index = index_path.parent / f"{stem}_npz.tsv"
        df.to_csv(output_index, sep="\t", index=False)
        print(f"[patch_npz] 已生成新索引文件: {output_index}")
    
    print(f"[patch_npz] 完成！输出目录: {outdir}")
    print(f"[patch_npz] 索引文件: {output_index}")


if __name__ == "__main__":
    main()

