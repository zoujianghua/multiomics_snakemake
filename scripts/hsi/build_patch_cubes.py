#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【HSI patch pipeline - Patch 集合文件生成】

将 base 索引中的所有 patch 按 cube_npz 分组，生成「一 cube 一个 patch 集合文件」的存储结构。

功能：
1. 读取 patch_index_base.tsv（包含 patch_id, sample_id, cube_npz, y0, x0, size）
2. 按 cube_npz 分组，对每个 cube：
   - 一次性加载 cube_npz 的 R 和 mask
   - 裁剪该 cube 下的所有 patch
   - 保存为一个 patch_cubes 文件：{sample_id}_patches.npz
3. 更新 base 索引，添加 cube_patch_npz 和 patch_idx 列

输出：
- results/hsi/patch_cubes/{sample_id}_patches.npz（每个 cube 一个文件）
- results/hsi/ml/patch_index_base_cubes.tsv（更新后的索引）

性能优化：
- 使用多进程并行处理（按 cube 分组）
- 单个进程内一次性加载 cube，避免重复 I/O
- 使用 memory-mapped 模式可进一步优化（可选）
"""

import argparse
import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd


def process_single_cube(cube_group, outdir, dtype):
    """
    处理单个 cube 的所有 patch
    
    参数：
        cube_group: (sample_id, cube_npz_path, patch_rows) 元组
        outdir: 输出目录
        dtype: 数据类型（float32 / uint16 / uint8）
    
    返回：
        (sample_id, cube_patch_npz_path, patch_data_list, success)
    """
    sample_id, cube_npz_path, patch_rows = cube_group
    
    try:
        # 加载 cube
        cube_data = np.load(cube_npz_path, allow_pickle=True)
        R = cube_data["R"]  # (H, W, B)
        
        # 获取波长信息（如果存在）
        wavelength = cube_data.get("wavelength", None)
        
        # 收集该 cube 的所有 patch
        patches_list = []
        coords_list = []
        
        for _, row in patch_rows.iterrows():
            y0 = int(row["y0"])
            x0 = int(row["x0"])
            size = int(row["size"])
            
            # 裁剪 patch: R[y0:y0+size, x0:x0+size, :] -> (size, size, B)
            # 转换为 (B, size, size) 格式
            patch = R[y0:y0+size, x0:x0+size, :]
            patch = np.moveaxis(patch, -1, 0)  # (B, H, W)
            
            # 转换数据类型
            if dtype == "float32":
                patch = patch.astype(np.float32)
            elif dtype == "uint16":
                # 假设原始数据在 [0, 1] 范围，转换为 uint16 [0, 65535]
                patch = np.clip(patch * 65535, 0, 65535).astype(np.uint16)
            elif dtype == "uint8":
                # 假设原始数据在 [0, 1] 范围，转换为 uint8 [0, 255]
                patch = np.clip(patch * 255, 0, 255).astype(np.uint8)
            else:
                raise ValueError(f"不支持的 dtype: {dtype}")
            
            patches_list.append(patch)
            coords_list.append([y0, x0, size])
        
        # 堆叠所有 patch: [N, B, H, W]
        patches_array = np.stack(patches_list, axis=0)
        coords_array = np.array(coords_list, dtype=np.int32)
        
        # 保存 patch_cubes 文件
        # 文件命名：{sample_id}_patches.npz
        # 注意：由于按 (sample_id, cube_npz) 分组，每个组合生成一个独立文件
        # 如果同一个 sample_id 有多个 cube_npz，每个 cube_npz 会生成一个文件
        # 为了确保唯一性，如果检测到 cube_npz 的 basename 与 sample_id 不同，
        # 则使用 cube_npz 的 basename（通常包含 sample_id 信息）
        cube_basename = Path(cube_npz_path).stem
        
        # 优先使用 sample_id（符合用户要求）
        # 但如果 cube_basename 与 sample_id 不同，使用 cube_basename 避免冲突
        if cube_basename == sample_id:
            cube_patch_npz_path = outdir / f"{sample_id}_patches.npz"
        elif sample_id in cube_basename:
            # cube_basename 包含 sample_id，但可能还有其他后缀，使用 cube_basename 更安全
            cube_patch_npz_path = outdir / f"{cube_basename}_patches.npz"
        else:
            # cube_basename 与 sample_id 完全不同，使用 cube_basename 确保唯一性
            cube_patch_npz_path = outdir / f"{cube_basename}_patches.npz"
        
        np.savez_compressed(
            cube_patch_npz_path,
            patches=patches_array,
            coords=coords_array,
            wavelength=wavelength if wavelength is not None else np.array([]),
        )
        
        # 构建 patch_data_list（用于更新索引）
        patch_data_list = []
        for idx, (_, row) in enumerate(patch_rows.iterrows()):
            patch_data_list.append({
                "patch_id": row["patch_id"],
                "sample_id": sample_id,
                "cube_npz": str(cube_npz_path),  # 保留原始 cube_npz 路径
                "cube_patch_npz": str(cube_patch_npz_path),
                "patch_idx": idx,
                "y0": row["y0"],
                "x0": row["x0"],
                "size": row["size"],
            })
        
        return (sample_id, str(cube_patch_npz_path), patch_data_list, True)
        
    except Exception as e:
        print(f"[ERROR] 处理 {sample_id} ({cube_npz_path}) 失败: {e}", file=sys.stderr)
        return (sample_id, None, [], False)


def main():
    ap = argparse.ArgumentParser(
        description="将 base 索引中的所有 patch 按 cube 分组，生成 patch 集合文件"
    )
    ap.add_argument(
        "--base-index",
        required=True,
        help="base 索引文件：results/hsi/ml/patch_index_base.tsv"
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="输出目录：results/hsi/patch_cubes"
    )
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "uint16", "uint8"],
        help="数据类型（默认 float32）"
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行进程数（默认 min(cpu_count, cube_count)）"
    )
    ap.add_argument(
        "--out-index",
        required=True,
        help="更新后的索引文件：results/hsi/ml/patch_index_base_cubes.tsv"
    )
    args = ap.parse_args()
    
    # 读取 base 索引
    print(f"[build_patch_cubes] 读取 base 索引: {args.base_index}")
    df_base = pd.read_csv(args.base_index, sep="\t")
    df_base.columns = [c.strip().lower() for c in df_base.columns]
    
    # 检查必需列
    required_cols = ["patch_id", "sample_id", "cube_npz", "y0", "x0", "size"]
    missing_cols = [col for col in required_cols if col not in df_base.columns]
    if missing_cols:
        raise RuntimeError(f"base_index 缺少必需的列: {missing_cols}")
    
    # 按 (sample_id, cube_npz) 分组
    # 每个组合会生成一个独立的 patch_cubes 文件
    print(f"[build_patch_cubes] 按 (sample_id, cube_npz) 分组...")
    cube_groups = []
    for (sample_id, cube_npz), group in df_base.groupby(["sample_id", "cube_npz"]):
        cube_npz_path = Path(cube_npz)
        if not cube_npz_path.is_file():
            print(f"[WARN] cube_npz 不存在: {cube_npz_path}，跳过")
            continue
        
        # 检查该 cube 是否有 patch
        if len(group) == 0:
            print(f"[WARN] {sample_id} ({cube_npz_path}) 没有 patch，跳过")
            continue
        
        cube_groups.append((sample_id, cube_npz_path, group))
    
    print(f"[build_patch_cubes] 共 {len(cube_groups)} 个 cube 需要处理")
    
    # 创建输出目录
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 确定 workers 数量
    if args.workers is None:
        workers = min(cpu_count(), len(cube_groups))
    else:
        workers = min(args.workers, len(cube_groups))
    
    print(f"[build_patch_cubes] 使用 {workers} 个进程并行处理")
    print(f"[build_patch_cubes] dtype={args.dtype}")
    
    # 多进程处理
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(
                partial(process_single_cube, outdir=outdir, dtype=args.dtype),
                cube_groups
            )
    else:
        results = [process_single_cube(g, outdir, args.dtype) for g in cube_groups]
    
    # 收集结果
    all_patch_data = []
    success_count = 0
    for sample_id, cube_patch_npz, patch_data_list, success in results:
        if success:
            all_patch_data.extend(patch_data_list)
            success_count += 1
        else:
            print(f"[WARN] {sample_id} 处理失败，跳过")
    
    print(f"[build_patch_cubes] 成功处理 {success_count}/{len(cube_groups)} 个 cube")
    print(f"[build_patch_cubes] 总 patch 数: {len(all_patch_data)}")
    
    # 构建更新后的索引 DataFrame
    df_updated = pd.DataFrame(all_patch_data)
    
    # 统计信息
    if len(df_updated) > 0:
        patches_per_cube = df_updated.groupby("cube_patch_npz").size()
        print(f"[build_patch_cubes] 每个 cube 的 patch 数统计:")
        print(f"  min: {patches_per_cube.min()}")
        print(f"  max: {patches_per_cube.max()}")
        print(f"  mean: {patches_per_cube.mean():.1f}")
        print(f"  median: {patches_per_cube.median():.1f}")
        
        # 文件大小统计（采样几个文件）
        sample_files = df_updated["cube_patch_npz"].unique()[:5]
        file_sizes_mb = []
        for sample_file in sample_files:
            if Path(sample_file).exists():
                file_size_mb = Path(sample_file).stat().st_size / (1024 * 1024)
                file_sizes_mb.append(file_size_mb)
        
        if file_sizes_mb:
            print(f"[build_patch_cubes] patch_cubes 文件大小统计 (采样 {len(file_sizes_mb)} 个文件):")
            print(f"  min: {min(file_sizes_mb):.2f} MB")
            print(f"  max: {max(file_sizes_mb):.2f} MB")
            print(f"  mean: {sum(file_sizes_mb) / len(file_sizes_mb):.2f} MB")
        
        print(f"[build_patch_cubes] 总 patch 数: {len(df_updated)}")
        print(f"[build_patch_cubes] 总 cube 数: {len(patches_per_cube)}")
        print(f"[build_patch_cubes] 平均每个 cube 的 patch 数: {len(df_updated) / len(patches_per_cube):.1f}")
        print(f"[build_patch_cubes] 保存 dtype: {args.dtype}")
    
    # 保存更新后的索引
    out_index_path = Path(args.out_index)
    out_index_path.parent.mkdir(parents=True, exist_ok=True)
    df_updated.to_csv(out_index_path, sep="\t", index=False)
    
    print(f"[build_patch_cubes] 完成！")
    print(f"[build_patch_cubes] 输出索引: {out_index_path}")
    print(f"[build_patch_cubes] 输出目录: {outdir}")


if __name__ == "__main__":
    main()

