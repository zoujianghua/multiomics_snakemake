#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【HSI patch pipeline - 通用几何索引生成】

从 image_features.tsv 生成通用的 patch 几何索引（不含 target/split）。

功能：
1. 从 image_features.tsv 读取 sample_id 和 cube_npz
2. 对每张图通过滑窗生成候选 patch
3. 根据 mask 覆盖率筛选 patch
4. 输出 patch_index_base.tsv，包含 patch_id, sample_id, cube_npz, y0, x0, size

注意：此索引不包含 target 和 split 信息，这些信息由后续的 make_patch_index_for_target.py 添加。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="生成通用的 patch 几何索引（不含 target/split）"
    )
    ap.add_argument(
        "--images",
        required=True,
        help="results/hsi/image_features.tsv（包含 sample_id, cube_npz）"
    )
    ap.add_argument(
        "--out",
        required=True,
        help="输出路径：results/hsi/ml/patch_index_base.tsv"
    )
    ap.add_argument(
        "--run-params-out",
        default=None,
        help="可选，运行参数留痕文件路径；未指定时写至 base 同目录 patch_index_run_params.tsv"
    )
    ap.add_argument(
        "--patch-size",
        type=int,
        default=32,
        help="patch 尺寸（默认 32）"
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=16,
        help="滑窗步长（默认 16）"
    )
    ap.add_argument(
        "--min-mask-frac",
        type=float,
        default=1.0,
        help="patch 中 mask>0 像素占比阈值（默认 1.0，即完全在叶片内）。"
             "对于分类问题建议 1.0；对于 pixel-mix 场景可以调低。"
    )
    ap.add_argument(
        "--max-patches-per-image",
        type=int,
        default=200,
        help="每张图最多保留多少 patch（默认 200）；0 或负数表示不设上限，使用全部候选"
    )
    args = ap.parse_args()

    # 读取 image_features.tsv
    print(f"[patch_index_base] 读取 image_features: {args.images}")
    df = pd.read_csv(args.images, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]

    # 检查必需的列
    required_cols = ["sample_id", "cube_npz"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise RuntimeError(f"image_features 缺少必需的列: {missing_cols}")

    df["sample_id"] = df["sample_id"].astype(str)

    rows = []
    ps = args.patch_size
    stride = args.stride
    rng = np.random.default_rng(42)

    print(f"[patch_index_base] 开始处理 {len(df)} 张图像...")
    print(f"[patch_index_base] patch_size={ps}, stride={stride}, min_mask_frac={args.min_mask_frac}")

    for i, row in df.iterrows():
        sid = row["sample_id"]
        cube_npz = row["cube_npz"]

        cube_npz_path = Path(cube_npz)
        if not cube_npz_path.is_file():
            print(f"[WARN] cube_npz 不存在: {cube_npz_path}")
            continue

        try:
            data = np.load(cube_npz_path, allow_pickle=True)
            R = data["R"]  # (H, W, B)
            mask = data["mask"].astype(bool)  # ROI mask

            H, W, _ = R.shape
            if mask.shape[:2] != (H, W):
                print(f"[WARN] mask 尺寸不匹配 {sid}: cube={R.shape}, mask={mask.shape}")
                continue

            # 生成候选 patch
            candidates = []
            for y0 in range(0, H - ps + 1, stride):
                for x0 in range(0, W - ps + 1, stride):
                    sub_mask = mask[y0:y0+ps, x0:x0+ps]
                    frac = sub_mask.mean()
                    if frac >= args.min_mask_frac:
                        candidates.append((y0, x0))

            if not candidates:
                print(f"[INFO] {sid} 没有满足条件的 patch")
                continue

            # 控制每图最多 patch 数量（0 或负数表示不设上限，使用全部候选）
            if args.max_patches_per_image > 0 and len(candidates) > args.max_patches_per_image:
                idx = rng.choice(
                    len(candidates),
                    size=args.max_patches_per_image,
                    replace=False
                )
                candidates = [candidates[j] for j in idx]

            # 添加到结果
            for (y0, x0) in candidates:
                patch_id = f"{sid}_{y0}_{x0}"
                rows.append({
                    "patch_id": patch_id,
                    "sample_id": sid,
                    "cube_npz": str(cube_npz_path),
                    "y0": int(y0),
                    "x0": int(x0),
                    "size": ps,
                })

            if (i + 1) % 10 == 0:
                print(f"[patch_index_base] 已处理 {i+1}/{len(df)} 张图像...")

        except Exception as e:
            print(f"[ERROR] 处理 {sid} 失败: {e}")
            continue

    # 保存结果
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    patch_df = pd.DataFrame(rows)
    patch_df.to_csv(out_path, sep="\t", index=False)

    # 运行参数留痕，便于追溯当前 patch 索引是用哪组几何参数生成的
    from datetime import datetime
    params_path = Path(args.run_params_out) if args.run_params_out else (out_path.parent / "patch_index_run_params.tsv")
    maxp_val = str(args.max_patches_per_image) if args.max_patches_per_image > 0 else "0 (no limit)"
    params_df = pd.DataFrame([
        {"param": "patch_size", "value": str(args.patch_size)},
        {"param": "stride", "value": str(args.stride)},
        {"param": "min_mask_frac", "value": str(args.min_mask_frac)},
        {"param": "max_patches_per_image", "value": maxp_val},
        {"param": "n_patches_total", "value": str(len(patch_df))},
        {"param": "generated_at", "value": datetime.now().isoformat()},
    ])
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_df.to_csv(params_path, sep="\t", index=False)
    print(f"[patch_index_base] 参数记录: {params_path}")

    print(f"[patch_index_base] 完成！")
    print(f"[patch_index_base] 总 patch 数: {len(patch_df)}")
    print(f"[patch_index_base] 输出文件: {out_path}")


if __name__ == "__main__":
    main()

