# scripts/hsi/hsi_patch_dataset.py
# 【HSI patch pipeline - PyTorch Dataset】
#
# 用于从 patch_cubes（一 cube 一个 patch 集合文件）加载数据
# 支持 2D/3D CNN 训练，使用 memory-mapped 模式减少 I/O 开销

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class _PatchTransform2D:
    """轻量数据增强：水平/垂直翻转（各 0.5 概率）+ 可选光谱高斯噪声。仅用于 train。"""

    def __init__(self, noise_std=0.0, seed=None):
        self.noise_std = float(noise_std)
        self._rng = random.Random(seed)

    def __call__(self, x):
        # x: [C, H, W]
        if self._rng.random() < 0.5:
            x = torch.flip(x, dims=[-1])  # 水平翻转
        if self._rng.random() < 0.5:
            x = torch.flip(x, dims=[-2])  # 垂直翻转
        if self.noise_std > 0:
            x = x + torch.randn_like(x, device=x.device, dtype=x.dtype) * self.noise_std
        return x


class _PatchTransform3D:
    """轻量数据增强：水平/垂直翻转（各 0.5 概率）+ 可选光谱高斯噪声。仅用于 train，输入 [1, C, H, W]。"""

    def __init__(self, noise_std=0.0, seed=None):
        self.noise_std = float(noise_std)
        self._rng = random.Random(seed)

    def __call__(self, x):
        # x: [1, C, H, W]
        if self._rng.random() < 0.5:
            x = torch.flip(x, dims=[-1])  # 水平翻转
        if self._rng.random() < 0.5:
            x = torch.flip(x, dims=[-2])  # 垂直翻转
        if self.noise_std > 0:
            x = x + torch.randn_like(x, device=x.device, dtype=x.dtype) * self.noise_std
        return x


def get_train_transform_2d(noise_std=0.0, seed=42):
    """返回 2D patch 训练用 transform（水平/垂直翻转 + 可选光谱噪声）。仅在 train 时使用。"""
    return _PatchTransform2D(noise_std=noise_std, seed=seed)


def get_train_transform_3d(noise_std=0.0, seed=42):
    """返回 3D patch 训练用 transform（水平/垂直翻转 + 可选光谱噪声）。仅在 train 时使用。"""
    return _PatchTransform3D(noise_std=noise_std, seed=seed)


class HSIPatchDataset(Dataset):
    def __init__(self, index_tsv, split="train",
                 mode="2d", bands=None, transform=None):
        """
        index_tsv: patch_index_{target}_seed{seed}.tsv（包含 cube_patch_npz 和 patch_idx）
        split: "train", "test" 或 "all"（"all" 表示不按 split 过滤，使用全部行）
        mode: "2d" 或 "3d"
        bands: 可选，指定要用的谱段索引；None 表示用所有
        transform: 可选，对 tensor 做数据增强
        
        支持的索引格式（按优先级）：
        1. cube_patch_npz + patch_idx（新模式，推荐）：从 patch_cubes 文件读取
        2. patch_npz（旧模式，deprecated）：每个 patch 一个文件
        3. cube_npz + y0/x0/size（兼容模式）：从 cube 动态裁剪
        """
        df = pd.read_csv(index_tsv, sep="\t")
        df.columns = [c.strip().lower() for c in df.columns]

        if "split" not in df.columns:
            raise RuntimeError("index_tsv 缺少 split 列")

        # 根据 split 参数过滤数据
        if split == "all":
            # 不按 split 过滤，使用全部行，但保持原始顺序
            df = df.reset_index(drop=True)
        else:
            df = df[df["split"] == split].reset_index(drop=True)
        
        self.df = df
        self.mode = mode
        self.bands = bands
        self.transform = transform

        # 检查索引格式（按优先级）
        if "cube_patch_npz" in df.columns and "patch_idx" in df.columns:
            # 新模式：patch_cubes（推荐）
            df = df[df["cube_patch_npz"].notna() & (df["cube_patch_npz"] != "")].reset_index(drop=True)
            self.mode_type = "patch_cubes"
            print(f"[HSIPatchDataset] 使用 patch_cubes 模式（{len(df)} patches）")
        elif "patch_npz" in df.columns and df["patch_npz"].notna().any():
            # 旧模式：每个 patch 一个文件（deprecated）
            df = df[df["patch_npz"].notna() & (df["patch_npz"] != "")].reset_index(drop=True)
            self.mode_type = "patch_npz"
            print(f"[HSIPatchDataset] 使用 patch_npz 模式（deprecated，{len(df)} patches）")
        elif "cube_npz" in df.columns and "y0" in df.columns and "x0" in df.columns and "size" in df.columns:
            # 兼容模式：从 cube 动态裁剪
            self.mode_type = "cube"
            print(f"[HSIPatchDataset] 使用 cube 模式（兼容旧格式，{len(df)} patches）")
        else:
            raise RuntimeError(
                "index_tsv 必须包含以下列之一：\n"
                "  - cube_patch_npz + patch_idx（推荐）\n"
                "  - patch_npz（deprecated）\n"
                "  - cube_npz + y0 + x0 + size（兼容）"
            )

        self.df = df

        targets = df["target"].astype(str).tolist()
        classes = sorted(set(targets))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # 维护 meta_df：包含 patch_id, source_sample_id, split, target 等信息
        # 用于后续导出 embedding 时保持顺序一致
        self.meta_df = df.copy()
        # 确保包含必要的列（如果不存在则添加默认值）
        if "patch_id" not in self.meta_df.columns:
            # 如果没有 patch_id，尝试从其他列构造或使用索引
            if "patch_idx" in self.meta_df.columns and "cube_patch_npz" in self.meta_df.columns:
                # 使用 cube_patch_npz 和 patch_idx 构造唯一 ID
                self.meta_df["patch_id"] = (
                    self.meta_df["cube_patch_npz"].astype(str) + "_" + 
                    self.meta_df["patch_idx"].astype(str)
                )
            else:
                self.meta_df["patch_id"] = self.meta_df.index.astype(str)
        if "source_sample_id" not in self.meta_df.columns:
            # 尝试从其他列推断，或使用默认值
            if "sample_id" in self.meta_df.columns:
                self.meta_df["source_sample_id"] = self.meta_df["sample_id"]
            elif "cube_patch_npz" in self.meta_df.columns:
                # 从 cube_patch_npz 路径提取样本 ID
                self.meta_df["source_sample_id"] = self.meta_df["cube_patch_npz"].apply(
                    lambda x: str(x).split("/")[-1].replace(".npz", "") if pd.notna(x) else "unknown"
                )
            else:
                self.meta_df["source_sample_id"] = "unknown"

        # IO cache：缓存已加载的 patch_cubes 文件
        # key: cube_patch_npz path, value: patches ndarray (memory-mapped)
        self._cube_cache = {}
        
        # 兼容旧模式的缓存
        self._cache_npz_path = None
        self._cache_cube = None

    def __len__(self):
        return len(self.df)

    def _load_cube(self, npz_path):
        if self._cache_npz_path == npz_path:
            return self._cache_cube

        data = np.load(npz_path, allow_pickle=True)
        R = data["R"]  # [H, W, B]
        # H, W, B -> B, H, W
        cube = np.moveaxis(R, -1, 0)  # [bands, H, W]

        if self.bands is not None:
            cube = cube[self.bands, :, :]

        self._cache_npz_path = npz_path
        self._cache_cube = cube
        return cube

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if self.mode_type == "patch_cubes":
            # 新模式：从 patch_cubes 文件读取（推荐）
            cube_file = row["cube_patch_npz"]
            local_idx = int(row["patch_idx"])
            
            # 检查 cache
            if cube_file not in self._cube_cache:
                # 使用 memory-mapped 模式加载，避免一次性读入内存
                data = np.load(cube_file, mmap_mode="r", allow_pickle=True)
                patches = data["patches"]  # [N_patches, B, H, W]
                self._cube_cache[cube_file] = patches
            
            patches = self._cube_cache[cube_file]
            patch = patches[local_idx]  # [B, H, W]
            
            # 如果指定了 bands，进行筛选
            if self.bands is not None:
                patch = patch[self.bands, :, :]
            
            if self.mode == "2d":
                x = torch.from_numpy(patch).float()  # [B, H, W]
            elif self.mode == "3d":
                x = torch.from_numpy(patch).float().unsqueeze(0)  # [1, B, H, W]
            else:
                raise ValueError(f"unknown mode: {self.mode}")
        
        elif self.mode_type == "patch_npz":
            # 旧模式：每个 patch 一个文件（deprecated）
            patch_npz_path = row["patch_npz"]
            if not patch_npz_path or pd.isna(patch_npz_path):
                raise RuntimeError(f"patch_npz 路径为空: row {idx}")
            
            data = np.load(patch_npz_path, allow_pickle=True)
            patch = data["R"]  # [bands, H, W]
            
            # 如果指定了 bands，进行筛选
            if self.bands is not None:
                patch = patch[self.bands, :, :]
            
            if self.mode == "2d":
                x = torch.from_numpy(patch).float()  # [bands, H, W]
            elif self.mode == "3d":
                x = torch.from_numpy(patch).float().unsqueeze(0)  # [1, bands, H, W]
            else:
                raise ValueError(f"unknown mode: {self.mode}")
        
        elif self.mode_type == "cube":
            # 兼容模式：从 cube_npz 动态裁剪
            npz_path = row["cube_npz"]
            y0 = int(row["y0"])
            x0 = int(row["x0"])
            size = int(row["size"])

            cube = self._load_cube(npz_path)  # [bands, H, W]
            patch = cube[:, y0:y0+size, x0:x0+size]  # [bands, size, size]

            if self.mode == "2d":
                x = torch.from_numpy(patch).float()  # [bands, H, W]
            elif self.mode == "3d":
                x = torch.from_numpy(patch).float().unsqueeze(0)  # [1, bands, H, W]
            else:
                raise ValueError(f"unknown mode: {self.mode}")
        else:
            raise RuntimeError(f"unknown mode_type: {self.mode_type}")

        label_str = str(row["target"])
        y = torch.tensor(self.class_to_idx[label_str], dtype=torch.long)

        if self.transform is not None:
            x = self.transform(x)

        return x, y

