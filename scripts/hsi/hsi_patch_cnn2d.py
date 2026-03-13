#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【HSI patch pipeline - 2D CNN 训练】

Patch-level 2D CNN:
- 输入: --index-tsv patch_index_phase_core_seed42.tsv
- 使用 HSIPatchDataset 从 cube_npz 中按坐标裁 patch
- Patch 形状假定为 [bands, H, W]，bands 作为通道
- 输出: best_model.pt + metrics.tsv
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# 尝试优先使用你已有的 HSIPatchDataset
try:
    # 尝试多种导入方式
    try:
        from hsi_patch_dataset import HSIPatchDataset as _ExtHSIPatchDataset, get_train_transform_2d
    except ImportError:
        # 如果直接导入失败，尝试从 scripts.hsi 导入
        import sys
        from pathlib import Path
        script_dir = Path(__file__).parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        from hsi_patch_dataset import HSIPatchDataset as _ExtHSIPatchDataset, get_train_transform_2d
    HAS_EXT_DS = True
except Exception as e:
    HAS_EXT_DS = False
    get_train_transform_2d = None
    print(f"[WARN] 无法导入 HSIPatchDataset: {e}", file=sys.stderr)


class HSIPatchDatasetFallback(Dataset):
    """
    简单版 HSIPatchDataset（Fallback）：
    - 支持两种模式：
      1. patch_npz 模式：index_tsv 包含 patch_npz 列（优先使用）
      2. cube 模式：index_tsv 包含 cube_npz, y0, x0, size 列（兼容旧格式）
    - 返回 x: [bands, H, W], y: int label
    """

    def __init__(self, index_tsv, split="train", bands=None):
        df = pd.read_csv(index_tsv, sep="\t")
        df.columns = [c.strip().lower() for c in df.columns]

        if "split" not in df.columns:
            raise RuntimeError("index_tsv 缺少 split 列")

        df = df[df["split"] == split].reset_index(drop=True)
        
        # 检查是否存在 patch_npz 列（新模式）
        if "patch_npz" in df.columns and df["patch_npz"].notna().any():
            # 过滤掉 patch_npz 为空的行
            df = df[df["patch_npz"].notna() & (df["patch_npz"] != "")].reset_index(drop=True)
            self.mode_type = "patch_npz"
            print(f"[HSIPatchDatasetFallback] 使用 patch_npz 模式（{len(df)} patches）")
        elif "cube_npz" in df.columns and "y0" in df.columns and "x0" in df.columns and "size" in df.columns:
            self.mode_type = "cube"
            print(f"[HSIPatchDatasetFallback] 使用 cube 模式（兼容旧格式，{len(df)} patches）")
        else:
            raise RuntimeError("index_tsv 必须包含 patch_npz 列，或包含 cube_npz/y0/x0/size 列")

        self.df = df
        self.bands = bands

        targets = df["target"].astype(str).tolist()
        classes = sorted(set(targets))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # 缓存（仅在 cube 模式下使用）
        self._cache_npz_path = None
        self._cache_cube = None

    def __len__(self):
        return len(self.df)

    def _load_cube(self, npz_path):
        if self._cache_npz_path == npz_path:
            return self._cache_cube

        data = np.load(npz_path, allow_pickle=True)
        if "R" in data:
            R = data["R"]
        else:
            first_key = list(data.keys())[0]
            R = data[first_key]
        # [H, W, B] -> [B, H, W]
        cube = np.moveaxis(R, -1, 0)
        if self.bands is not None:
            cube = cube[self.bands, :, :]

        self._cache_npz_path = npz_path
        self._cache_cube = cube
        return cube

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if self.mode_type == "patch_npz":
            # 直接从 patch npz 文件加载
            patch_npz_path = row["patch_npz"]
            if not patch_npz_path or pd.isna(patch_npz_path):
                raise RuntimeError(f"patch_npz 路径为空: row {idx}")
            
            data = np.load(patch_npz_path, allow_pickle=True)
            patch = data["R"]  # [bands, H, W]
            
            # 如果指定了 bands，进行筛选
            if self.bands is not None:
                patch = patch[self.bands, :, :]
            
            x = torch.from_numpy(patch).float()
        
        elif self.mode_type == "cube":
            # 兼容旧模式：从 cube_npz 动态裁剪
            npz_path = row["cube_npz"]
            y0 = int(row["y0"])
            x0 = int(row["x0"])
            size = int(row["size"])

            cube = self._load_cube(npz_path)
            patch = cube[:, y0:y0+size, x0:x0+size]  # [bands, H, W]
            x = torch.from_numpy(patch).float()
        else:
            raise RuntimeError(f"unknown mode_type: {self.mode_type}")

        label_str = str(row["target"])
        y = self.class_to_idx[label_str]
        y = torch.tensor(y, dtype=torch.long)

        return x, y


class ResidualBlock2D(nn.Module):
    """2D 残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class HSIPatchNet2D(nn.Module):
    """
    升级版 2D CNN（小型 ResNet 风格）：
    - 先用 1x1 conv 压缩光谱通道（从 B 波段降到 32 个通道）
    - 堆 2-3 个残差 block
    - GAP 后接全连接
    """

    def __init__(self, in_channels, n_classes, embedding_dim=128):
        super().__init__()
        # 光谱通道降维
        self.conv_reduce = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.bn_reduce = nn.BatchNorm2d(32)

        # 残差块
        self.res_block1 = ResidualBlock2D(32, 64, stride=1)
        self.res_block2 = ResidualBlock2D(64, 128, stride=2)
        self.res_block3 = ResidualBlock2D(128, 128, stride=1)
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 特征提取层（用于 forward_features）
        self.feature_fc = nn.Linear(128, embedding_dim)
        self.dropout = nn.Dropout(0.3)  # 分类头前 Dropout，缓解过拟合

        # 分类头
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward_features(self, x):
        """
        返回倒数第二层的 embedding，shape [N, D]
        """
        # x: [B, C, H, W]
        x = F.relu(self.bn_reduce(self.conv_reduce(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.gap(x)  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 128]
        x = self.feature_fc(x)  # [B, embedding_dim]
        return x

    def forward(self, x):
        # x: [B, C, H, W]
        features = self.forward_features(x)
        logits = self.classifier(self.dropout(features))
        return logits


def get_datasets(index_tsv, augment=True, augment_noise_std=0.0, seed=42):
    """
    获取训练和测试数据集。
    augment: 是否为 train 集启用轻量增强（水平/垂直翻转 + 可选光谱噪声）。
    """
    if not HAS_EXT_DS:
        raise RuntimeError(
            "无法导入 HSIPatchDataset。"
            "请确保 scripts/hsi/hsi_patch_dataset.py 存在且可导入。"
            "新流程必须使用 patch_cubes 模式，不再支持 Fallback。"
        )
    train_transform = None
    if augment and get_train_transform_2d is not None:
        train_transform = get_train_transform_2d(noise_std=augment_noise_std, seed=seed)
    try:
        train_ds = _ExtHSIPatchDataset(index_tsv, split="train", mode="2d", transform=train_transform)
        test_ds = _ExtHSIPatchDataset(index_tsv, split="test", mode="2d")
        print(f"[2dcnn] 使用 HSIPatchDataset (mode=2d, patch_cubes 模式, train_augment={augment})")
        return train_ds, test_ds
    except Exception as e:
        raise RuntimeError(
            f"使用 HSIPatchDataset 失败: {e}\n"
            "请确保 index_tsv 包含 cube_patch_npz 和 patch_idx 列（patch_cubes 模式）。"
        ) from e


def train_epoch(model, loader, device, optimizer, criterion, use_amp=False, scaler=None, accum_steps=1):
    model.train()
    all_y_true = []
    all_y_pred = []
    total_loss = 0.0

    optimizer.zero_grad()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y) / accum_steps
            scaler.scale(loss).backward()
        else:
            logits = model(x)
            loss = criterion(logits, y) / accum_steps
            loss.backward()

        total_loss += loss.item() * accum_steps * x.size(0)

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
        all_y_true.append(y.detach().cpu().numpy())
        all_y_pred.append(preds.detach().cpu().numpy())

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    acc = accuracy_score(all_y_true, all_y_pred)
    f1_w = f1_score(all_y_true, all_y_pred, average="weighted")
    f1_m = f1_score(all_y_true, all_y_pred, average="macro")
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, acc, f1_w, f1_m


@torch.no_grad()
def eval_epoch(model, loader, device, criterion, use_amp=False):
    model.eval()
    all_y_true = []
    all_y_pred = []
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
        else:
            logits = model(x)
            loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        all_y_true.append(y.detach().cpu().numpy())
        all_y_pred.append(preds.detach().cpu().numpy())

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    acc = accuracy_score(all_y_true, all_y_pred)
    f1_w = f1_score(all_y_true, all_y_pred, average="weighted")
    f1_m = f1_score(all_y_true, all_y_pred, average="macro")
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, acc, f1_w, f1_m


def export_patch_embeddings(model, index_tsv, out_tsv, batch_size, num_workers, pin_memory, device, use_amp=False):
    """
    导出所有 patch 的 embedding 到 TSV 文件
    use_amp: 为 True 时用 FP16 推理，节省显存
    """
    model.eval()
    
    # 使用 split="all" 加载所有数据
    dataset = _ExtHSIPatchDataset(
        index_tsv=index_tsv,
        split="all",
        mode="2d",
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 保持顺序一致
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    all_emb = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    emb = model.forward_features(x)  # [N, D]
            else:
                emb = model.forward_features(x)
            all_emb.append(emb.float().cpu().numpy())
    
    # 拼接所有 embedding
    emb_array = np.concatenate(all_emb, axis=0)  # [N_total, D]
    emb_dim = emb_array.shape[1]
    
    # 获取 meta 信息（顺序与 DataLoader 一致）
    meta_df = dataset.meta_df.reset_index(drop=True)
    
    # 确保 meta_df 的行数与 embedding 数量一致
    if len(meta_df) != len(emb_array):
        raise RuntimeError(
            f"meta_df 行数 ({len(meta_df)}) 与 embedding 数量 ({len(emb_array)}) 不一致"
        )
    
    # 构造输出 DataFrame：meta 列 + emb_0...emb_{D-1}
    output_df = meta_df.copy()
    for i in range(emb_dim):
        output_df[f"emb_{i}"] = emb_array[:, i]
    
    # 确保包含必要的列
    required_cols = ["patch_id", "source_sample_id", "split", "target"]
    for col in required_cols:
        if col not in output_df.columns:
            if col == "patch_id":
                output_df["patch_id"] = output_df.index.astype(str)
            elif col == "source_sample_id":
                output_df["source_sample_id"] = "unknown"
            elif col == "split":
                # 如果 split 列不存在，尝试从 dataset.meta_df 获取
                if "split" in dataset.meta_df.columns:
                    output_df["split"] = dataset.meta_df["split"].values
                else:
                    output_df["split"] = "unknown"
            elif col == "target":
                if "target" in output_df.columns:
                    pass  # 已存在
                else:
                    output_df["target"] = "unknown"
    
    # 选择要输出的列：meta 列 + embedding 列
    output_cols = required_cols + [f"emb_{i}" for i in range(emb_dim)]
    output_df = output_df[output_cols]
    
    output_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"[2dcnn] 导出 {len(output_df)} 个 patch 的 embedding（维度 {emb_dim}）到 {out_tsv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-tsv", required=True,
                    help="patch_index_phase_core_seed42.tsv")
    ap.add_argument("--outdir", required=True,
                    help="results/hsi/patch/2dcnn_phase_core")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=5e-4,
                    help="Adam 优化器的 L2 正则系数，默认 5e-4，缓解过拟合")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader num_workers，默认 0，避免多进程 OOM")
    ap.add_argument("--no-pin-memory", action="store_true",
                    help="禁用 pin_memory，默认使用 pin_memory=True")
    ap.add_argument("--amp", action="store_true",
                    help="使用混合精度 (FP16) 训练，显著降低显存")
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1,
                    help="梯度累积步数，与 batch_size 配合可减小显存；有效 batch = batch_size * 此值")
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20,
        help="early stopping 的耐心轮数（基于 val_f1_weighted），默认 20",
    )
    ap.add_argument(
        "--min-epochs",
        type=int,
        default=20,
        help="至少训练的 epoch 数，默认 20，避免过早停止",
    )
    ap.add_argument("--no-augment", action="store_true",
                    help="禁用训练集数据增强（默认启用：水平/垂直翻转）")
    ap.add_argument("--augment-noise-std", type=float, default=0.0,
                    help="训练增强光谱高斯噪声标准差，默认 0（不启用）")

    args = ap.parse_args()
    
    num_workers = args.num_workers
    pin_memory = not args.no_pin_memory

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    use_amp = getattr(args, "amp", False)
    accum_steps = max(1, getattr(args, "gradient_accumulation_steps", 1))
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None
    print(f"[2dcnn] device: {device}, amp={use_amp}, gradient_accumulation_steps={accum_steps}")

    train_ds, test_ds = get_datasets(
        args.index_tsv,
        augment=not getattr(args, "no_augment", False),
        augment_noise_std=getattr(args, "augment_noise_std", 0.0),
        seed=args.seed,
    )
    n_classes = len(getattr(train_ds, "class_to_idx"))
    print(f"[2dcnn] n_train={len(train_ds)}, n_test={len(test_ds)}, n_classes={n_classes}")

    # DataLoader 使用命令行参数
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print(f"[2dcnn] DataLoader num_workers={num_workers}, pin_memory={pin_memory}")

    # 取一个 batch 确定输入通道数
    example_x, _ = next(iter(train_loader))
    in_channels = example_x.shape[1]

    model = HSIPatchNet2D(in_channels=in_channels, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_f1w = -1.0
    best_state = None
    history = []
    best_epoch = 0
    
    import time

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc, train_f1w, train_f1m = train_epoch(
            model, train_loader, device, optimizer, criterion,
            use_amp=use_amp, scaler=scaler, accum_steps=accum_steps,
        )
        val_loss, val_acc, val_f1w, val_f1m = eval_epoch(
            model, test_loader, device, criterion, use_amp=use_amp,
        )
        
        epoch_time = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "epoch_time_sec": epoch_time,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1_weighted": train_f1w,
            "train_f1_macro": train_f1m,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_weighted": val_f1w,
            "val_f1_macro": val_f1m,
        })

        print(f"[2dcnn][epoch {epoch}] time={epoch_time:.1f}s, "
              f"train_loss={train_loss:.4f}, train_f1w={train_f1w:.4f}, "
              f"val_loss={val_loss:.4f}, val_f1w={val_f1w:.4f}, val_acc={val_acc:.4f}")

        if val_f1w > best_f1w:
            best_f1w = val_f1w
            best_state = model.state_dict()
            best_epoch = epoch

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # early stopping：基于 val_f1_weighted，先保证达到 min_epochs
        if epoch >= args.min_epochs:
            no_improve_epochs = epoch - best_epoch
            if no_improve_epochs >= args.early_stopping_patience:
                print(
                    f"[2dcnn] early stopping at epoch {epoch} "
                    f"(best_epoch={best_epoch}, best_val_f1w={best_f1w:.4f})"
                )
                break

    # 用 best_state 重新评估一次
    if best_state is not None:
        model.load_state_dict(best_state)

    _, val_acc, val_f1w, val_f1m = eval_epoch(
        model, test_loader, device, criterion, use_amp=use_amp,
    )

    # 保存模型
    model_path = outdir / "best_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_classes": n_classes,
        "in_channels": in_channels,
        "best_val_f1_weighted": float(val_f1w),
        "best_val_f1_macro": float(val_f1m),
        "val_acc": float(val_acc),
    }, model_path)

    # 保存 metrics.tsv（单行）
    metrics_path = outdir / "metrics.tsv"
    mdf = pd.DataFrame([{
        "model": "HSIPatchNet2D",
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "n_classes": n_classes,
        "val_accuracy": float(val_acc),
        "val_f1_weighted": float(val_f1w),
        "val_f1_macro": float(val_f1m),
    }])
    mdf.to_csv(metrics_path, sep="\t", index=False)

    # 运行参数留痕，便于追溯“这次结果用什么参数跑的”
    import re
    patch_target_from_path = "unknown"
    m = re.search(r"patch_index_([^_]+(?:_[^_]+)*)_seed\d+\.tsv", str(args.index_tsv))
    if m:
        patch_target_from_path = m.group(1)
    run_params = pd.DataFrame([
        {"param": "index_tsv", "value": str(args.index_tsv)},
        {"param": "patch_target", "value": patch_target_from_path},
        {"param": "seed", "value": str(args.seed)},
        {"param": "batch_size", "value": str(args.batch_size)},
        {"param": "epochs", "value": str(args.epochs)},
        {"param": "weight_decay", "value": str(args.weight_decay)},
        {"param": "augment", "value": str(not getattr(args, "no_augment", False))},
        {"param": "augment_noise_std", "value": str(getattr(args, "augment_noise_std", 0.0))},
        {"param": "early_stopping_patience", "value": str(args.early_stopping_patience)},
        {"param": "min_epochs", "value": str(args.min_epochs)},
        {"param": "lr", "value": str(args.lr)},
    ])
    run_params.to_csv(outdir / "run_params.tsv", sep="\t", index=False)

    # 保存完整训练过程并绘制 train/val 曲线
    hist_path = outdir / "history.tsv"
    pd.DataFrame(history).to_csv(hist_path, sep="\t", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = pd.DataFrame(history)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.plot(h["epoch"], h["train_loss"], label="train_loss")
        ax1.plot(h["epoch"], h["val_loss"], label="val_loss")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(h["epoch"], h["train_f1_weighted"], label="train_f1_weighted")
        ax2.plot(h["epoch"], h["val_f1_weighted"], label="val_f1_weighted")
        ax2.set_ylabel("F1 weighted")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig.savefig(outdir / "train_val_curves.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[2dcnn] train_val_curves -> {outdir / 'train_val_curves.png'}")
    except Exception as e:
        print(f"[2dcnn] 未绘制 train_val_curves: {e}")

    print(f"[2dcnn] best_model -> {model_path}")
    print(f"[2dcnn] metrics -> {metrics_path}")
    
    # 导出 patch embeddings
    emb_out = outdir / "patch_embeddings_2d.tsv"
    export_patch_embeddings(
        model=model,
        index_tsv=args.index_tsv,
        out_tsv=emb_out,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        device=device,
        use_amp=use_amp,
    )
    print(f"[2dcnn] patch embeddings -> {emb_out}")


if __name__ == "__main__":
    main()

