#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HSI 1D-CNN baseline（基于 image_features.tsv / leaf_features.tsv）：
- 使用 ml_utils.load_dataset() 加载数据（特征列与 XGB 完全一致）
- 输入：image_features.tsv / leaf_features.tsv
- 目标列：例如 phase_core / session_id（通过 --target 指定）
- 模型：Conv1d 堆几层 + GlobalAveragePooling + 全连接
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score

# 导入 ml_utils 里的工具 + 画图函数
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ml_utils import (  # noqa: E402
    ensure_dir,
    load_dataset,
    plot_confusion_matrix,
    plot_pca_scatter,
    try_plot_roc_pr,
)


class SeqDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, L) 一维特征向量
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(0)  # (1, L)，作为 Conv1d 输入
        y = self.y[idx]
        return x, y


class ResidualBlock1D(nn.Module):
    """1D 残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class HSI1DCNN(nn.Module):
    """
    升级版 1D-CNN（带残差结构）：
    - 多个 Conv1d + BatchNorm1d + ReLU 的 block
    - block 内部带 shortcut（ResidualBlock1D）
    - 通道数逐步增加（32 → 64 → 128）
    - 最后用 GlobalAveragePooling1D 得到固定长度 embedding
    """
    def __init__(self, seq_len, n_classes, embedding_dim=128):
        super().__init__()
        # 初始卷积层
        self.conv_init = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn_init = nn.BatchNorm1d(32)
        
        # 残差块
        self.res_block1 = ResidualBlock1D(32, 64, stride=2)
        self.res_block2 = ResidualBlock1D(64, 128, stride=2)
        self.res_block3 = ResidualBlock1D(128, 128, stride=1)
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool1d(1)  # → (N, 128, 1)
        
        # 特征提取层（用于 forward_features）
        self.feature_fc = nn.Linear(128, embedding_dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, n_classes),
        )

    def forward_features(self, x):
        """
        返回倒数第二层的 embedding，shape [N, D]
        """
        # x: [N, 1, L]
        x = F.relu(self.bn_init(self.conv_init(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.gap(x)  # [N, 128, 1]
        x = x.view(x.size(0), -1)  # [N, 128]
        x = self.feature_fc(x)  # [N, embedding_dim]
        return x

    def forward(self, x):
        # x: [N, 1, L]
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images",
        required=True,
        help="image_features.tsv / leaf_features.tsv 路径（制表符分隔）",
    )
    ap.add_argument(
        "--sep",
        default="\t",
        help="特征文件分隔符，默认 TAB",
    )
    ap.add_argument(
        "--target",
        required=True,
        help="标签列名，例如 phase_core / session_id",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="输出目录，例如 results/hsi/dl/1dcnn_image",
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--device",
        default="cuda",
        help="cuda / cpu，默认 cuda（如果可用的话）",
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="测试集比例（train/test 划分），默认 0.2",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（同时传给 train_test_split 和 PyTorch）",
    )
    ap.add_argument(
        "--split-path",
        default=None,
        help="可选：split_phase_seedXX.tsv 路径，用于固定 train/test 划分",
    )
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20,
        help="early stopping 的耐心轮数（基于 test_f1_weighted），默认 20",
    )
    ap.add_argument(
        "--min-epochs",
        type=int,
        default=20,
        help="至少训练的 epoch 数，默认 20，避免过早停止",
    )
    return ap.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)

    outdir = ensure_dir(args.outdir)
    print(f"[1D-CNN] outdir = {outdir}")

    # ------- 复用 ml_utils.load_dataset（和 XGB 完全一致的特征 & 划分逻辑） -------
    X_train, X_test, y_train, y_test, classes, feat_cols, le = load_dataset(
        images_path=args.images,
        sep=args.sep,
        target=args.target,
        test_size=args.test_size,
        random_state=args.seed,
        split_path=args.split_path,
        feature_mode="spec",       # 关键：只用光谱列
    )

    print(
        f"[1D-CNN] #train={len(y_train)}, #test={len(y_test)}, "
        f"#classes={len(classes)}, seq_len={X_train.shape[1]}"
    )

    # 保存一下特征列和类别映射，方便以后复现
    pd.Series(feat_cols, name="feature").to_csv(
        outdir / "feature_columns.tsv", sep="\t", index=False
    )
    pd.Series(classes, name="class").to_csv(
        outdir / "classes.tsv", sep="\t", index=False
    )

    seq_len = X_train.shape[1]
    n_classes = len(classes)

    train_ds = SeqDataset(X_train, y_train)
    test_ds = SeqDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] 指定了 cuda 但当前不可用，退回 cpu")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = HSI1DCNN(seq_len, n_classes).to(device)

    # class weights（简单用 1/freq）
    class_counts = np.bincount(y_train, minlength=n_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.from_numpy(class_weights).float().to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_ds)

        # 每个 epoch 在 test 上评估一次
        model.eval()
        all_pred = []
        all_true = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                all_pred.append(pred)
                all_true.append(yb.numpy())
        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)

        f1w = f1_score(all_true, all_pred, average="weighted")
        f1m = f1_score(all_true, all_pred, average="macro")
        acc = accuracy_score(all_true, all_pred)

        print(
            f"[epoch {epoch:03d}] loss={avg_loss:.4f}, "
            f"test_f1w={f1w:.4f}, f1mac={f1m:.4f}, acc={acc:.4f}",
            flush=True,
        )

        if f1w > best_f1:
            best_f1 = f1w
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "f1w": f1w,
                "f1m": f1m,
                "acc": acc,
            }
            best_epoch = epoch

        # early stopping：基于 test 集 f1_weighted，先保证达到 min_epochs
        if epoch >= args.min_epochs:
            no_improve_epochs = epoch - best_epoch
            if no_improve_epochs >= args.early_stopping_patience:
                print(
                    f"[1D-CNN] early stopping at epoch {epoch} "
                    f"(best_epoch={best_epoch}, best_test_f1w={best_f1:.4f})"
                )
                break

    # 保存最好模型 & 报告 + 画图
    if best_state is not None:
        best_model_path = outdir / "best_model.pt"
        torch.save(best_state, best_model_path)

        # 再在 test 上出一次报告（用 best_state）
        model.load_state_dict(best_state["model"])
        model.eval()
        all_pred = []
        all_true = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                all_pred.append(pred)
                all_true.append(yb.numpy())
        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)

        report = classification_report(
            all_true,
            all_pred,
            labels=list(range(n_classes)),
            target_names=[str(c) for c in classes],
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).T.to_csv(
            outdir / "classification_report.tsv", sep="\t"
        )

        summary = {
            "best_epoch": best_state["epoch"],
            "best_test_f1_weighted": best_state["f1w"],
            "best_test_f1_macro": best_state["f1m"],
            "best_test_accuracy": best_state["acc"],
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_classes": int(n_classes),
        }
        pd.DataFrame([summary]).to_csv(
            outdir / "metrics.tsv", sep="\t", index=False
        )

        print("[1D-CNN] best:", summary)

        # ------------- 额外图：混淆矩阵 + PCA + ROC/PR -------------
        # 混淆矩阵（归一化）
        plot_confusion_matrix(
            y_true=all_true,
            y_pred=all_pred,
            class_names=[str(c) for c in classes],
            out_png=outdir / "1dcnn_confusion_matrix_norm.png",
            normalize=True,
        )

        # PCA 散点图（使用原始 X_test）
        y_test_str = np.array([str(classes[i]) for i in all_true])
        plot_pca_scatter(
            X_test,
            y_labels_str=y_test_str,
            out_png=outdir / "1dcnn_pca_scatter.png",
            title="PCA (test) - 1D-CNN",
        )

        # ROC / PR 曲线：写一个封装器，给 ml_utils 用
        class TorchProbWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device

            def predict_proba(self, X):
                self.model.eval()
                X = np.asarray(X, dtype=np.float32)
                X_tensor = torch.from_numpy(X).unsqueeze(1).to(self.device)  # (N,1,L)
                probs_list = []
                bs = 256
                with torch.no_grad():
                    for i in range(0, X_tensor.size(0), bs):
                        xb = X_tensor[i:i + bs]
                        logits = self.model(xb)
                        p = F.softmax(logits, dim=1)
                        probs_list.append(p.cpu().numpy())
                return np.concatenate(probs_list, axis=0)

        wrapper = TorchProbWrapper(model, device)
        try:
            try_plot_roc_pr(
                wrapper,
                X_test,
                y_test,
                out_prefix=str(outdir / "1dcnn_prob"),
            )
        except Exception as e:
            print(f"[1D-CNN][WARN] ROC/PR plot failed: {e}")
    else:
        print("[WARN] 没有保存 best_state，训练过程可能被异常中断。")


if __name__ == "__main__":
    main()

