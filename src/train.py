# -*- coding: utf-8 -*-
"""
脑肿瘤 MRI 图像分类 —— 迁移学习训练脚本
四分类: glioma(胶质瘤) / meningioma(脑膜瘤) / notumor(无肿瘤) / pituitary(垂体瘤)

两阶段训练（适合 CPU）：
  --stage head     冻结骨干网络，只训练分类头（快，先跑这个）
  --stage finetune 解冻最后一个残差块继续微调（在 head 基础上）

用法示例：
  python src/train.py --stage head --epochs 3
  python src/train.py --stage finetune --epochs 3
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(arch: str, num_classes: int = 4):
    """构建迁移学习模型，返回 (model, 骨干参数组, 头部参数组)"""
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)
        head_params = list(m.fc.parameters())
        backbone_params = [p for n, p in m.named_parameters() if not n.startswith("fc")]
        last_block = m.layer4
    elif arch == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)
        head_params = list(m.fc.parameters())
        backbone_params = [p for n, p in m.named_parameters() if not n.startswith("fc")]
        last_block = m.layer4
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)
        head_params = list(m.classifier.parameters())
        backbone_params = [p for n, p in m.named_parameters() if not n.startswith("classifier")]
        last_block = m.features[-1]
    else:
        raise ValueError(f"未知模型: {arch}")
    return m, backbone_params, head_params, last_block


EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_loaders(batch_size: int, seed: int = 42):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # 验证集使用与测试集一致的固定预处理（不带随机增强），模型选择更稳定
    train_full = datasets.ImageFolder(DATA / "Training", transform=train_tf)
    val_full = datasets.ImageFolder(DATA / "Training", transform=EVAL_TF)
    test_ds = datasets.ImageFolder(DATA / "Testing", transform=EVAL_TF)
    # 从训练集划 10% 做验证集（同一批下标，两套预处理）
    val_size = int(0.1 * len(train_full))
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_split = torch.utils.data.random_split(
        train_full, [len(train_full) - val_size, val_size], generator=gen)
    val_ds = torch.utils.data.Subset(val_full, val_split.indices)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_ld, val_ld, test_ld


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "efficientnet_b0"])
    ap.add_argument("--stage", default="head", choices=["head", "finetune"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42, help="随机种子（划分/初始化/增强）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"设备: {DEVICE} | 模型: {args.arch} | 阶段: {args.stage} | 种子: {args.seed}")
    model, backbone_params, head_params, last_block = build_model(args.arch)
    model = model.to(DEVICE)

    ckpt = MODELS / f"{args.arch}_head.pth"
    if args.stage == "head":
        for p in backbone_params:
            p.requires_grad = False
        optimizer = torch.optim.Adam(head_params, lr=args.lr or 1e-3)
        history_path = RESULTS / f"{args.arch}_history.json"
    else:
        assert ckpt.exists(), f"先运行 head 阶段: python src/train.py --stage head"
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        for p in backbone_params:
            p.requires_grad = False
        for p in last_block.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(
            list(last_block.parameters()) + head_params, lr=args.lr or 1e-4)
        history_path = RESULTS / f"{args.arch}_finetune_history.json"

    train_ld, val_ld, test_ld = get_loaders(args.batch, seed=args.seed)
    criterion = nn.CrossEntropyLoss()
    history = []

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_ld, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_ld, criterion)
        dt = time.time() - t0
        print(f"Epoch {ep}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.0f}s", flush=True)
        history.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})

    # 测试集评估
    te_loss, te_acc = run_epoch(model, test_ld, criterion)
    print(f"\n测试集: loss {te_loss:.4f} | acc {te_acc:.4f}")

    # 保存
    out_ckpt = ckpt if args.stage == "head" else MODELS / f"{args.arch}_finetune.pth"
    torch.save(model.state_dict(), out_ckpt)
    history.append({"test_loss": te_loss, "test_acc": te_acc, "seed": args.seed})
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"模型已保存: {out_ckpt}")
    print(f"训练记录: {history_path}")


if __name__ == "__main__":
    main()
