# -*- coding: utf-8 -*-
"""
模型评估脚本：混淆矩阵 + 分类报告 + 训练曲线
用法: python src/evaluate.py --ckpt models/resnet18_finetune.pth
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import CLASSES, DEVICE, ROOT, build_model

RESULTS = ROOT / "results"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CN_NAMES = {"glioma": "胶质瘤", "meningioma": "脑膜瘤",
            "notumor": "无肿瘤", "pituitary": "垂体瘤"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = args.tag or Path(args.ckpt).stem

    model, *_ = build_model(args.arch)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(ROOT / "data" / "Testing", transform=tf)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(DEVICE)).argmax(1).cpu()
            y_true += y.tolist()
            y_pred += pred.tolist()

    labels = [CN_NAMES[c] for c in CLASSES]
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    print(report)
    (RESULTS / f"{tag}_report.txt").write_text(report, encoding="utf-8")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), labels, rotation=30)
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title(f"混淆矩阵 — {tag}")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{tag}_confusion_matrix.png", dpi=150)
    print(f"混淆矩阵已保存: results/{tag}_confusion_matrix.png")

    # 训练曲线（如果历史记录存在）
    for stage, hist_file in [("head", f"{args.arch}_history.json"),
                             ("finetune", f"{args.arch}_finetune_history.json")]:
        hp = RESULTS / hist_file
        if not hp.exists():
            continue
        hist = json.loads(hp.read_text(encoding="utf-8"))
        epochs = [h["epoch"] for h in hist if "epoch" in h]
        if not epochs:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, [h["train_acc"] for h in hist if "epoch" in h], "o-", label="训练集")
        ax.plot(epochs, [h["val_acc"] for h in hist if "epoch" in h], "s-", label="验证集")
        ax.set_xlabel("Epoch"); ax.set_ylabel("准确率")
        ax.set_title(f"训练曲线 — {args.arch} ({stage})")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / f"{args.arch}_{stage}_curve.png", dpi=150)
        print(f"训练曲线已保存: results/{args.arch}_{stage}_curve.png")


if __name__ == "__main__":
    main()
