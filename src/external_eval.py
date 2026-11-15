# -*- coding: utf-8 -*-
"""
外部验证实验：在独立来源数据集上评估模型泛化能力
（模型不重新训练，直接跨数据集测试 —— 模拟"换一家医院的数据"）

用法: python src/external_eval.py --ckpt models/resnet18_finetune.pth
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import CLASSES, DEVICE, ROOT, build_model

RESULTS = ROOT / "results"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CN_NAMES = {"glioma": "胶质瘤", "meningioma": "脑膜瘤",
            "notumor": "无肿瘤", "pituitary": "垂体瘤"}

# 外部数据集文件夹名 → 本系统类别名
FOLDER_MAP = {
    "glioma_tumor": "glioma",
    "meningioma_tumor": "meningioma",
    "normal": "notumor",
    "pituitary_tumor": "pituitary",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    tag = f"{args.arch}_external"

    model, *_ = build_model(args.arch)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(ROOT / "data_external", transform=tf)
    # ImageFolder 按文件夹名字母序给标签，需要映射回我们的类别序号
    folder_to_idx = {name: CLASSES.index(FOLDER_MAP[name])
                     for name in ds.classes}
    remap = {orig: folder_to_idx[folder] for folder, orig in
             zip(ds.classes, range(len(ds.classes)))}
    print("外部数据集类别映射:", ds.classes, "->", remap)

    loader = DataLoader(ds, batch_size=64, shuffle=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(DEVICE)).argmax(1).cpu()
            y_pred += pred.tolist()
            y_true += [remap[t] for t in y.tolist()]

    acc = accuracy_score(y_true, y_pred)
    labels = [CN_NAMES[c] for c in CLASSES]
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    print(f"\n外部验证总体准确率: {acc:.4f}  (n={len(y_true)})")
    print(report)

    out = {"overall_acc": acc, "n": len(y_true), "report": report}
    (RESULTS / f"{tag}_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), labels, rotation=30)
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title(f"外部验证混淆矩阵（独立数据集, n={len(y_true)}）")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{tag}_confusion_matrix.png", dpi=150)
    print(f"结果已保存: results/{tag}_confusion_matrix.png")


if __name__ == "__main__":
    main()
