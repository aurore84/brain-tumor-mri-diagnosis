# -*- coding: utf-8 -*-
"""
小样本学习曲线实验（含对照组）
认知科学动机：人类医生看几十例就能鉴别肿瘤，机器需要多少样本？
关键对照：特征来自哪里决定了小样本效率——
  finetuned  微调后骨干（含任务相关视觉经验）
  pretrained ImageNet 预训练未微调（含通用视觉经验）
  random     随机初始化（无视觉经验 ← 对照组）
用法: python src/fewshot.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from train import CLASSES, DEVICE, ROOT, build_model
from hopfield import get_backbone, hopfield_logits

RESULTS = ROOT / "results"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RATIOS = [0.01, 0.05, 0.10, 0.25, 1.00]
BACKBONES = ["finetuned", "pretrained", "random"]
BB_LABEL = {"finetuned": "微调骨干（任务视觉经验）",
            "pretrained": "预训练骨干（通用视觉经验）",
            "random": "随机初始化（无视觉经验，对照组）"}
BB_STYLE = {"finetuned": ("o-", "#684765"),
            "pretrained": ("^-", "#B89A3A"),
            "random": ("v--", "#9B9489")}


def get_bb(name):
    if name == "finetuned":
        return get_backbone()
    if name == "pretrained":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        m = models.resnet18(weights=None)
    m.fc = nn.Identity()
    return m.to(DEVICE).eval()


def stratified_subset(X, y, ratio, seed=42):
    g = torch.Generator().manual_seed(seed)
    idx = []
    for c in range(4):
        cidx = (y == c).nonzero(as_tuple=True)[0]
        k = max(1, int(len(cidx) * ratio))
        idx.append(cidx[torch.randperm(len(cidx), generator=g)[:k]])
    return torch.cat(idx)


@torch.no_grad()
def extract(backbone, loader):
    feats, labels = [], []
    for x, y in loader:
        feats.append(backbone(x.to(DEVICE)).cpu())
        labels.append(y)
    return F.normalize(torch.cat(feats), dim=1), torch.cat(labels)


def train_fc(X, y, Xte, yte, epochs=200):
    head = nn.Linear(512, 4).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    X, y = X.to(DEVICE), y.to(DEVICE)
    for _ in range(epochs):
        loss = crit(head(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (head(Xte.to(DEVICE)).argmax(1).cpu() == yte).float().mean().item()


def proto_acc(X, y, Xte, yte, beta=8):
    protos = torch.stack([F.normalize(X[y == c].mean(0), dim=0) for c in range(4)])
    pred = hopfield_logits(Xte.to(DEVICE), protos.to(DEVICE), beta).argmax(1).cpu()
    return (pred == yte).float().mean().item()


def main():
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ld = DataLoader(datasets.ImageFolder(ROOT / "data" / "Training", transform=tf),
                          batch_size=64, shuffle=False)
    test_ld = DataLoader(datasets.ImageFolder(ROOT / "data" / "Testing", transform=tf),
                         batch_size=64, shuffle=False)

    all_rows = []
    feats_cache = {}
    for bb in BACKBONES:
        print(f"提取 {bb} 骨干特征...", flush=True)
        backbone = get_bb(bb)
        Xtr, ytr = extract(backbone, train_ld)
        Xte, yte = extract(backbone, test_ld)
        feats_cache[bb] = (Xtr, ytr, Xte, yte)
        del backbone

    for bb in BACKBONES:
        Xtr, ytr, Xte, yte = feats_cache[bb]
        for r in RATIOS:
            idx = stratified_subset(Xtr, ytr, r)
            Xs, ys = Xtr[idx], ytr[idx]
            row = {"backbone": bb, "ratio": r, "per_class": int((ys == 0).sum()),
                   "fc_acc": train_fc(Xs, ys, Xte, yte)}
            if bb == "finetuned":
                row["hopfield_acc"] = proto_acc(Xs, ys, Xte, yte)
            all_rows.append(row)
            print(f"{bb:>10} | {r*100:>5.0f}%（每类 {row['per_class']:>4}）| fc {row['fc_acc']:.4f}"
                  + (f" | hopfield {row['hopfield_acc']:.4f}" if "hopfield_acc" in row else ""),
                  flush=True)

    # 画图
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = [row["per_class"] for row in all_rows if row["backbone"] == "finetuned"]
    for bb in BACKBONES:
        style, color = BB_STYLE[bb]
        ys_plot = [row["fc_acc"] * 100 for row in all_rows if row["backbone"] == bb]
        ax.plot(x, ys_plot, style, color=color, label=f"判别式头 @ {BB_LABEL[bb]}", linewidth=2)
    hp = [row["hopfield_acc"] * 100 for row in all_rows
          if row["backbone"] == "finetuned" and "hopfield_acc" in row]
    ax.plot(x[:len(hp)], hp, "s--", color="#C9828B",
            label="记忆原型头 @ 微调骨干（Hopfield）", linewidth=2)
    ax.axvspan(10, 60, color="#C9828B", alpha=0.12)
    ymax = max(row["fc_acc"] for row in all_rows) * 100
    ax.text(24, 35, "人类医生\n≈数十例即可上手", fontsize=10, color="#684765", ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("每类训练样本数（对数轴）")
    ax.set_ylabel("测试准确率 (%)")
    ax.set_title("小样本学习曲线：视觉经验（预训练）让机器像人一样举一反三")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "fewshot_curve.png", dpi=150)
    (RESULTS / "fewshot_result.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已保存: results/fewshot_curve.png, results/fewshot_result.json")


if __name__ == "__main__":
    main()
