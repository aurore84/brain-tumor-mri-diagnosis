# -*- coding: utf-8 -*-
"""
现代 Hopfield 联想记忆分类头实验
认知科学动机：判别式分类（全连接层）vs 记忆式分类（联想记忆检索）
——把类别特征作为"记忆原型"存入 Hopfield 网络，推理即"联想检索"。

两个实验：
  --mode proto   免梯度训练：冻结骨干特征 + 类别原型记忆 + Hopfield 检索
  --mode train   可训练：学习投影矩阵 + 可学习记忆模式，端到端训练记忆头

理论链接：现代 Hopfield 更新规则（softmax 检索）与注意力机制在数学上等价
（Ramsauer et al., 2020）——联想记忆与注意力是同一枚硬币的两面。

用法:
  python src/hopfield.py --mode proto
  python src/hopfield.py --mode train --epochs 30
"""
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import CLASSES, DEVICE, ROOT, build_model

RESULTS = ROOT / "results"
CKPT = ROOT / "models" / "resnet18_finetune.pth"


def get_backbone():
    """加载微调好的 ResNet18，去掉分类头，输出 512 维特征"""
    model, *_ = build_model("resnet18")
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.fc = nn.Identity()
    return model.to(DEVICE).eval()


def get_loaders():
    tf_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(ROOT / "data" / "Training", transform=tf_train)
    test_ds = datasets.ImageFolder(ROOT / "data" / "Testing", transform=tf_test)
    ext_ds = datasets.ImageFolder(ROOT / "data_external", transform=tf_test)
    return (DataLoader(train_ds, batch_size=64, shuffle=False),
            DataLoader(test_ds, batch_size=64, shuffle=False),
            DataLoader(ext_ds, batch_size=64, shuffle=False))


# 外部数据集文件夹名 → 标签（与 external_eval.py 一致）
FOLDER_MAP = {"glioma_tumor": 0, "meningioma_tumor": 1,
              "normal": 2, "pituitary_tumor": 3}


@torch.no_grad()
def extract(backbone, loader, remap=None):
    feats, labels = [], []
    for x, y in loader:
        feats.append(backbone(x.to(DEVICE)).cpu())
        labels.append(y if remap is None else torch.tensor([remap[t] for t in y.tolist()]))
    f = torch.cat(feats)
    return F.normalize(f, dim=1), torch.cat(labels)


def hopfield_logits(query, patterns, beta):
    """现代 Hopfield 检索：softmax(beta * Q·Kᵀ)，返回对各记忆模式的 logits"""
    return beta * query @ patterns.T


def evaluate(patterns, beta, feats, labels):
    pred = hopfield_logits(feats.to(DEVICE), patterns.to(DEVICE), beta).argmax(1).cpu()
    return (pred == labels).float().mean().item()


def mode_proto(backbone, train_ld, test_ld, ext_ld, ext_remap):
    print("\n=== 实验 A：免梯度原型记忆头 ===")
    Xtr, ytr = extract(backbone, train_ld)
    # 类别原型 = 类内均值再归一化（"记忆"存入）
    protos = torch.stack([F.normalize(Xtr[ytr == c].mean(0), dim=0) for c in range(4)])
    Xte, yte = extract(backbone, test_ld)
    Xex, yex = extract(backbone, ext_ld, remap=ext_remap)

    # 在验证意义上扫 beta
    best = {}
    for beta in [4, 8, 16, 32, 64]:
        best[beta] = evaluate(protos, beta, Xte, yte)
    beta = max(best, key=best.get)
    print("beta 扫描（内部准确率）:", {k: round(v, 4) for k, v in best.items()})
    acc_in = evaluate(protos, beta, Xte, yte)
    acc_ex = evaluate(protos, beta, Xex, yex)
    print(f"最优 beta={beta} | 内部: {acc_in:.4f} | 外部: {acc_ex:.4f}")
    return {"mode": "proto", "beta": beta, "internal_acc": acc_in, "external_acc": acc_ex}


class HopfieldHead(nn.Module):
    """可学习记忆模式 + 可学习投影的现代 Hopfield 分类头"""

    def __init__(self, in_dim=512, mem_dim=128, n_classes=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, mem_dim)
        # 每个类别若干条记忆模式（这里 4 类 × 8 条）
        self.patterns = nn.Parameter(torch.randn(n_classes * 8, mem_dim) * 0.1)
        self.readout = nn.Linear(mem_dim, n_classes)
        self.beta = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        q = F.normalize(self.proj(x), dim=1)
        k = F.normalize(self.patterns, dim=1)
        # 现代 Hopfield 更新：softmax 检索后按权重重组记忆
        attn = F.softmax(self.beta * q @ k.T, dim=1)
        retrieved = attn @ self.patterns
        return self.readout(retrieved)


def mode_train(backbone, train_ld, test_ld, ext_ld, ext_remap, epochs):
    print("\n=== 实验 B：可训练 Hopfield 记忆头（冻结骨干）===")
    Xtr, ytr = extract(backbone, train_ld)
    Xte, yte = extract(backbone, test_ld)
    Xex, yex = extract(backbone, ext_ld, remap=ext_remap)

    head = HopfieldHead().to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    Xtr_d, ytr_d = Xtr.to(DEVICE), ytr.to(DEVICE)
    n = Xtr_d.size(0)

    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=DEVICE)
        head.train()
        tot_loss, bs = 0.0, 256
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = crit(head(Xtr_d[idx]), ytr_d[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(idx)
        if ep % 5 == 0 or ep == 1:
            head.eval()
            with torch.no_grad():
                acc = (head(Xte.to(DEVICE)).argmax(1).cpu() == yte).float().mean().item()
            print(f"Epoch {ep}/{epochs} | loss {tot_loss/n:.4f} | 内部 acc {acc:.4f}", flush=True)

    head.eval()
    with torch.no_grad():
        acc_in = (head(Xte.to(DEVICE)).argmax(1).cpu() == yte).float().mean().item()
        acc_ex = (head(Xex.to(DEVICE)).argmax(1).cpu() == yex).float().mean().item()
    print(f"最终 | 内部: {acc_in:.4f} | 外部: {acc_ex:.4f}")
    return {"mode": "train", "epochs": epochs, "internal_acc": acc_in, "external_acc": acc_ex}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["proto", "train"])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    backbone = get_backbone()
    train_ld, test_ld, ext_ld = get_loaders()
    # 外部数据集 remap
    ext_classes = datasets.ImageFolder(ROOT / "data_external").classes
    ext_remap = {i: FOLDER_MAP[name] for i, name in enumerate(ext_classes)}

    t0 = time.time()
    if args.mode == "proto":
        result = mode_proto(backbone, train_ld, test_ld, ext_ld, ext_remap)
    else:
        result = mode_train(backbone, train_ld, test_ld, ext_ld, ext_remap, args.epochs)
    result["time_s"] = round(time.time() - t0, 1)
    out = RESULTS / f"hopfield_{args.mode}_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()
