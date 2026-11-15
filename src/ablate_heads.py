# -*- coding: utf-8 -*-
"""
分类头公平对比 + Hopfield 消融实验（建议②④）

设计原则：
- 同一冻结骨干（ResNet18 finetune）、同一特征缓存、同一数据划分
- 所有可训练头使用相同训练预算：Adam lr=1e-3, batch=256, epochs=30
- 模型选择（如 beta）只在验证集上进行，测试集只评估一次
- 线性头 / MLP头 / 最近原型 / Hopfield头 各跑 5 个随机种子，报告 mean±std
- MLP 头与 Hopfield 头参数量对齐（约 6.6 万），控制"额外参数"因素
- 消融：每类记忆条数 {1,2,4,8,16,32} × 3 种子
- 统计检验：线性头 vs Hopfield头 在同一测试集预测上的 McNemar 配对检验

用法: python src/ablate_heads.py
输出: results/ablation_heads.json / ablation_heads.png / hopfield_memory_sweep.png
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binomtest
from torch.utils.data import DataLoader
from torchvision import datasets

from train import CLASSES, DEVICE, EVAL_TF, ROOT, build_model
from hopfield import FOLDER_MAP

RESULTS = ROOT / "results"
CACHE = RESULTS / "features_cache.npz"
CKPT = ROOT / "models" / "resnet18_finetune.pth"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

SEEDS = [0, 1, 2, 3, 4]
EPOCHS, LR, BS = 30, 1e-3, 256


# ---------------- 特征提取（带缓存） ----------------

def get_features():
    if CACHE.exists():
        print(f"加载特征缓存: {CACHE}")
        z = np.load(CACHE)
        return {k: torch.from_numpy(z[k]) for k in z.files}

    print("提取骨干特征（首次运行，约几分钟）...", flush=True)
    model, *_ = build_model("resnet18")
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.fc = nn.Identity()
    model = model.to(DEVICE).eval()

    def make_loaders():
        train_full = datasets.ImageFolder(ROOT / "data" / "Training", transform=EVAL_TF)
        test_ds = datasets.ImageFolder(ROOT / "data" / "Testing", transform=EVAL_TF)
        ext_ds = datasets.ImageFolder(ROOT / "data_external", transform=EVAL_TF)
        n_val = int(0.1 * len(train_full))
        gen = torch.Generator().manual_seed(42)
        _, val_idx = torch.utils.data.random_split(
            range(len(train_full)), [len(train_full) - n_val, n_val], generator=gen)
        val_set = set(val_idx.indices if hasattr(val_idx, "indices") else list(val_idx))
        tr_idx = [i for i in range(len(train_full)) if i not in val_set]
        va_idx = sorted(val_set)
        train_ds = torch.utils.data.Subset(train_full, tr_idx)
        val_ds = torch.utils.data.Subset(train_full, va_idx)
        mk = lambda ds, sh=False: DataLoader(ds, batch_size=64, shuffle=sh)
        return mk(train_ds), mk(val_ds), mk(test_ds), mk(ext_ds), ext_ds

    @torch.no_grad()
    def extract(loader, remap=None):
        feats, labels = [], []
        for x, y in loader:
            feats.append(model(x.to(DEVICE)).cpu())
            if remap is None:
                labels.append(y)
            else:
                labels.append(torch.tensor([remap[t] for t in y.tolist()]))
        f = torch.cat(feats)
        return F.normalize(f, dim=1).numpy(), torch.cat(labels).numpy()

    tr_ld, va_ld, te_ld, ex_ld, ext_ds = make_loaders()
    ext_remap = {i: FOLDER_MAP[n] for i, n in enumerate(ext_ds.classes)}
    out = {}
    out["Xtr"], out["ytr"] = extract(tr_ld)
    print(f"  train {out['Xtr'].shape}", flush=True)
    out["Xva"], out["yva"] = extract(va_ld)
    out["Xte"], out["yte"] = extract(te_ld)
    out["Xex"], out["yex"] = extract(ex_ld, remap=ext_remap)
    np.savez(CACHE, **out)
    print(f"特征已缓存: {CACHE}")
    return {k: torch.from_numpy(v) for k, v in out.items()}


# ---------------- 分类头定义 ----------------

class LinearHead(nn.Module):
    def __init__(self, d=512, c=4):
        super().__init__()
        self.fc = nn.Linear(d, c)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    """与 HopfieldHead 参数量对齐（约 6.6 万）"""
    def __init__(self, d=512, h=128, c=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, c))

    def forward(self, x):
        return self.net(x)


class HopfieldHead(nn.Module):
    """现代 Hopfield 记忆头：可学习投影 + 每类 n_mem 条可学习记忆"""
    def __init__(self, d=512, h=128, c=4, n_mem=8):
        super().__init__()
        self.proj = nn.Linear(d, h)
        self.patterns = nn.Parameter(torch.randn(c * n_mem, h) * 0.1)
        self.readout = nn.Linear(h, c)
        self.beta = nn.Parameter(torch.tensor(10.0))
        self.n_mem, self.c = n_mem, c

    def forward(self, x):
        q = F.normalize(self.proj(x), dim=1)
        k = F.normalize(self.patterns, dim=1)
        attn = F.softmax(self.beta * q @ k.T, dim=1)
        retrieved = attn @ self.patterns
        return self.readout(retrieved)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------- 训练与评估 ----------------

def train_head(head, Xtr, ytr, seed):
    torch.manual_seed(seed)
    head = head.to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    X, y = Xtr.to(DEVICE), ytr.to(DEVICE)
    n = X.size(0)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        head.train()
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            loss = crit(head(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()


@torch.no_grad()
def predict(head, X):
    return head(X.to(DEVICE)).argmax(1).cpu()


def acc_of(pred, y):
    return (pred == y).float().mean().item()


def run_trainable(name, ctor, feat, seeds=SEEDS):
    """训练型头：多种子，返回 test/external 准确率列表与逐种子预测"""
    accs_te, accs_ex, preds = [], [], []
    for s in seeds:
        head = train_head(ctor(), feat["Xtr"], feat["ytr"], s)
        with torch.no_grad():
            p_te = predict(head, feat["Xte"])
            p_ex = predict(head, feat["Xex"])
        accs_te.append(acc_of(p_te, feat["yte"]))
        accs_ex.append(acc_of(p_ex, feat["yex"]))
        preds.append(p_te.numpy())
        print(f"  {name} seed={s} | test {accs_te[-1]:.4f} | external {accs_ex[-1]:.4f}",
              flush=True)
    return {"test": accs_te, "external": accs_ex, "preds": preds,
            "n_params": n_params(ctor())}


def run_proto(feat):
    """最近原型记忆：beta 只在验证集上选，再在测试/外部集评估"""
    Xtr, ytr = feat["Xtr"], feat["ytr"]
    protos = torch.stack([F.normalize(Xtr[ytr == c].mean(0), dim=0) for c in range(4)])
    betas = [1, 2, 4, 8, 16, 32, 64]
    va_accs = {}
    for b in betas:
        pred = (b * feat["Xva"] @ protos.T).argmax(1)
        va_accs[b] = acc_of(pred, feat["yva"])
    best_b = max(va_accs, key=va_accs.get)
    print(f"  proto 验证集 beta 扫描: { {k: round(v,4) for k,v in va_accs.items()} } -> 选 {best_b}")
    p_te = (best_b * feat["Xte"] @ protos.T).argmax(1)
    p_ex = (best_b * feat["Xex"] @ protos.T).argmax(1)
    return {"beta": best_b, "val_scan": va_accs,
            "test": [acc_of(p_te, feat["yte"])],
            "external": [acc_of(p_ex, feat["yex"])],
            "preds": [p_te.numpy()], "n_params": 0}


def mcnemar(preds_a, preds_b, y_true):
    """McNemar 精确检验：a/b 为两模型在同一测试集上的预测"""
    y = np.asarray(y_true)
    a_ok = np.asarray(preds_a) == y
    b_ok = np.asarray(preds_b) == y
    b_only = int((~a_ok & b_ok).sum())   # b 对 a 错
    a_only = int((a_ok & ~b_ok).sum())   # a 对 b 错
    n = b_only + a_only
    p = binomtest(min(b_only, a_only), n, 0.5).pvalue if n > 0 else 1.0
    return {"a_only_correct": a_only, "b_only_correct": b_only, "p_value": p}


# ---------------- 主流程 ----------------

def main():
    t0 = time.time()
    feat = get_features()
    results = {"config": {"seeds": SEEDS, "epochs": EPOCHS, "lr": LR, "batch": BS,
                          "backbone": "resnet18_finetune (冻结)", "device": str(DEVICE)}}

    print("\n=== 1. 线性头（参数量最小基线） ===")
    results["linear"] = run_trainable("linear", lambda: LinearHead(), feat)

    print("\n=== 2. MLP 头（参数量对齐 Hopfield） ===")
    results["mlp"] = run_trainable("mlp", lambda: MLPHead(), feat)

    print("\n=== 3. 最近原型记忆（零训练） ===")
    results["proto"] = run_proto(feat)

    print("\n=== 4. Hopfield 记忆头（每类 8 条记忆） ===")
    results["hopfield"] = run_trainable("hopfield", lambda: HopfieldHead(n_mem=8), feat)

    print("\n=== 5. 消融：每类记忆条数 sweep ===")
    sweep = {}
    for n_mem in [1, 2, 4, 8, 16, 32]:
        r = run_trainable(f"hopfield_mem{n_mem}",
                          lambda n=n_mem: HopfieldHead(n_mem=n), feat, seeds=[0, 1, 2])
        sweep[n_mem] = {"test_mean": float(np.mean(r["test"])),
                        "test_std": float(np.std(r["test"])),
                        "external_mean": float(np.mean(r["external"])),
                        "n_params": r["n_params"]}
        print(f"  mem={n_mem}: test {sweep[n_mem]['test_mean']:.4f}"
              f"±{sweep[n_mem]['test_std']:.4f}", flush=True)
    results["memory_sweep"] = sweep

    print("\n=== 6. McNemar 配对检验（线性头 vs Hopfield头，逐种子） ===")
    mc = []
    for i, s in enumerate(SEEDS):
        r = mcnemar(results["linear"]["preds"][i],
                    results["hopfield"]["preds"][i], feat["yte"].numpy())
        r["seed"] = s
        mc.append(r)
        print(f"  seed={s}: {r}", flush=True)
    results["mcnemar_linear_vs_hopfield"] = mc

    # 汇总
    summary = {}
    for name in ["linear", "mlp", "proto", "hopfield"]:
        r = results[name]
        summary[name] = {
            "test_mean": float(np.mean(r["test"])), "test_std": float(np.std(r["test"])),
            "external_mean": float(np.mean(r["external"])),
            "external_std": float(np.std(r["external"])),
            "n_params": r["n_params"],
        }
    results["summary"] = summary
    results["time_s"] = round(time.time() - t0, 1)

    out = RESULTS / "ablation_heads.json"
    # numpy 数组不可直接 json 序列化，preds 单独存
    preds = {k: results[k].pop("preds") for k in ["linear", "mlp", "proto", "hopfield"]}
    np.savez(RESULTS / "ablation_preds.npz", **{f"{k}_{i}": v
             for k, lst in preds.items() for i, v in enumerate(lst)})
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")
    print(f"\n结果已保存: {out}")

    # ---- 图1：四种头对比（误差棒） ----
    names = ["linear", "mlp", "proto", "hopfield"]
    labels = ["线性头", "MLP头\n(参数对齐)", "原型记忆\n(零训练)", "Hopfield\n记忆头"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(names))
    w = 0.38
    te = [summary[n]["test_mean"] for n in names]
    te_e = [summary[n]["test_std"] for n in names]
    ex = [summary[n]["external_mean"] for n in names]
    ex_e = [summary[n]["external_std"] for n in names]
    ax.bar(x - w/2, te, w, yerr=te_e, capsize=4, label="内部测试集")
    ax.bar(x + w/2, ex, w, yerr=ex_e, capsize=4, label="外部验证集")
    for i, v in enumerate(te):
        ax.text(i - w/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    for i, v in enumerate(ex):
        ax.text(i + w/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("准确率")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("四种分类头公平对比（同一冻结骨干，5 种子 mean±std）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "ablation_heads.png", dpi=150)
    print("图已保存: results/ablation_heads.png")

    # ---- 图2：记忆条数消融 ----
    mems = sorted(sweep.keys(), key=int)
    fig, ax = plt.subplots(figsize=(6, 4))
    means = [sweep[m]["test_mean"] for m in mems]
    stds = [sweep[m]["test_std"] for m in mems]
    ex_means = [sweep[m]["external_mean"] for m in mems]
    ax.errorbar(mems, means, yerr=stds, marker="o", capsize=4, label="内部测试集")
    ax.plot(mems, ex_means, marker="s", label="外部验证集")
    ax.set_xscale("log", base=2)
    ax.set_xticks(mems, [str(m) for m in mems])
    ax.set_xlabel("每类记忆条数"); ax.set_ylabel("准确率")
    ax.set_title("Hopfield 记忆容量消融（3 种子 mean±std）")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "hopfield_memory_sweep.png", dpi=150)
    print("图已保存: results/hopfield_memory_sweep.png")


if __name__ == "__main__":
    main()
