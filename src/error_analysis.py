# -*- coding: utf-8 -*-
"""
外部验证误差分析（建议③）：逐类召回 + 典型错误案例可视化 + 数据分布对比

用法: python src/error_analysis.py
输出: results/external_error_cases.png / external_analysis.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import CLASSES, DEVICE, EVAL_TF, ROOT, build_model

RESULTS = ROOT / "results"
CN_NAMES = {"glioma": "胶质瘤", "meningioma": "脑膜瘤",
            "notumor": "无肿瘤", "pituitary": "垂体瘤"}
FOLDER_MAP = {"glioma_tumor": "glioma", "meningioma_tumor": "meningioma",
              "normal": "notumor", "pituitary_tumor": "pituitary"}


def main():
    model, *_ = build_model("resnet18")
    model.load_state_dict(torch.load(ROOT / "models" / "resnet18_finetune.pth",
                                     map_location=DEVICE))
    model = model.to(DEVICE).eval()

    ds = datasets.ImageFolder(ROOT / "data_external", transform=EVAL_TF)
    remap = {orig: CLASSES.index(FOLDER_MAP[folder])
             for folder, orig in zip(ds.classes, range(len(ds.classes)))}
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    y_true, y_pred, probs_all, paths = [], [], [], []
    import torch.nn.functional as F
    with torch.no_grad():
        bi = 0
        for x, y in loader:
            logits = model(x.to(DEVICE))
            probs = F.softmax(logits, dim=1).cpu()
            pred = probs.argmax(1)
            y_pred += pred.tolist()
            probs_all += probs.tolist()
            y_true += [remap[t] for t in y.tolist()]
            bs = x.size(0)
            paths += [ds.samples[bi + i][0] for i in range(bs)]
            bi += bs

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = (y_true == y_pred).mean()

    # 逐类召回率
    report = classification_report(y_true, y_pred, target_names=[CN_NAMES[c] for c in CLASSES],
                                   digits=4, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=[CN_NAMES[c] for c in CLASSES], digits=4))

    # 类别分布对比（内部 vs 外部）
    ext_counts = {c: int((y_true == i).sum()) for i, c in enumerate(CLASSES)}
    int_counts = {"glioma": 400, "meningioma": 400, "notumor": 400, "pituitary": 400}

    # 典型错误：每个"真实类→预测类"取置信度最高的错误样本
    cm = confusion_matrix(y_true, y_pred)
    errors = np.where(y_true != y_pred)[0]
    err_groups = {}
    for i in errors:
        key = (y_true[i], y_pred[i])
        conf = probs_all[i][y_pred[i]]
        err_groups.setdefault(key, []).append((conf, i))
    # 选出最高置信度的错误（最"自信"的错误最有分析价值）
    top_errors = []
    for key, lst in sorted(err_groups.items(),
                           key=lambda kv: -len(kv[1]))[:8]:
        lst.sort(reverse=True)
        top_errors.append((key, lst[0]))

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    for ax, ((t, p), (conf, i)) in zip(axes.flat, top_errors):
        img = plt.imread(paths[i])
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(f"真实:{CN_NAMES[CLASSES[t]]} → 预测:{CN_NAMES[CLASSES[p]]}\n"
                     f"置信度 {conf:.2f}", fontsize=10)
        ax.axis("off")
    fig.suptitle("外部验证典型错误案例（按错误频次排序，取最高置信度样本）", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "external_error_cases.png", dpi=150)
    print(f"错误案例图已保存: results/external_error_cases.png")

    out = {
        "overall_acc": float(acc),
        "per_class": {CN_NAMES[c]: {
            "recall": report[CN_NAMES[c]]["recall"],
            "precision": report[CN_NAMES[c]]["precision"],
            "support": report[CN_NAMES[c]]["support"]} for c in CLASSES},
        "confusion_pairs_top": [
            {"true": CN_NAMES[CLASSES[t]], "pred": CN_NAMES[CLASSES[p]],
             "count": len(err_groups[(t, p)])}
            for (t, p) in sorted(err_groups, key=lambda k: -len(err_groups[k]))[:6]],
        "class_distribution": {"internal_test": int_counts, "external": ext_counts},
        "confusion_matrix": cm.tolist(),
    }
    (RESULTS / "external_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"分析结果已保存: results/external_analysis.json")


if __name__ == "__main__":
    main()
