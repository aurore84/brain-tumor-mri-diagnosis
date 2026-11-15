# -*- coding: utf-8 -*-
"""
筛查视角的二分类指标：肿瘤 vs 无肿瘤

四分类中"无肿瘤召回率"只衡量健康图像不被误判，不能说明"不漏诊"。
筛查漏诊应统计：真实肿瘤（胶质瘤/脑膜瘤/垂体瘤）被预测为无肿瘤的比例。
本脚本把四分类预测折叠为二分类（肿瘤 vs 无肿瘤），输出敏感度/特异度。

用法: python src/screening_metrics.py
输出: results/screening_metrics.json
"""
import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from train import CLASSES, DEVICE, EVAL_TF, ROOT, build_model

RESULTS = ROOT / "results"
NOTUMOR = CLASSES.index("notumor")
FOLDER_MAP = {"glioma_tumor": 0, "meningioma_tumor": 1,
              "normal": 2, "pituitary_tumor": 3}


@torch.no_grad()
def predict_all(model, loader, remap=None):
    y_true, y_pred = [], []
    for x, y in loader:
        pred = model(x.to(DEVICE)).argmax(1).cpu()
        y_pred += pred.tolist()
        y_true += [remap[t] for t in y.tolist()] if remap else y.tolist()
    return y_true, y_pred


def binary_metrics(y_true, y_pred):
    """肿瘤=阳性。敏感度=P(预测肿瘤|真肿瘤)，特异度=P(预测无肿瘤|真无肿瘤)"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t != NOTUMOR and p != NOTUMOR)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t != NOTUMOR and p == NOTUMOR)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == NOTUMOR and p == NOTUMOR)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == NOTUMOR and p != NOTUMOR)
    return {
        "n_tumor": tp + fn, "n_notumor": tn + fp,
        "tumor_missed_as_notumor": fn,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


def main():
    model, *_ = build_model("resnet18")
    model.load_state_dict(torch.load(ROOT / "models" / "resnet18_finetune.pth",
                                     map_location=DEVICE))
    model = model.to(DEVICE).eval()

    test_ds = datasets.ImageFolder(ROOT / "data" / "Testing", transform=EVAL_TF)
    ext_ds = datasets.ImageFolder(ROOT / "data_external", transform=EVAL_TF)
    ext_remap = {i: FOLDER_MAP[n] for i, n in enumerate(ext_ds.classes)}

    out = {}
    for name, ds, remap in [("internal_test", test_ds, None),
                            ("external", ext_ds, ext_remap)]:
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        y_true, y_pred = predict_all(model, loader, remap)
        m = binary_metrics(y_true, y_pred)
        out[name] = m
        print(f"{name}: 肿瘤敏感度 {m['sensitivity']:.4f} "
              f"({m['tp']}/{m['n_tumor']})，漏诊 {m['fn']} 例；"
              f"特异度 {m['specificity']:.4f}")

    (RESULTS / "screening_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已保存: results/screening_metrics.json")


if __name__ == "__main__":
    main()
