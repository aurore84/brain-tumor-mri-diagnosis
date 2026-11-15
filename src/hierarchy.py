# -*- coding: utf-8 -*-
"""
层级特征可视化实验
认知科学动机：CNN 的层级特征提取对应视觉皮层 V1 → IT 的层级表征假说
——低层响应边缘/纹理（类比 V1 简单细胞），高层响应病灶语义（类比 IT 区）。
做法：对同一层所有通道的激活做均值聚合，展示四个层级的"看到的图像"。
用法: python src/hierarchy.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from train import CLASSES, DEVICE, ROOT, build_model
from hopfield import get_backbone  # noqa: F401（备用）

CKPT = ROOT / "models" / "resnet18_finetune.pth"
RESULTS = ROOT / "results"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

LAYER_NAMES = ["layer1", "layer2", "layer3", "layer4"]
LAYER_CN = {"layer1": "低层（边缘/纹理）\n类比 V1 简单细胞",
            "layer2": "中低层（局部形状）",
            "layer3": "中高层（部件结构）",
            "layer4": "高层（病灶语义）\n类比腹侧通路 IT 区"}


def main():
    model, *_ = build_model("resnet18")
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    activations = {}
    hooks = []
    for name in LAYER_NAMES:
        layer = getattr(model, name)
        hooks.append(layer.register_forward_hook(
            lambda m, i, o, n=name: activations.__setitem__(n, o.detach())))

    samples = [
        ("meningioma", "Te-me_1.jpg", "脑膜瘤"),
        ("glioma", "Te-gl_10.jpg", "胶质瘤"),
    ]
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    fig, axes = plt.subplots(len(samples), 5, figsize=(13, 5.6))
    for row, (cls, fn, cn) in enumerate(samples):
        img = Image.open(ROOT / "data" / "Testing" / cls / fn).convert("RGB")
        x = tf(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(x).argmax(1).item()
        axes[row][0].imshow(img.resize((224, 224)))
        axes[row][0].set_title(f"输入：{cn}\n预测：{CLASSES[pred]}", fontsize=11)
        for col, name in enumerate(LAYER_NAMES, start=1):
            act = activations[name][0]            # [C, H, W]
            amap = act.mean(0, keepdim=True)       # 通道均值聚合
            amap = F.interpolate(amap.unsqueeze(0), size=(224, 224),
                                 mode="bilinear", align_corners=False)[0, 0]
            amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
            axes[row][col].imshow(amap.cpu(), cmap="magma")
            if row == 0:
                axes[row][col].set_title(LAYER_CN[name], fontsize=11)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("CNN 层级特征：从边缘纹理到病灶语义 —— 对应视觉皮层 V1 → IT 的层级加工", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS / "hierarchy_features.png", dpi=150)
    for h in hooks:
        h.remove()
    print("已保存: results/hierarchy_features.png")


if __name__ == "__main__":
    main()
