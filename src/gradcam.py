# -*- coding: utf-8 -*-
"""
Grad-CAM 注意力可视化 —— 展示模型"看"图像的哪个区域做判断
类比人类视觉注意机制，体现认知科学相关度。
用法: python src/gradcam.py --ckpt models/resnet18_finetune.pth --num 2
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from train import CLASSES, DEVICE, ROOT, build_model

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CN_NAMES = {"glioma": "胶质瘤", "meningioma": "脑膜瘤",
            "notumor": "无肿瘤", "pituitary": "垂体瘤"}


def get_target_layer(model, arch):
    if arch.startswith("resnet"):
        return model.layer4[-1]
    if arch.startswith("efficientnet"):
        return model.features[-1]
    raise ValueError(arch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num", type=int, default=2, help="每个类别展示几张")
    args = ap.parse_args()
    tag = Path(args.ckpt).stem

    model, *_ = build_model(args.arch)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    cam = GradCAM(model=model, target_layers=[get_target_layer(model, args.arch)])

    out_dir = ROOT / "results" / "gradcam"
    out_dir.mkdir(exist_ok=True)

    for cls in CLASSES:
        img_paths = sorted((ROOT / "data" / "Testing" / cls).glob("*.jpg"))[: args.num]
        for ip in img_paths:
            pil = Image.open(ip).convert("RGB").resize((224, 224))
            rgb = np.float32(pil) / 255.0
            x = tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred_idx = model(x).argmax(1).item()
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale = cam(input_tensor=x, targets=targets)[0]
            vis = show_cam_on_image(rgb, grayscale, use_rgb=True)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(rgb); axes[0].set_title("原始 MRI")
            axes[1].imshow(vis)
            axes[1].set_title(f"Grad-CAM | 预测: {CN_NAMES[CLASSES[pred_idx]]}")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"真实类别: {CN_NAMES[cls]}")
            fig.tight_layout()
            out = out_dir / f"{cls}_{ip.stem}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"已保存: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
