# -*- coding: utf-8 -*-
"""
脑肿瘤 MRI 分类辅助诊断系统 —— Web 演示界面
启动: python app/app.py
浏览器打开终端中显示的地址（默认 http://127.0.0.1:7860）
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from train import CLASSES, DEVICE, ROOT, build_model  # noqa: E402
from gradcam import get_target_layer  # noqa: E402

ARCH = "resnet18"
CKPT = ROOT / "models" / f"{ARCH}_finetune.pth"
if not CKPT.exists():
    CKPT = ROOT / "models" / f"{ARCH}_head.pth"

CN_NAMES = {"glioma": "胶质瘤", "meningioma": "脑膜瘤",
            "notumor": "无肿瘤", "pituitary": "垂体瘤"}

model, *_ = build_model(ARCH)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model = model.to(DEVICE).eval()
cam = GradCAM(model=model, target_layers=[get_target_layer(model, ARCH)])

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(img: Image.Image):
    pil = img.convert("RGB")
    x = tf(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    pred_idx = probs.argmax().item()

    # Grad-CAM 可视化
    grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_idx)])[0]
    rgb = np.float32(pil.resize((224, 224))) / 255.0
    cam_img = show_cam_on_image(rgb, grayscale, use_rgb=True)

    label_probs = {CN_NAMES[c]: float(probs[i]) for i, c in enumerate(CLASSES)}
    return label_probs, cam_img


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上传脑部 MRI 图像"),
    outputs=[
        gr.Label(num_top_classes=4, label="诊断预测（各类别概率）"),
        gr.Image(label="Grad-CAM 注意力可视化（模型关注的区域）"),
    ],
    title="🧠 脑肿瘤 MRI 图像分类辅助诊断系统",
    description=(
        "基于迁移学习（ResNet18）的脑肿瘤 MRI 四分类系统：胶质瘤 / 脑膜瘤 / 垂体瘤 / 无肿瘤。"
        "Grad-CAM 热力图展示模型做出判断时关注的脑区，类比人类视觉注意机制。"
        "（本项目仅供课程演示，不构成医学诊断建议）"
    ),
    examples=[
        [str(ROOT / "data" / "Testing" / "glioma" / "Te-gl_1.jpg")],
        [str(ROOT / "data" / "Testing" / "meningioma" / "Te-me_1.jpg")],
        [str(ROOT / "data" / "Testing" / "notumor" / "Te-no_1.jpg")],
        [str(ROOT / "data" / "Testing" / "pituitary" / "Te-pi_1.jpg")],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    print(f"加载模型: {CKPT}")
    demo.launch()
