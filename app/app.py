# -*- coding: utf-8 -*-
"""
脑肿瘤 MRI 分类辅助诊断系统 —— Web 演示界面
启动: python app/app.py
浏览器打开终端中显示的地址（默认 http://127.0.0.1:7860）

健壮性处理：
- 模型权重缺失时给出可操作的提示而非直接崩溃
- 无效/损坏图片、过小图片返回友好错误信息
- 示例图片不存在时自动跳过
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
CN_NAMES = {"glioma": "胶质瘤", "meningioma": "脑膜瘤",
            "notumor": "无肿瘤", "pituitary": "垂体瘤"}

WEIGHT_HINT = (
    "未找到模型权重。\n\n"
    "请先按 README 训练模型：\n"
    "  python src/train.py --stage head --epochs 3\n"
    "  python src/train.py --stage finetune --epochs 3\n"
    "或将训练好的 .pth 文件放入 models/ 目录后重启本程序。"
)


def load_model():
    """加载模型，权重缺失时返回 None 由界面友好提示"""
    for name in (f"{ARCH}_finetune.pth", f"{ARCH}_head.pth"):
        ckpt = ROOT / "models" / name
        if ckpt.exists():
            model, *_ = build_model(ARCH)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            print(f"加载模型: {ckpt}")
            return model.to(DEVICE).eval()
    return None


model = load_model()
cam = GradCAM(model=model, target_layers=[get_target_layer(model, ARCH)]) if model else None

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ERROR_IMG = np.zeros((224, 224, 3), dtype=np.float32)


def _fail(msg: str):
    """统一错误返回：概率栏显示提示，热力图栏显示空白"""
    return {msg: 1.0}, ERROR_IMG


def predict(img):
    if model is None:
        return _fail(WEIGHT_HINT)
    if img is None:
        return _fail("请先上传一张脑部 MRI 图像")
    try:
        pil = img.convert("RGB")
    except Exception:
        return _fail("图片无法读取：文件可能已损坏或格式不支持，请换一张 JPG/PNG 图片")
    if min(pil.size) < 32:
        return _fail(f"图片过小（{pil.size[0]}×{pil.size[1]}），请上传分辨率更高的 MRI 图像")

    try:
        x = tf(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

        grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_idx)])[0]
        rgb = np.float32(pil.resize((224, 224))) / 255.0
        cam_img = show_cam_on_image(rgb, grayscale, use_rgb=True)
    except Exception as e:
        return _fail(f"推理出错：{e}")

    label_probs = {CN_NAMES[c]: float(probs[i]) for i, c in enumerate(CLASSES)}
    return label_probs, cam_img


# 示例图片：只保留真实存在的，避免点击报错
EXAMPLE_CANDIDATES = [
    ROOT / "data" / "Testing" / "glioma" / "Te-gl_1.jpg",
    ROOT / "data" / "Testing" / "meningioma" / "Te-me_1.jpg",
    ROOT / "data" / "Testing" / "notumor" / "Te-no_1.jpg",
    ROOT / "data" / "Testing" / "pituitary" / "Te-pi_1.jpg",
]
examples = [[str(p)] for p in EXAMPLE_CANDIDATES if p.exists()]

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
        "Grad-CAM 热力图展示模型做出判断时关注的脑区，与认知科学中的视觉注意机制形成类比。"
        "（本项目仅供课程演示，不构成医学诊断建议）"
    ),
    examples=examples or None,
    cache_examples=False,
)

if __name__ == "__main__":
    if model is None:
        print("⚠ " + WEIGHT_HINT)
    demo.launch()
