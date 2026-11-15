# -*- coding: utf-8 -*-
"""脑肿瘤 MRI 分类、原型—样例双通路与记忆删除演示。"""
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from case_memory import CaseMemoryIndex, backbone_from_classifier  # noqa: E402
from gradcam import get_target_layer  # noqa: E402
from train import CLASSES, DEVICE, EVAL_TF, ROOT, build_model  # noqa: E402

ARCH = "resnet18"
CASE_MEMORY_PATH = ROOT / "models" / "case_memory_index.pt"
CN_NAMES = {"glioma": "胶质瘤", "meningioma": "脑膜瘤", "notumor": "无肿瘤", "pituitary": "垂体瘤"}
WEIGHT_HINT = (
    "未找到模型权重。\n\n请先按 README 训练模型：\n"
    "python src/train.py --stage head --epochs 3\n"
    "python src/train.py --stage finetune --epochs 3"
)
MEMORY_HINT = (
    "病例记忆库尚未建立。请运行：\npython src/case_memory.py --mode all\n"
    "完成后重启演示，即可显示原型—样例双通路和删除实验。"
)


def load_model():
    for name in (f"{ARCH}_finetune.pth", f"{ARCH}_head.pth"):
        ckpt = ROOT / "models" / name
        if ckpt.exists():
            classifier, *_ = build_model(ARCH)
            classifier.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            print(f"加载模型: {ckpt}")
            return classifier.to(DEVICE).eval()
    return None


def load_case_memory():
    if not CASE_MEMORY_PATH.exists():
        return None
    try:
        memory = CaseMemoryIndex.load(CASE_MEMORY_PATH)
        print(f"加载双通路病例记忆库: {CASE_MEMORY_PATH}（{memory.size} 个病例样例）")
        return memory
    except Exception as exc:
        print(f"病例记忆库加载失败: {exc}")
        return None


model = load_model()
cam = GradCAM(model=model, target_layers=[get_target_layer(model, ARCH)]) if model else None
case_memory = load_case_memory()
memory_backbone = backbone_from_classifier(model) if model and case_memory else None
ERROR_IMG = np.zeros((224, 224, 3), dtype=np.float32)


def _fail(msg: str):
    return {msg: 1.0}, ERROR_IMG, [], msg


def _case_gallery(rows: list[dict]):
    gallery = []
    for row in rows:
        try:
            with Image.open(ROOT / row["path"]) as source:
                image = source.convert("RGB").copy()
        except Exception:
            continue
        caption = (
            f"#{row['rank']} {CN_NAMES[row['class_name']]} | "
            f"余弦相似度 {row['similarity']:.3f} | 检索权重 {row['attention_weight']:.1%}"
        )
        gallery.append((image, caption))
    return gallery


def _format_probabilities(probabilities: torch.Tensor) -> str:
    return "，".join(
        f"{CN_NAMES[name]} {float(probabilities[idx]):.1%}"
        for idx, name in enumerate(CLASSES)
    )


def _memory_report(feature: torch.Tensor) -> tuple[list, str]:
    if case_memory is None or memory_backbone is None:
        return [], MEMORY_HINT
    dual = case_memory.dual_predict(feature)
    rows = case_memory.retrieve(feature, top_k=3)
    deletion = case_memory.delete_top_and_predict(feature, delete_count=1)

    prototype_idx = int(dual["prototype_prediction"][0])
    exemplar_idx = int(dual["exemplar_prediction"][0])
    fused_idx = int(dual["fused_prediction"][0])
    center_similarities = dual["prototype_similarities"][0]
    nearest_center_idx = int(center_similarities.argmax())
    prototype_weight = float(dual["prototype_weight"])
    exemplar_weight = 1.0 - prototype_weight
    agreement = bool(dual["agreement"][0])
    after_fused_idx = deletion["after_deletion_fused_prediction"]

    report = [
        "【原型通路：类别平均中心】",
        f"最接近中心：{CN_NAMES[CLASSES[nearest_center_idx]]}（相似度 {float(center_similarities[nearest_center_idx]):.3f}）",
        f"预测：{CN_NAMES[CLASSES[prototype_idx]]}｜{_format_probabilities(dual['prototype_probabilities'][0])}",
        "【样例通路：32 个真实病例】",
        f"预测：{CN_NAMES[CLASSES[exemplar_idx]]}｜{_format_probabilities(dual['exemplar_probabilities'][0])}",
        f"两路是否一致：{'一致' if agreement else '冲突'}",
        f"【融合决策】原型 {prototype_weight:.0%} + 样例 {exemplar_weight:.0%} → {CN_NAMES[CLASSES[fused_idx]]}",
        f"融合概率：{_format_probabilities(dual['fused_probabilities'][0])}",
    ]
    if rows:
        report.append(
            f"删除最相似样例：#1 {CN_NAMES[rows[0]['class_name']]}（相似度 {rows[0]['similarity']:.3f}）"
        )
    report.extend([
        f"删除后融合预测：{CN_NAMES[CLASSES[after_fused_idx]]}｜是否翻转：{'是' if deletion['fused_flipped'] else '否'}",
        "说明：类别中心是全部训练图像的平均特征，不是合成 MRI；删除只屏蔽病例索引。",
    ])
    return _case_gallery(rows), "\n".join(report)


def predict(img):
    if model is None:
        return _fail(WEIGHT_HINT)
    if img is None:
        return _fail("请先上传一张脑部 MRI 图像")
    try:
        pil = img.convert("RGB")
    except Exception:
        return _fail("图片无法读取：文件可能损坏或格式不支持，请更换 JPG/PNG 图片")
    if min(pil.size) < 32:
        return _fail(f"图片过小（{pil.size[0]}×{pil.size[1]}），请上传分辨率更高的 MRI 图像")
    try:
        x = EVAL_TF(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
            feature = F.normalize(memory_backbone(x), dim=1).cpu() if memory_backbone else None
        pred_idx = int(probs.argmax())
        grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_idx)])[0]
        rgb = np.float32(pil.resize((224, 224))) / 255.0
        cam_img = show_cam_on_image(rgb, grayscale, use_rgb=True)
        gallery, memory_report = _memory_report(feature) if feature is not None else ([], MEMORY_HINT)
    except Exception as exc:
        return _fail(f"推理出错：{exc}")
    label_probs = {CN_NAMES[c]: float(probs[i]) for i, c in enumerate(CLASSES)}
    return label_probs, cam_img, gallery, memory_report


EXAMPLE_CANDIDATES = [
    ROOT / "data" / "Testing" / "glioma" / "Te-gl_1.jpg",
    ROOT / "data" / "Testing" / "meningioma" / "Te-me_1.jpg",
    ROOT / "data" / "Testing" / "notumor" / "Te-no_1.jpg",
    ROOT / "data" / "Testing" / "pituitary" / "Te-pi_1.jpg",
]
examples = [[str(path)] for path in EXAMPLE_CANDIDATES if path.exists()]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上传脑部 MRI 图像"),
    outputs=[
        gr.Label(num_top_classes=4, label="神经网络预测（四类概率）"),
        gr.Image(label="Grad-CAM 注意力区域"),
        gr.Gallery(label="样例通路：最相似真实病例", columns=3, rows=1, height=280),
        gr.Textbox(label="原型—样例双通路与记忆删除", lines=12),
    ],
    title="🧠 脑肿瘤 MRI 原型—样例双通路分类",
    description=(
        "类别平均原型通路判断像不像典型类别，真实病例样例通路判断像不像具体病例。"
        "界面展示两路一致或冲突、显式融合权重，以及删除最相似病例后的预测变化。"
        "（课程演示，不构成医学诊断建议。）"
    ),
    examples=examples or None,
    cache_examples=False,
)

if __name__ == "__main__":
    if model is None:
        print("⚠ " + WEIGHT_HINT)
    if case_memory is None:
        print("⚠ " + MEMORY_HINT)
    demo.launch()
