# 基于迁移学习与联想记忆机制的脑肿瘤 MRI 辅助诊断系统

**人机视觉认知的对比研究** · 认知科学课程大作业

一个完整可演示的脑肿瘤 MRI 四分类系统（胶质瘤 / 脑膜瘤 / 垂体瘤 / 无肿瘤），
并围绕"机器认知与人类认知的异同"开展四组对照实验：
跨来源外部验证、三模型对比、现代 Hopfield 联想记忆分类头、小样本学习曲线（含随机初始化对照组）。

## 核心结果

| 实验 | 结果 |
|------|------|
| 两阶段迁移学习（ResNet18，CPU 训练） | 内部测试集 **93.0%**（1600 张） |
| Hopfield 联想记忆头（可训练） | **93.9%**（反超判别式分类头） |
| Hopfield 原型记忆（零训练） | 92.0% |
| 跨来源外部验证（3096 张，零调整） | 81.7%（领域偏移 -11.3 pp） |
| 小样本学习（每类 14 张，微调骨干） | 92.0%（无视觉经验对照组仅 57%） |

认知科学主线：Grad-CAM 显示模型注意区域与人类专家注视模式一致，错误呈"看对了地方、下错了结论"的人类初学者特征；层级特征可视化对应 V1→IT 腹侧通路假说；现代 Hopfield 检索与注意力机制的数学等价将"记忆"与"注意"两条认知线索统一。

## 快速开始

```bash
pip install torch torchvision gradio scikit-learn opencv-python-headless grad-cam kagglehub

# 下载数据集（约 200MB）
python -c "import kagglehub; p=kagglehub.dataset_download('masoudnickparvar/brain-tumor-mri-dataset'); print(p)"
# 将解压后的 Training/ 与 Testing/ 放入 data/
python -c "import kagglehub; p=kagglehub.dataset_download('thomasdubail/brain-tumors-256x256'); print(p)"
# 将 Data/ 下的四个类别文件夹放入 data_external/

# 训练（两阶段，CPU 约各 5 分钟）
python src/train.py --stage head --epochs 3
python src/train.py --stage finetune --epochs 3

# 启动 Web 演示
python app/app.py   # 浏览器打开 http://127.0.0.1:7860
```

Windows 用户可直接双击 `启动演示.bat` 启动演示系统。

## 复现实验

```bash
python src/evaluate.py --ckpt models/resnet18_finetune.pth     # 内部评估（混淆矩阵/报告/曲线）
python src/external_eval.py --ckpt models/resnet18_finetune.pth # 外部验证
python src/hopfield.py --mode proto     # Hopfield 原型记忆（零训练）
python src/hopfield.py --mode train     # Hopfield 可训练记忆头
python src/fewshot.py                   # 小样本学习曲线（三骨干对照，约 8 分钟）
python src/gradcam.py --ckpt models/resnet18_finetune.pth --num 2  # Grad-CAM 可视化
python src/hierarchy.py                 # 层级特征可视化（V1→IT）
python src/run_comparison.py            # 多模型对比训练（ResNet34/EfficientNet-B0）
```

## 目录结构

```
├── app/app.py          # Gradio Web 演示
├── src/                # 训练、评估、Hopfield、小样本、可视化脚本
├── docs/               # 工作日志、答辩讲稿、大作业报告、项目计划
├── ppt/                # 答辩 PPT（pptx + pptd 源文件）
├── results/            # 全部实验图表与 JSON 结果
├── data/               # 主数据集（git 忽略，需自行下载）
├── data_external/      # 外部验证集（git 忽略）
└── models/             # 模型权重（git 忽略，训练可复现）
```

## 文档

- [大作业报告](docs/大作业报告.md)：选题背景、系统设计、全部实验与分析
- [答辩讲稿](docs/答辩讲稿.md)：5 分钟逐页讲稿 + 12 问 Q&A 预案
- [工作日志与结果分析](docs/工作日志与结果分析.md)：完整实验记录与数字出处

## 数据集来源

- [Brain Tumor MRI Dataset (Nickparvar, 2021)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Brain Tumors 256x256](https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256)（外部验证）

> 本项目仅供课程学习演示，不构成医学诊断建议。
