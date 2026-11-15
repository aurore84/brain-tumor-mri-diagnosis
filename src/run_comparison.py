# -*- coding: utf-8 -*-
"""排队跑多模型对比实验：ResNet34 → EfficientNet-B0，各两阶段"""
import subprocess
import sys
import time

RUNS = [
    ["resnet34", "finetune"],
    ["efficientnet_b0", "head"],
    ["efficientnet_b0", "finetune"],
]

for arch, stage in RUNS:
    print(f"\n{'='*50}\n开始: {arch} / {stage}  {time.strftime('%H:%M:%S')}\n{'='*50}", flush=True)
    r = subprocess.run(
        [sys.executable, "src/train.py", "--arch", arch, "--stage", stage,
         "--epochs", "3", "--batch", "64"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    # 只保留每个 epoch 的摘要行
    for line in (r.stdout + r.stderr).splitlines():
        if any(k in line for k in ["Epoch", "测试集", "已保存", "设备"]):
            print(line, flush=True)
    if r.returncode != 0:
        print("失败，中止:", r.stderr[-500:], flush=True)
        break

print("\n全部对比实验完成", flush=True)
