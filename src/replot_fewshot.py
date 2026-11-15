# -*- coding: utf-8 -*-
"""重绘小样本学习曲线（读取已保存的结果，修正标注位置）"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import ROOT

RESULTS = ROOT / "results"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BB_LABEL = {"finetuned": "判别式头 @ 微调骨干（任务视觉经验）",
            "pretrained": "判别式头 @ 预训练骨干（通用视觉经验）",
            "random": "判别式头 @ 随机初始化（无视觉经验，对照组）"}
BB_STYLE = {"finetuned": ("o-", "#684765"),
            "pretrained": ("^-", "#B89A3A"),
            "random": ("v--", "#9B9489")}

rows = json.loads((RESULTS / "fewshot_result.json").read_text(encoding="utf-8"))
x = [r["per_class"] for r in rows if r["backbone"] == "finetuned"]

fig, ax = plt.subplots(figsize=(7.5, 4.8))
for bb in ["finetuned", "pretrained", "random"]:
    style, color = BB_STYLE[bb]
    ys = [r["fc_acc"] * 100 for r in rows if r["backbone"] == bb]
    ax.plot(x, ys, style, color=color, label=BB_LABEL[bb], linewidth=2)
hp = [r["hopfield_acc"] * 100 for r in rows
      if r["backbone"] == "finetuned" and "hopfield_acc" in r]
ax.plot(x[:len(hp)], hp, "s--", color="#C9828B",
        label="记忆原型头 @ 微调骨干（Hopfield）", linewidth=2)
ax.axvspan(10, 60, color="#C9828B", alpha=0.12)
ax.annotate("少样本区间\n（每类十余人量级）", xy=(20, 62), fontsize=10,
            color="#684765", ha="center")
ax.annotate("", xy=(16, 92), xytext=(20, 66),
            arrowprops=dict(arrowstyle="->", color="#684765", lw=1.2))
ax.set_xscale("log")
ax.set_xlabel("每类训练样本数（对数轴）")
ax.set_ylabel("测试准确率 (%)")
ax.set_title("小样本学习曲线：视觉经验是样本效率的主要来源")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS / "fewshot_curve.png", dpi=150)
print("已重绘: results/fewshot_curve.png")
