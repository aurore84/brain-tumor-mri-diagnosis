# -*- coding: utf-8 -*-
"""真实病例原型记忆：检索、分类与记忆删除实验。

核心思想：
1. 用微调后的 ResNet18 提取训练图像的归一化特征；
2. 每类做 KMeans，并选择距离各聚类中心最近的真实病例作为原型；
3. 查询图像对原型病例做余弦相似度检索，以 softmax 注意权重按类别投票；
4. 删除查询最依赖的若干病例，重新预测并测量准确率、翻转率与信心变化。

用法：
  python src/case_memory.py --mode all
  python src/case_memory.py --mode build --prototypes-per-class 8
  python src/case_memory.py --mode evaluate --deletions 1,2,4
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from train import CLASSES, DEVICE, EVAL_TF, ROOT, build_model

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

MODELS = ROOT / "models"
RESULTS = ROOT / "results"
DEFAULT_INDEX = MODELS / "case_memory_index.pt"
DEFAULT_CKPT = MODELS / "resnet18_finetune.pth"
CN_NAMES = {
    "glioma": "胶质瘤",
    "meningioma": "脑膜瘤",
    "notumor": "无肿瘤",
    "pituitary": "垂体瘤",
}


def load_backbone(ckpt: Path = DEFAULT_CKPT) -> nn.Module:
    """加载分类模型并移除线性头，输出 512 维特征。"""
    if not ckpt.exists():
        raise FileNotFoundError(f"模型权重不存在: {ckpt}")
    model, *_ = build_model("resnet18")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.fc = nn.Identity()
    return model.to(DEVICE).eval()


def backbone_from_classifier(model: nn.Module) -> nn.Module:
    """从已经加载的 ResNet 分类器复制一个特征提取骨干。"""
    backbone = copy.deepcopy(model)
    backbone.fc = nn.Identity()
    return backbone.to(DEVICE).eval()


def _balanced_subset(dataset: datasets.ImageFolder, per_class: int | None, seed: int) -> Subset | datasets.ImageFolder:
    if per_class is None:
        return dataset
    rng = np.random.default_rng(seed)
    labels = np.asarray(dataset.targets)
    chosen: list[int] = []
    for class_idx in range(len(dataset.classes)):
        indices = np.flatnonzero(labels == class_idx)
        rng.shuffle(indices)
        chosen.extend(indices[: min(per_class, len(indices))].tolist())
    chosen.sort()
    return Subset(dataset, chosen)


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    dataset: datasets.ImageFolder | Subset,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """按数据集顺序提取 L2 归一化特征，并保持路径与标签对齐。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, target in loader:
        features.append(F.normalize(backbone(images.to(DEVICE)), dim=1).cpu())
        labels.append(target.cpu())

    if isinstance(dataset, Subset):
        base = dataset.dataset
        paths = [base.samples[i][0] for i in dataset.indices]
    else:
        paths = [sample[0] for sample in dataset.samples]
    return torch.cat(features), torch.cat(labels), paths


def select_real_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    paths: list[str],
    prototypes_per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[int]]:
    """每类聚类后选择离中心最近且不重复的真实病例。"""
    selected: list[int] = []
    for class_idx in range(len(CLASSES)):
        class_indices = torch.where(labels == class_idx)[0].numpy()
        if len(class_indices) < prototypes_per_class:
            raise ValueError(
                f"类别 {CLASSES[class_idx]} 只有 {len(class_indices)} 个样本，"
                f"不足以选择 {prototypes_per_class} 个病例原型"
            )
        class_features = features[class_indices].numpy()
        kmeans = MiniBatchKMeans(
            n_clusters=prototypes_per_class,
            random_state=seed,
            batch_size=min(512, len(class_indices)),
            n_init=10,
        ).fit(class_features)
        distances = kmeans.transform(class_features)
        used_local: set[int] = set()
        for cluster_idx in range(prototypes_per_class):
            order = np.argsort(distances[:, cluster_idx])
            local_idx = next(int(i) for i in order if int(i) not in used_local)
            used_local.add(local_idx)
            selected.append(int(class_indices[local_idx]))

    selected_tensor = torch.tensor(selected, dtype=torch.long)
    return (
        features[selected_tensor].clone(),
        labels[selected_tensor].clone(),
        [paths[i] for i in selected],
        selected,
    )


class CaseMemoryIndex:
    """类别平均原型与真实病例样例组成的双通路记忆库。"""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        paths: list[str],
        beta: float = 16.0,
        metadata: dict | None = None,
        class_centers: torch.Tensor | None = None,
        prototype_beta: float | None = None,
        prototype_weight: float = 0.5,
    ):
        if features.ndim != 2 or len(features) != len(labels) or len(paths) != len(labels):
            raise ValueError("病例特征、标签和路径的数量必须一致")
        self.features = F.normalize(features.float().cpu(), dim=1)
        self.labels = labels.long().cpu()
        self.paths = [str(p) for p in paths]
        self.beta = float(beta)
        self.prototype_beta = float(prototype_beta if prototype_beta is not None else beta)
        if not 0.0 <= prototype_weight <= 1.0:
            raise ValueError("prototype_weight 必须位于 [0, 1]")
        self.prototype_weight = float(prototype_weight)
        self.metadata = metadata or {}
        self.classes = list(CLASSES)
        if class_centers is None:
            class_centers = torch.stack([
                self.features[self.labels == class_idx].mean(dim=0)
                for class_idx in range(len(self.classes))
            ])
        if class_centers.shape != (len(self.classes), self.features.shape[1]):
            raise ValueError("类别中心形状与病例特征不一致")
        self.class_centers = F.normalize(class_centers.float().cpu(), dim=1)

    @property
    def size(self) -> int:
        return len(self.paths)

    def similarities(self, queries: torch.Tensor) -> torch.Tensor:
        return F.normalize(queries.float().cpu(), dim=1) @ self.features.T

    def prototype_predict(
        self, queries: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """原型通路：查询与四个全训练集类别平均中心比较。"""
        queries = F.normalize(queries.float().cpu(), dim=1)
        similarities = queries @ self.class_centers.T
        probabilities = F.softmax(self.prototype_beta * similarities, dim=1)
        return probabilities, probabilities.argmax(dim=1), similarities

    def predict_from_similarities(
        self,
        similarities: torch.Tensor,
        deleted: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """样例通路：softmax 检索权重按真实病例标签投票。"""
        scores = self.beta * similarities
        if deleted is not None:
            if deleted.shape != scores.shape:
                raise ValueError("deleted 掩码形状必须与相似度矩阵一致")
            if torch.any(deleted.all(dim=1)):
                raise ValueError("不能删除某个查询的全部记忆")
            scores = scores.masked_fill(deleted, float("-inf"))
        weights = F.softmax(scores, dim=1)
        class_probs = torch.zeros((len(scores), len(self.classes)), dtype=weights.dtype)
        class_probs.scatter_add_(1, self.labels.unsqueeze(0).expand(len(scores), -1), weights)
        pred = class_probs.argmax(dim=1)
        return class_probs, pred, weights

    def predict(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """保留兼容接口：返回样例通路结果。"""
        similarities = self.similarities(queries)
        probs, pred, weights = self.predict_from_similarities(similarities)
        return probs, pred, similarities, weights

    def dual_predict(self, queries: torch.Tensor) -> dict[str, torch.Tensor | float]:
        """分别运行原型与样例通路，并按显式权重线性融合。"""
        exemplar_probs, exemplar_pred, similarities, weights = self.predict(queries)
        prototype_probs, prototype_pred, prototype_similarities = self.prototype_predict(queries)
        fused_probs = (
            self.prototype_weight * prototype_probs
            + (1.0 - self.prototype_weight) * exemplar_probs
        )
        fused_pred = fused_probs.argmax(dim=1)
        return {
            "prototype_probabilities": prototype_probs,
            "prototype_prediction": prototype_pred,
            "prototype_similarities": prototype_similarities,
            "exemplar_probabilities": exemplar_probs,
            "exemplar_prediction": exemplar_pred,
            "exemplar_similarities": similarities,
            "exemplar_weights": weights,
            "fused_probabilities": fused_probs,
            "fused_prediction": fused_pred,
            "agreement": prototype_pred == exemplar_pred,
            "prototype_weight": self.prototype_weight,
        }

    def retrieve(self, query: torch.Tensor, top_k: int = 3) -> list[dict]:
        if query.ndim == 1:
            query = query.unsqueeze(0)
        _, _, similarities, weights = self.predict(query)
        count = min(top_k, self.size)
        top = similarities[0].topk(count)
        rows = []
        for rank, memory_idx in enumerate(top.indices.tolist(), start=1):
            class_idx = int(self.labels[memory_idx])
            rows.append({
                "rank": rank,
                "memory_index": memory_idx,
                "path": self.paths[memory_idx],
                "class_index": class_idx,
                "class_name": self.classes[class_idx],
                "similarity": float(similarities[0, memory_idx]),
                "attention_weight": float(weights[0, memory_idx]),
            })
        return rows

    def delete_top_and_predict(self, query: torch.Tensor, delete_count: int = 1) -> dict:
        if query.ndim == 1:
            query = query.unsqueeze(0)
        dual = self.dual_predict(query)
        similarities = dual["exemplar_similarities"]
        delete_count = min(max(int(delete_count), 0), self.size - 1)
        deleted = torch.zeros_like(similarities, dtype=torch.bool)
        deleted_indices: list[int] = []
        if delete_count:
            deleted_indices = similarities[0].topk(delete_count).indices.tolist()
            deleted[0, deleted_indices] = True
        new_exemplar_probs, new_exemplar_pred, _ = self.predict_from_similarities(similarities, deleted)
        prototype_probs = dual["prototype_probabilities"]
        new_fused_probs = (
            self.prototype_weight * prototype_probs
            + (1.0 - self.prototype_weight) * new_exemplar_probs
        )
        new_fused_pred = new_fused_probs.argmax(dim=1)
        return {
            "baseline_prediction": int(dual["exemplar_prediction"].item()),
            "baseline_probabilities": dual["exemplar_probabilities"][0].tolist(),
            "after_deletion_prediction": int(new_exemplar_pred.item()),
            "after_deletion_probabilities": new_exemplar_probs[0].tolist(),
            "baseline_fused_prediction": int(dual["fused_prediction"].item()),
            "baseline_fused_probabilities": dual["fused_probabilities"][0].tolist(),
            "after_deletion_fused_prediction": int(new_fused_pred.item()),
            "after_deletion_fused_probabilities": new_fused_probs[0].tolist(),
            "deleted_indices": deleted_indices,
            "flipped": bool(dual["exemplar_prediction"].item() != new_exemplar_pred.item()),
            "fused_flipped": bool(dual["fused_prediction"].item() != new_fused_pred.item()),
        }

    def save(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 2,
            "features": self.features,
            "labels": self.labels,
            "paths": self.paths,
            "class_centers": self.class_centers,
            "beta": self.beta,
            "prototype_beta": self.prototype_beta,
            "prototype_weight": self.prototype_weight,
            "classes": self.classes,
            "metadata": self.metadata,
        }, output)

    @classmethod
    def load(cls, source: Path) -> "CaseMemoryIndex":
        payload = torch.load(source, map_location="cpu")
        version = payload.get("format_version")
        if version not in {1, 2}:
            raise ValueError(f"不支持的病例记忆格式: {version}")
        if payload.get("classes") != list(CLASSES):
            raise ValueError("病例记忆库类别顺序与当前模型不一致")
        return cls(
            payload["features"],
            payload["labels"],
            payload["paths"],
            beta=float(payload["beta"]),
            metadata=payload.get("metadata") or {},
            class_centers=payload.get("class_centers"),
            prototype_beta=float(payload.get("prototype_beta", payload["beta"])),
            prototype_weight=float(payload.get("prototype_weight", 0.5)),
        )

def build_case_memory(
    backbone: nn.Module,
    output: Path,
    prototypes_per_class: int = 8,
    beta: float = 16.0,
    prototype_beta: float = 16.0,
    prototype_weight: float = 0.5,
    batch_size: int = 64,
    seed: int = 42,
    max_train_per_class: int | None = None,
) -> CaseMemoryIndex:
    train_full = datasets.ImageFolder(ROOT / "data" / "Training", transform=EVAL_TF)
    if train_full.classes != list(CLASSES):
        raise ValueError(f"训练集类别顺序异常: {train_full.classes}")
    train_ds = _balanced_subset(train_full, max_train_per_class, seed)
    features, labels, paths = extract_features(backbone, train_ds, batch_size)
    exemplar_features, exemplar_labels, exemplar_paths, selected = select_real_prototypes(
        features, labels, paths, prototypes_per_class, seed
    )
    class_centers = torch.stack([
        F.normalize(features[labels == class_idx].mean(dim=0), dim=0)
        for class_idx in range(len(CLASSES))
    ])
    relative_paths = [str(Path(p).resolve().relative_to(ROOT.resolve())) for p in exemplar_paths]
    index = CaseMemoryIndex(
        exemplar_features,
        exemplar_labels,
        relative_paths,
        beta=beta,
        class_centers=class_centers,
        prototype_beta=prototype_beta,
        prototype_weight=prototype_weight,
        metadata={
            "architecture": "resnet18",
            "checkpoint": str(DEFAULT_CKPT.relative_to(ROOT)),
            "prototypes_per_class": prototypes_per_class,
            "prototype_definition": "normalized mean feature of all training images in each class",
            "exemplar_definition": "real training image nearest to each within-class cluster center",
            "prototype_weight": prototype_weight,
            "exemplar_weight": 1.0 - prototype_weight,
            "seed": seed,
            "training_feature_count": len(features),
            "selected_training_indices": selected,
        },
    )
    index.save(output)
    return index

def _deletion_mask(similarities: torch.Tensor, count: int) -> torch.Tensor:
    count = min(max(int(count), 0), similarities.shape[1] - 1)
    mask = torch.zeros_like(similarities, dtype=torch.bool)
    if count:
        indices = similarities.topk(count, dim=1).indices
        mask.scatter_(1, indices, True)
    return mask


def evaluate_deletion(
    index: CaseMemoryIndex,
    backbone: nn.Module,
    deletion_counts: Iterable[int] = (1, 2, 4),
    batch_size: int = 64,
    max_test_per_class: int | None = None,
    seed: int = 42,
) -> dict:
    test_full = datasets.ImageFolder(ROOT / "data" / "Testing", transform=EVAL_TF)
    test_ds = _balanced_subset(test_full, max_test_per_class, seed)
    features, labels, paths = extract_features(backbone, test_ds, batch_size)
    dual = index.dual_predict(features)
    similarities = dual["exemplar_similarities"]
    base_probs = dual["exemplar_probabilities"]
    base_pred = dual["exemplar_prediction"]
    prototype_probs = dual["prototype_probabilities"]
    prototype_pred = dual["prototype_prediction"]
    fused_probs = dual["fused_probabilities"]
    fused_pred = dual["fused_prediction"]
    agreement = dual["agreement"]
    base_conf = base_probs.max(dim=1).values
    base_true_prob = base_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    base_acc = float((base_pred == labels).float().mean())
    fused_acc = float((fused_pred == labels).float().mean())
    conflicts = ~agreement

    results = {
        "index_metadata": index.metadata,
        "memory_count": index.size,
        "test_count": len(labels),
        "dual_routes": {
            "prototype_accuracy": float((prototype_pred == labels).float().mean()),
            "exemplar_accuracy": base_acc,
            "fused_accuracy": fused_acc,
            "prototype_exemplar_agreement_rate": float(agreement.float().mean()),
            "conflict_count": int(conflicts.sum()),
            "prototype_accuracy_on_conflicts": (
                float((prototype_pred[conflicts] == labels[conflicts]).float().mean()) if conflicts.any() else None
            ),
            "exemplar_accuracy_on_conflicts": (
                float((base_pred[conflicts] == labels[conflicts]).float().mean()) if conflicts.any() else None
            ),
            "fused_accuracy_on_conflicts": (
                float((fused_pred[conflicts] == labels[conflicts]).float().mean()) if conflicts.any() else None
            ),
            "prototype_weight": index.prototype_weight,
            "exemplar_weight": 1.0 - index.prototype_weight,
        },
        "baseline": {
            "accuracy": base_acc,
            "mean_confidence": float(base_conf.mean()),
            "mean_true_class_probability": float(base_true_prob.mean()),
        },
        "deletions": [],
    }
    for count in deletion_counts:
        mask = _deletion_mask(similarities, count)
        probs, pred, _ = index.predict_from_similarities(similarities, mask)
        deletion_fused_probs = (
            index.prototype_weight * prototype_probs
            + (1.0 - index.prototype_weight) * probs
        )
        deletion_fused_pred = deletion_fused_probs.argmax(dim=1)
        conf = probs.max(dim=1).values
        true_prob = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        deleted_labels = index.labels.unsqueeze(0).expand(len(labels), -1)[mask].reshape(len(labels), -1)
        same_true_fraction = float((deleted_labels == labels.unsqueeze(1)).float().mean())
        results["deletions"].append({
            "deleted_top_memories": int(count),
            "accuracy": float((pred == labels).float().mean()),
            "accuracy_change": float((pred == labels).float().mean()) - base_acc,
            "prediction_flip_rate": float((pred != base_pred).float().mean()),
            "fused_accuracy": float((deletion_fused_pred == labels).float().mean()),
            "fused_accuracy_change": float((deletion_fused_pred == labels).float().mean()) - fused_acc,
            "fused_prediction_flip_rate": float((deletion_fused_pred != fused_pred).float().mean()),
            "mean_confidence": float(conf.mean()),
            "mean_confidence_change": float((conf - base_conf).mean()),
            "mean_true_class_probability": float(true_prob.mean()),
            "mean_true_class_probability_change": float((true_prob - base_true_prob).mean()),
            "deleted_memory_same_as_true_class_fraction": same_true_fraction,
        })

    one_mask = _deletion_mask(similarities, 1)
    one_probs, one_pred, _ = index.predict_from_similarities(similarities, one_mask)
    sensitivity = base_true_prob - one_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    top_sensitive = sensitivity.topk(min(12, len(sensitivity))).indices.tolist()
    results["most_sensitive_cases"] = [
        {
            "path": str(Path(paths[i]).resolve().relative_to(ROOT.resolve())),
            "true_class": CLASSES[int(labels[i])],
            "prototype_prediction": CLASSES[int(prototype_pred[i])],
            "exemplar_prediction": CLASSES[int(base_pred[i])],
            "fused_prediction": CLASSES[int(fused_pred[i])],
            "after_delete_exemplar_prediction": CLASSES[int(one_pred[i])],
            "true_class_probability_drop": float(sensitivity[i]),
            "deleted_memory": index.paths[int(similarities[i].argmax())],
        }
        for i in top_sensitive
    ]
    return results

def save_deletion_plot(results: dict, output: Path) -> None:
    rows = results["deletions"]
    x = [0] + [r["deleted_top_memories"] for r in rows]
    exemplar_accuracy = [results["dual_routes"]["exemplar_accuracy"]] + [r["accuracy"] for r in rows]
    fused_accuracy = [results["dual_routes"]["fused_accuracy"]] + [r["fused_accuracy"] for r in rows]
    prototype_accuracy = [results["dual_routes"]["prototype_accuracy"]] * len(x)
    exemplar_flip = [0.0] + [r["prediction_flip_rate"] for r in rows]
    fused_flip = [0.0] + [r["fused_prediction_flip_rate"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.8, 4.6))
    ax1.plot(x, prototype_accuracy, "^-", color="#7B9E87", linewidth=2, label="原型通路准确率")
    ax1.plot(x, exemplar_accuracy, "o-", color="#684765", linewidth=2, label="样例通路准确率")
    ax1.plot(x, fused_accuracy, "D-", color="#C9828B", linewidth=2, label="融合通路准确率")
    ax1.set_xlabel("每个查询删除的最相似样例数")
    ax1.set_ylabel("准确率")
    ax1.set_xticks(x)
    ax1.set_ylim(0.80, 0.96)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, exemplar_flip, "o--", color="#684765", alpha=0.55, label="样例翻转率")
    ax2.plot(x, fused_flip, "D--", color="#C9828B", alpha=0.55, label="融合翻转率")
    ax2.set_ylabel("预测翻转率")
    ax2.set_ylim(0.0, max(0.16, max(exemplar_flip) * 1.15))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)
    ax1.set_title("原型—样例双通路：记忆删除与融合稳定性")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_counts(raw: str) -> list[int]:
    values = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("删除数量必须是以逗号分隔的正整数")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="真实病例原型检索与记忆删除实验")
    parser.add_argument("--mode", choices=["all", "build", "evaluate"], default="all")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--prototypes-per-class", type=int, default=8)
    parser.add_argument("--beta", type=float, default=16.0, help="样例通路 softmax 温度系数")
    parser.add_argument("--prototype-beta", type=float, default=16.0,
                        help="类别平均原型通路 softmax 温度系数")
    parser.add_argument("--prototype-weight", type=float, default=0.5,
                        help="融合预测中原型通路权重；样例通路权重为 1-该值")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deletions", type=parse_counts, default=parse_counts("1,2,4"))
    parser.add_argument("--max-train-per-class", type=int, default=None,
                        help="仅用于快速试跑；每类最多提取多少训练样本")
    parser.add_argument("--max-test-per-class", type=int, default=None,
                        help="仅用于快速试跑；每类最多评估多少测试样本")
    args = parser.parse_args()

    if args.prototypes_per_class < 2:
        parser.error("--prototypes-per-class 至少为 2")
    if not 0.0 <= args.prototype_weight <= 1.0:
        parser.error("--prototype-weight 必须位于 [0, 1]")
    ckpt = args.ckpt if args.ckpt.is_absolute() else ROOT / args.ckpt
    index_path = args.index if args.index.is_absolute() else ROOT / args.index
    backbone = load_backbone(ckpt)
    t0 = time.time()

    if args.mode in {"all", "build"}:
        print(f"构建真实病例记忆库：每类 {args.prototypes_per_class} 个原型")
        index = build_case_memory(
            backbone,
            index_path,
            prototypes_per_class=args.prototypes_per_class,
            beta=args.beta,
            prototype_beta=args.prototype_beta,
            prototype_weight=args.prototype_weight,
            batch_size=args.batch,
            seed=args.seed,
            max_train_per_class=args.max_train_per_class,
        )
        print(f"病例记忆库已保存: {index_path}（共 {index.size} 个真实病例）")
    else:
        if not index_path.exists():
            raise FileNotFoundError(f"病例记忆库不存在，请先运行 --mode build: {index_path}")
        index = CaseMemoryIndex.load(index_path)

    if args.mode in {"all", "evaluate"}:
        print(f"执行记忆删除实验：删除 {args.deletions}")
        results = evaluate_deletion(
            index,
            backbone,
            deletion_counts=args.deletions,
            batch_size=args.batch,
            max_test_per_class=args.max_test_per_class,
            seed=args.seed,
        )
        results["runtime_seconds"] = round(time.time() - t0, 1)
        result_path = RESULTS / "case_memory_deletion.json"
        plot_path = RESULTS / "case_memory_deletion.png"
        result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        save_deletion_plot(results, plot_path)
        print(json.dumps({"dual_routes": results["dual_routes"], "baseline": results["baseline"], "deletions": results["deletions"]},
                         ensure_ascii=False, indent=2))
        print(f"结果已保存: {result_path}")
        print(f"图表已保存: {plot_path}")


if __name__ == "__main__":
    main()
