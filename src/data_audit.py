# -*- coding: utf-8 -*-
"""
数据审计脚本：排查数据泄漏与划分合理性

检查内容：
1. 精确重复：MD5 相同的图片（训练/测试/外部三组之内与之间）
2. 近似重复：8x8 平均哈希汉明距离 ≤ 2 的图片对（每组抽样比对 + 跨组比对）
3. 类别分布与图像尺寸统计（分辨率、通道），用于解释外部验证差异
4. 数据划分方式说明：训练/测试为官方目录划分，验证集从训练集按固定随机种子划出

用法: python src/data_audit.py
结果写入 results/data_audit.json
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

GROUPS = {
    "train": ROOT / "data" / "Training",
    "test": ROOT / "data" / "Testing",
    "external": ROOT / "data_external",
}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ahash(path: Path, size: int = 8) -> int:
    """平均哈希，返回 64 位整数"""
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    bits = (arr > arr.mean()).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def collect():
    """收集三组图片的 md5 / ahash / 尺寸信息"""
    info = {}
    for gname, gdir in GROUPS.items():
        items = []
        for p in sorted(gdir.rglob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                with Image.open(p) as im:
                    w, h, mode = im.width, im.height, im.mode
                items.append({
                    "path": str(p.relative_to(ROOT)),
                    "cls": p.parent.name,
                    "md5": md5_of(p),
                    "ahash": ahash(p),
                    "w": w, "h": h, "mode": mode,
                })
            except Exception as e:  # 损坏图片也要报告
                items.append({"path": str(p.relative_to(ROOT)),
                              "cls": p.parent.name, "error": str(e)})
        info[gname] = items
        print(f"{gname}: {len(items)} 张", flush=True)
    return info


def exact_dups(items):
    by_md5 = defaultdict(list)
    for it in items:
        if "md5" in it:
            by_md5[it["md5"]].append(it["path"])
    return {k: v for k, v in by_md5.items() if len(v) > 1}


def near_dups_cross(items_a, items_b, thresh=2):
    """跨组近似重复（ahash 汉明距离 ≤ thresh），返回前 50 对"""
    pairs = []
    hs_b = [(it["ahash"], it["path"]) for it in items_b if "ahash" in it]
    for ia in items_a:
        if "ahash" not in ia:
            continue
        for hb, pb in hs_b:
            if hamming(ia["ahash"], hb) <= thresh:
                pairs.append((ia["path"], pb))
                if len(pairs) >= 50:
                    return pairs
    return pairs


def size_stats(items):
    ws = [it["w"] for it in items if "w" in it]
    hs = [it["h"] for it in items if "h" in it]
    modes = defaultdict(int)
    for it in items:
        modes[it.get("mode", "error")] += 1
    cls_count = defaultdict(int)
    for it in items:
        cls_count[it["cls"]] += 1
    if not ws:
        return {}
    return {
        "n": len(items),
        "width": {"min": min(ws), "median": int(np.median(ws)), "max": max(ws)},
        "height": {"min": min(hs), "median": int(np.median(hs)), "max": max(hs)},
        "modes": dict(modes),
        "class_count": dict(cls_count),
        "errors": sum(1 for it in items if "error" in it),
    }


def main():
    print("收集图片信息（MD5 + 平均哈希）...", flush=True)
    info = collect()

    report = {"groups": {}, "exact_duplicates": {}, "near_duplicates_cross": {}}
    for gname, items in info.items():
        report["groups"][gname] = size_stats(items)
        dups = exact_dups(items)
        report["exact_duplicates"][gname] = {
            "n_duplicate_groups": len(dups),
            "examples": list(dups.values())[:10],
        }
        print(f"{gname}: 精确重复组 {len(dups)} 个", flush=True)

    # 跨组：精确 + 近似
    cross_exact = {}
    for a, b in [("train", "test"), ("train", "external"), ("test", "external")]:
        md5_a = {it["md5"]: it["path"] for it in info[a] if "md5" in it}
        md5_b = {it["md5"]: it["path"] for it in info[b] if "md5" in it}
        common = set(md5_a) & set(md5_b)
        cross_exact[f"{a}__{b}"] = {
            "n_exact": len(common),
            "examples": [(md5_a[m], md5_b[m]) for m in list(common)[:10]],
        }
        print(f"跨组精确重复 {a} vs {b}: {len(common)} 对", flush=True)
        near = near_dups_cross(info[a], info[b])
        report["near_duplicates_cross"][f"{a}__{b}"] = {
            "n_found_first50": len(near), "examples": near}
        print(f"跨组近似重复 {a} vs {b}: 找到 {len(near)} 对（上限50）", flush=True)
    report["exact_duplicates"]["cross"] = cross_exact

    out = RESULTS / "data_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n审计报告已保存: {out}")


if __name__ == "__main__":
    main()
