# -*- coding: utf-8 -*-
"""
复核跨组"近似重复"：8x8 平均哈希命中的图片对是否真重复

对 data_audit.json 中跨组命中的图片对，计算：
- 16x16 平均哈希汉明距离（更细粒度）
- 64x64 灰度图的归一化均方误差 (MSE)
真重复应同时满足：16x16 距离很小 且 MSE 接近 0。

用法: python src/verify_near_dups.py
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def ahash16(path):
    img = Image.open(path).convert("L").resize((16, 16), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    bits = (arr > arr.mean()).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def mse64(pa, pb):
    a = np.asarray(Image.open(pa).convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float32) / 255.0
    b = np.asarray(Image.open(pb).convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float32) / 255.0
    return float(((a - b) ** 2).mean())


def hamming(a, b):
    return bin(a ^ b).count("1")


def main():
    audit = json.loads((ROOT / "results" / "data_audit.json").read_text(encoding="utf-8"))
    verified = {}
    for pair_key, data in audit["near_duplicates_cross"].items():
        rows = []
        for pa, pb in data["examples"]:
            fa, fb = ROOT / pa, ROOT / pb
            d = hamming(ahash16(fa), ahash16(fb))
            m = mse64(fa, fb)
            rows.append({"a": pa, "b": pb, "hash16_dist": d, "mse": round(m, 4)})
        # 真重复：16x16 距离 ≤ 10 且 MSE ≤ 0.005
        real = [r for r in rows if r["hash16_dist"] <= 10 and r["mse"] <= 0.005]
        verified[pair_key] = {
            "checked": len(rows),
            "likely_real_duplicates": len(real),
            "real_examples": real[:5],
            "hash16_dist_range": [min(r["hash16_dist"] for r in rows),
                                  max(r["hash16_dist"] for r in rows)] if rows else None,
            "mse_range": [min(r["mse"] for r in rows),
                          max(r["mse"] for r in rows)] if rows else None,
        }
        print(f"{pair_key}: 复核 {len(rows)} 对, 疑似真重复 {len(real)} 对")
        for r in rows[:5]:
            print(f"   {Path(r['a']).name} vs {Path(r['b']).name} | "
                  f"hash16={r['hash16_dist']} mse={r['mse']}")
    out = ROOT / "results" / "near_dup_verify.json"
    out.write_text(json.dumps(verified, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"复核结果已保存: {out}")


if __name__ == "__main__":
    main()
