"""只读检查跨目录的相同 RGB 像素；不代表患者独立或近似图像去重。"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps
from torchvision.datasets import ImageFolder


def audit(roots):
    groups = defaultdict(list)
    counts = {}
    for root in roots:
        root = Path(root).resolve()
        dataset = ImageFolder(root)
        counts[str(root)] = len(dataset)
        for filename, target in dataset.samples:
            with Image.open(filename) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                digest = hashlib.sha256(
                    str(image.size).encode() + b":" + image.tobytes()
                ).hexdigest()
            groups[digest].append({
                "root": str(root), "path": str(Path(filename).relative_to(root)),
                "class": dataset.classes[target],
            })
    duplicates = [items for items in groups.values() if len(items) > 1]
    cross_root = [items for items in duplicates if len({i["root"] for i in items}) > 1]
    return {"images": counts, "duplicate_groups": duplicates,
            "cross_root_groups": cross_root,
            "limitation": "Exact decoded RGB pixels only; no patient or near-duplicate check."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    if len({root.resolve() for root in args.roots}) != len(args.roots):
        parser.error("请提供不同的数据目录")
    report = audit(args.roots)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["cross_root_groups"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
