"""
Build the Ultralytics classification dataset from the shared 70/15/15 split.

    yolo_dataset/
        train/{good,bad}/   1260 images
        val/{good,bad}/      270 images
        test/{good,bad}/     270 images

The split comes from 6_practice/common/split_70_15_15.jsonl, the same file
used by Task 2, so all models are compared on exactly the same test images.

Train images get a random 0/90/180/270 degree rotation (both classes, seed =
student ID). This is the static equivalent of the rotation augmentation used
for the torchvision models: in the source dataset only GOOD images are
rotated, and without it a model can learn "rotated = good". Ultralytics has
no 90-degree rotation augmentation for classification, so it is applied
here. Validation and test images are copied unchanged.

Usage:
    python build_yolo_dataset.py
"""

import csv
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import quality_common as qc

OUT_DIR = Path(__file__).resolve().parent / "yolo_dataset"
YOLO_SPLIT_NAMES = {"train": "train", "validation": "val", "test": "test"}


def build(out_dir=OUT_DIR, seed=qc.SEED):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    splits = qc.load_split()
    qc.check_split(splits)
    rng = random.Random(seed)

    rows = []
    for split, records in splits.items():
        yolo_split = YOLO_SPLIT_NAMES[split]
        for r in records:
            src = qc.DATASET_DIR / r["image_path"]
            dst = out_dir / yolo_split / r["quality_label"] / Path(r["image_path"]).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            k = rng.randrange(4) if split == "train" else 0
            if k:
                with Image.open(src) as im:
                    im.convert("RGB").rotate(90 * k, expand=True).save(dst, quality=95)
            else:
                shutil.copy2(src, dst)
            rows.append({"image_id": r["image_id"], "split": yolo_split,
                         "quality_label": r["quality_label"], "rotation_deg": 90 * k,
                         "path": dst.relative_to(out_dir).as_posix()})

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for yolo_split in YOLO_SPLIT_NAMES.values():
        counts = {c: len(list((out_dir / yolo_split / c).glob("*.jpg"))) for c in ("good", "bad")}
        print(f"{yolo_split:<6} {counts}")
    print(f"Dataset written to {out_dir}")
    return out_dir


if __name__ == "__main__":
    build()
