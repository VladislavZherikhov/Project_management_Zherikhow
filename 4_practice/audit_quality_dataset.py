"""
Audit a good/bad quality dataset built by build_good_bad_dataset.py.

Checks:
- every metadata line points to an existing image, every image has a record
- good/bad and class balance, defect balance
- data leakage: every source image has exactly one split
- saves contact sheets (audit/sheet_good.png, audit/sheet_bad.png) for visual review

Usage:
    python audit_quality_dataset.py data/demo_quality_dataset
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

SHEET_COLS, SHEET_ROWS, THUMB = 5, 3, 250


def contact_sheet(paths, out_path):
    sheet = Image.new("RGB", (SHEET_COLS * (THUMB + 10), SHEET_ROWS * (THUMB + 30)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(paths[:SHEET_COLS * SHEET_ROWS]):
        x, y = (i % SHEET_COLS) * (THUMB + 10), (i // SHEET_COLS) * (THUMB + 30)
        with Image.open(path) as im:
            sheet.paste(im.convert("RGB").resize((THUMB, THUMB)), (x, y))
        draw.text((x, y + THUMB + 2), path.stem[:42], fill="black")
    sheet.save(out_path)


def audit(folder):
    folder = Path(folder)
    records = [json.loads(line) for line in
               (folder / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    errors = []

    listed = {r["image_path"] for r in records}
    for r in records:
        if not (folder / r["image_path"]).exists():
            errors.append(f"missing image: {r['image_path']}")
    for label in ("good", "bad"):
        for path in (folder / label).glob("*.jpg"):
            if f"{label}/{path.name}" not in listed:
                errors.append(f"image without metadata: {label}/{path.name}")

    quality = Counter(r["quality_label"] for r in records)
    by_class = Counter((r["class_name"], r["quality_label"]) for r in records)
    defects = Counter(r["defect_type"] for r in records if r["quality_label"] == "bad")
    if len(set(quality.values())) > 1:
        errors.append(f"good/bad not balanced: {dict(quality)}")
    if len(set(by_class.values())) > 1:
        errors.append(f"classes not balanced: {dict(by_class)}")
    if len(set(defects.values())) > 1:
        errors.append(f"defects not balanced: {dict(defects)}")

    splits = defaultdict(set)
    for r in records:
        splits[(r["class_name"], r["source_id"])].add(r["split"])
    leaks = [src for src, s in splits.items() if len(s) > 1]
    for src in leaks:
        errors.append(f"data leakage: source {src} is in splits {sorted(splits[src])}")

    print(f"Dataset: {folder}")
    print(f"Records: {len(records)}  sources: {len(splits)}")
    print("Quality:", dict(quality))
    print("Class x quality:", {f"{c}/{q}": n for (c, q), n in sorted(by_class.items())})
    print("Defects:", dict(defects))
    print("Split (images):", dict(Counter(r["split"] for r in records)))
    print("Split (sources):", dict(Counter(next(iter(s)) for s in splits.values())))

    audit_dir = folder / "audit"
    audit_dir.mkdir(exist_ok=True)
    rng = random.Random(0)
    for label in ("good", "bad"):
        paths = sorted((folder / label).glob("*.jpg"))
        rng.shuffle(paths)
        contact_sheet(sorted(paths[:SHEET_COLS * SHEET_ROWS]), audit_dir / f"sheet_{label}.png")
    print(f"Contact sheets: {audit_dir}")

    if errors:
        print(f"FAILED: {len(errors)} problem(s)")
        for e in errors[:50]:
            print("  -", e)
        return 1
    print("OK: files match metadata, classes and defects balanced, no split leakage")
    return 0


if __name__ == "__main__":
    sys.exit(audit(sys.argv[1] if len(sys.argv) > 1 else "data/demo_quality_dataset"))
