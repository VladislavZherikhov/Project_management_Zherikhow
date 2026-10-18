"""
Metadata audit: every JSONL line must be valid JSON, point to an existing image
and contain all required fields; every image must have exactly one record.

Usage:
    python audit_metadata.py generated_passports
"""

import json
import sys
from pathlib import Path

from PIL import Image

REQUIRED_FIELDS = [
    'surname', 'name', 'patronymic', 'gender', 'birth_date', 'given_date',
    'police_department', 'oblast', 'dep_number', 'city', 'district', 'region',
    'pass_number',
]


def audit(folder):
    folder = Path(folder)
    meta_path = folder / "metadata.jsonl"
    errors, ids = [], set()

    lines = meta_path.read_text(encoding="utf-8").splitlines()
    for no, line in enumerate(lines, 1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {no}: invalid JSON ({e})")
            continue

        image_id = rec.get("image_id")
        if image_id in ids:
            errors.append(f"line {no}: duplicate image_id {image_id}")
        ids.add(image_id)

        image_path = folder / f"{image_id}.jpg"
        if not image_path.exists():
            errors.append(f"line {no}: image not found {image_path}")
        else:
            try:
                with Image.open(image_path) as im:
                    im.verify()
            except Exception as e:
                errors.append(f"line {no}: broken image {image_path} ({e})")

        missing = [f for f in REQUIRED_FIELDS if not rec.get("fields", {}).get(f)]
        if missing:
            errors.append(f"line {no}: missing fields {missing}")
        if rec.get("synthetic") is not True:
            errors.append(f"line {no}: record is not marked synthetic")

    orphans = sorted(p.name for p in folder.glob("*.jpg") if p.stem not in ids)
    for name in orphans:
        errors.append(f"image without metadata: {name}")

    print(f"Records: {len(lines)}, images: {len(list(folder.glob('*.jpg')))}")
    if errors:
        print(f"FAILED: {len(errors)} problem(s)")
        for e in errors:
            print("  -", e)
        return 1
    print("OK: every JSONL line has an existing, readable image and all fields")
    return 0


if __name__ == "__main__":
    sys.exit(audit(sys.argv[1] if len(sys.argv) > 1 else "generated_passports"))
