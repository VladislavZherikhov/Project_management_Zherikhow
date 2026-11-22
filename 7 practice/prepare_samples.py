from collections import defaultdict
import json
from pathlib import Path
import shutil

from pipeline import BASE, DocumentPipeline

ROOT = BASE.parent
DATA = ROOT / "5_practice/data/quality_dataset"


def main():
    manifest = ROOT / "6_practice/common/split_70_15_15.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows = defaultdict(list)
    for record in records:
        if record["split"] == "test":
            rows[record["quality_label"], record["safe_class"], record["defect_type"]].append(record)
    pipe = DocumentPipeline()
    samples = BASE / "samples"
    benchmark = BASE / "benchmark_samples"
    samples.mkdir(exist_ok=True)
    benchmark.mkdir(exist_ok=True)
    chosen = {}
    for kind in ("inn", "passport", "snils"):
        for row in rows["good", kind, "none"]:
            path = DATA / row["image_path"]
            result = pipe.run(path)
            if result["status"] == "accepted" and result["document_type"] == kind.upper():
                chosen[f"good_{kind}.jpg"] = row
                break
        else:
            raise RuntimeError(f"No accepted example for {kind}")
    for defect in ("blur", "glare"):
        candidates = [row for (quality, _, label), group in rows.items()
                      if quality == "bad" and label == defect for row in group]
        chosen[f"bad_{defect}.jpg"] = next(
            row for row in candidates if pipe.run(DATA / row["image_path"])["status"] == "rejected")
    for filename, row in chosen.items():
        shutil.copyfile(DATA / row["image_path"], samples / filename)
    (samples / "renamed_text.jpg").write_bytes(b"This is a text file with a JPEG extension.")
    (samples / "corrupt.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00")

    for quality in ("good", "bad"):
        candidates = [row for row in records if row["split"] == "test"
                      and row["quality_label"] == quality]
        for index, row in enumerate(candidates[::len(candidates) // 10][:10], start=1):
            shutil.copyfile(DATA / row["image_path"], benchmark / f"{quality}_{index:02d}.jpg")


if __name__ == "__main__":
    main()
