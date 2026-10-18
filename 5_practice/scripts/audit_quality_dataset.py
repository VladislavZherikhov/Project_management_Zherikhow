import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


REQUIRED_FIELDS = [
    "image_id",
    "quality_label",
    "image_path",
    "split",
    "source_id",
    "class_name",
]

VALID_CLASSES = {
    "inn",
    "snils",
    "passport",
}

VALID_QUALITY_LABELS = {
    "good",
    "bad",
}

VALID_SPLITS = {
    "train",
    "validation",
    "test",
}


def audit_dataset(
    dataset_dir,
    target_size=(640, 640),
    check_pixels=True,
):
    """
    Audit a quality dataset.

    Expected structure:
        dataset_dir/
            good/
            bad/
            metadata.jsonl
            summary.json
    """
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "metadata.jsonl"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.jsonl not found in {dataset_dir}. "
            f"Please generate the dataset first."
        )

    records = []
    errors = []

    image_id_counts = Counter()
    image_path_counts = Counter()
    source_split_map = defaultdict(set)

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line_number, line in enumerate(metadata_file, start=1):
            line = line.strip()

            if not line:
                continue

            # ----------------------------------------
            # 1. Parse JSON
            # ----------------------------------------
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append({
                    "line": line_number,
                    "error": f"Invalid JSON: {e}",
                })
                continue

            # ----------------------------------------
            # 2. Check required fields
            # ----------------------------------------
            missing_fields = [
                field
                for field in REQUIRED_FIELDS
                if field not in record
            ]

            if missing_fields:
                errors.append({
                    "line": line_number,
                    "image_id": record.get("image_id", "unknown"),
                    "error": f"Missing fields: {missing_fields}",
                })
                continue

            image_id = record["image_id"]
            image_path = record["image_path"]
            quality_label = record["quality_label"]
            split = record["split"]
            source_id = record["source_id"]

            image_id_counts[image_id] += 1
            image_path_counts[image_path] += 1

            # ----------------------------------------
            # 3. Validate labels
            # ----------------------------------------
            if quality_label not in VALID_QUALITY_LABELS:
                errors.append({
                    "line": line_number,
                    "image_id": image_id,
                    "error": f"Invalid quality_label: {quality_label}",
                })
                continue

            if record["class_name"] not in VALID_CLASSES:
                errors.append({
                    "line": line_number,
                    "image_id": image_id,
                    "error": f"Invalid class_name: {record['class_name']}",
                })
                continue

            # the folder must match the label: good/... or bad/...
            if Path(image_path).parts[0] != quality_label:
                errors.append({
                    "line": line_number,
                    "image_id": image_id,
                    "error": (
                        f"Label/folder mismatch: quality_label={quality_label}, "
                        f"image_path={image_path}"
                    ),
                })
                continue

            if split not in VALID_SPLITS:
                errors.append({
                    "line": line_number,
                    "image_id": image_id,
                    "error": f"Invalid split: {split}",
                })
                continue

            # leakage is counted only for valid split values
            source_split_map[source_id].add(split)

            # ----------------------------------------
            # 4. Validate image file
            # ----------------------------------------
            full_image_path = dataset_dir / image_path

            if not full_image_path.exists():
                errors.append({
                    "line": line_number,
                    "image_id": image_id,
                    "error": f"Image file not found: {full_image_path}",
                })
                continue

            if check_pixels:
                try:
                    # verify() detects truncated / broken files,
                    # convert() forces a full decode of the pixels
                    with Image.open(full_image_path) as img:
                        img.verify()
                    with Image.open(full_image_path) as img:
                        img = img.convert("RGB")

                        if img.size != tuple(target_size):
                            errors.append({
                                "line": line_number,
                                "image_id": image_id,
                                "error": (
                                    f"Wrong image size: {img.size}. "
                                    f"Expected {target_size}."
                                ),
                            })
                            continue

                except Exception as e:
                    errors.append({
                        "line": line_number,
                        "image_id": image_id,
                        "error": f"Cannot open image: {e}",
                    })
                    continue

            records.append(record)

    # --------------------------------------------------------
    # Create summary
    # --------------------------------------------------------

    duplicate_image_ids = [
        image_id
        for image_id, count in image_id_counts.items()
        if count > 1
    ]

    duplicate_image_paths = [
        image_path
        for image_path, count in image_path_counts.items()
        if count > 1
    ]

    # image files on disk that have no metadata record
    listed_paths = set(image_path_counts)
    orphan_files = sorted(
        f"{folder}/{path.name}"
        for folder in sorted(VALID_QUALITY_LABELS)
        if (dataset_dir / folder).exists()
        for path in (dataset_dir / folder).iterdir()
        if path.is_file() and f"{folder}/{path.name}" not in listed_paths
    )

    leakage_sources = {
        source_id: sorted(splits)
        for source_id, splits in source_split_map.items()
        if len(splits) > 1
    }

    summary = {
        "dataset_dir": str(dataset_dir),
        "total_valid_records": len(records),
        "total_errors": len(errors),
        "quality_label": dict(Counter(
            record["quality_label"]
            for record in records
        )),
        "class_name": dict(Counter(
            record.get("class_name", record.get("doc_type", "unknown"))
            for record in records
        )),
        "split": dict(Counter(
            record["split"]
            for record in records
        )),
        "defect_type_bad": dict(Counter(
            record.get("defect_type", "unknown")
            for record in records
            if record["quality_label"] == "bad"
        )),
        "good_augmentation": dict(Counter(
            # slight rotations are grouped, the exact angle stays in metadata
            re.sub(r"^slight_rotation_.*", "slight_rotation",
                   record.get("good_augmentation", "none"))
            for record in records
            if record["quality_label"] == "good"
        )),
        "duplicate_image_ids": duplicate_image_ids,
        "duplicate_image_paths": duplicate_image_paths,
        "source_id_leakage": leakage_sources,
        "orphan_files": orphan_files,
        "warnings": [],
    }

    # --------------------------------------------------------
    # Add warnings
    # --------------------------------------------------------

    quality_counts = summary["quality_label"]
    good_count = quality_counts.get("good", 0)
    bad_count = quality_counts.get("bad", 0)

    if good_count == 0 or bad_count == 0:
        summary["warnings"].append(
            "Dataset has only one quality class."
        )
    else:
        imbalance_ratio = abs(good_count - bad_count) / max(good_count, bad_count)

        if imbalance_ratio > 0.1:
            summary["warnings"].append(
                f"Good/bad imbalance detected: "
                f"good={good_count}, bad={bad_count}."
            )

    class_counts = summary["class_name"]

    if len(class_counts) > 1:
        max_class_count = max(class_counts.values())
        min_class_count = min(class_counts.values())

        if min_class_count == 0:
            summary["warnings"].append(
                "One or more document classes have zero images."
            )
        elif max_class_count / min_class_count > 1.1:
            summary["warnings"].append(
                f"Document class imbalance detected: {class_counts}."
            )

    for split_name in VALID_SPLITS:
        if summary["split"].get(split_name, 0) == 0:
            summary["warnings"].append(
                f"Split '{split_name}' is empty."
            )

    if duplicate_image_ids:
        summary["warnings"].append(
            "Duplicate image_id values found."
        )

    if duplicate_image_paths:
        summary["warnings"].append(
            "Duplicate image_path values found."
        )

    if orphan_files:
        summary["warnings"].append(
            f"{len(orphan_files)} image file(s) have no metadata record."
        )

    if leakage_sources:
        summary["warnings"].append(
            "Data leakage detected: same source_id appears in multiple splits."
        )

    return summary, errors


def main():
    parser = argparse.ArgumentParser(
        description="Audit a good/bad quality dataset."
    )

    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("data/demo_quality_dataset"),
        help="Folder containing good/, bad/, and metadata.jsonl",
    )

    parser.add_argument(
        "--output_report",
        type=Path,
        default=Path("reports/dataset_audit_report.json"),
        help="Where to save the audit report",
    )

    parser.add_argument(
        "--no_pixel_check",
        action="store_true",
        help="Skip opening images and checking pixel size",
    )

    args = parser.parse_args()

    summary, errors = audit_dataset(
        dataset_dir=args.dataset_dir,
        check_pixels=not args.no_pixel_check,
    )

    report = {
        "summary": summary,
        "errors": errors,
    }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)

    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Dataset audit finished.")
    print(f"Report saved to: {args.output_report}")
    print()
    print("Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()