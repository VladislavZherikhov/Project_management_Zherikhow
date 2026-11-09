"""
Run the document photo quality gate (YOLOv8n-cls) on your own images.

Usage:
    python predict.py --source path/to/images
    python predict.py --source photo.jpg
    python predict.py --source path/to/dataset --threshold 0.5 --out predictions.csv

--source can be a single image, a folder of images, or a folder that has
good/ and bad/ sub-folders (any depth, e.g. test/good, test/bad). In the last
case the script also prints accuracy, precision, recall, F1 and the
confusion matrix, with "bad" as the positive class.

Output CSV columns: path, prob_bad, prediction, true_label (if known).
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_WEIGHTS = Path(__file__).with_name("yolov8n_quality_gate.pt")


def collect_images(source):
    source = Path(source)
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def true_label(path):
    """good/bad from the nearest parent folder named good or bad, else None."""
    for parent in path.parents:
        if parent.name.lower() in ("good", "bad"):
            return parent.name.lower()
    return None


def report(rows):
    labelled = [r for r in rows if r["true_label"]]
    if not labelled:
        return
    tp = sum(r["true_label"] == "bad" and r["prediction"] == "bad" for r in labelled)
    fn = sum(r["true_label"] == "bad" and r["prediction"] == "good" for r in labelled)
    fp = sum(r["true_label"] == "good" and r["prediction"] == "bad" for r in labelled)
    tn = sum(r["true_label"] == "good" and r["prediction"] == "good" for r in labelled)
    accuracy = (tp + tn) / len(labelled)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"\nLabelled images: {len(labelled)} (good={tn + fp}, bad={tp + fn})")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Precision (bad): {precision:.4f}")
    print(f"Recall (bad):    {recall:.4f}")
    print(f"F1 (bad):        {f1:.4f}")
    print("Confusion matrix (rows = true, columns = predicted):")
    print("              pred good  pred bad")
    print(f"  true good   {tn:>9}  {fp:>8}")
    print(f"  true bad    {fn:>9}  {tp:>8}")


def main():
    parser = argparse.ArgumentParser(description="Document photo quality gate (good / bad).")
    parser.add_argument("--source", required=True, help="image, folder, or folder with good/ and bad/")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--threshold", type=float, default=0.5, help="P(bad) >= threshold -> bad")
    parser.add_argument("--device", default="cpu", help="cpu or 0 for the first GPU")
    parser.add_argument("--out", default="predictions.csv")
    args = parser.parse_args()

    model = YOLO(args.weights)
    bad_idx = [k for k, v in model.names.items() if v == "bad"][0]
    paths = collect_images(args.source)
    if not paths:
        raise SystemExit(f"No images found in {args.source}")
    print(f"Images: {len(paths)} | weights: {args.weights} | threshold: {args.threshold}")

    rows = []
    for i in range(0, len(paths), 32):
        batch = paths[i:i + 32]
        for path, result in zip(batch, model.predict([str(p) for p in batch], imgsz=224,
                                                     device=args.device, verbose=False)):
            prob_bad = float(result.probs.data[bad_idx])
            rows.append({"path": str(path), "prob_bad": round(prob_bad, 4),
                         "prediction": "bad" if prob_bad >= args.threshold else "good",
                         "true_label": true_label(path) or ""})

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Predictions saved to {args.out}")
    report(rows)


if __name__ == "__main__":
    main()
