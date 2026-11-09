"""Test our trained YOLOv8n classification model on images from another dataset.

Examples:
    python predict.py --source photo.jpg
    python predict.py --source path/to/test
    python predict.py --source path/to/test --out peer_predictions.csv

If images are inside good/ or bad/ folders, print accuracy, precision,
recall, F1 for each class and a confusion matrix. No training is performed.
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_WEIGHTS = Path(__file__).resolve().with_name("yolov8n_quality_gate.pt")


def find_images(source):
    source = Path(source).expanduser()
    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise SystemExit(f"Unsupported image extension: {source.suffix}")
        return [source]
    return sorted((p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: str(p).lower())


def ground_truth(path):
    """Use the nearest ancestor folder named good or bad, if present."""
    return next((p.name.lower() for p in path.parents if p.name.lower() in ("good", "bad")), "")


def per_class_stats(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def print_metrics(rows):
    labeled = [r for r in rows if r["true_label"] in ("bad", "good")]
    if not labeled:
        print("No good/bad folder labels found; metrics skipped (CSV predictions still saved).")
        return
    if len(labeled) != len(rows):
        print(f"Only {len(labeled)} of {len(rows)} images have good/bad labels; metrics use labeled images only.")

    # Rows: actual labels [good, bad]; columns: predicted [good, bad].
    tn = sum(r["true_label"] == "good" and r["prediction"] == "good" for r in labeled)
    fp = sum(r["true_label"] == "good" and r["prediction"] == "bad" for r in labeled)
    fn = sum(r["true_label"] == "bad" and r["prediction"] == "good" for r in labeled)
    tp = sum(r["true_label"] == "bad" and r["prediction"] == "bad" for r in labeled)
    p_bad, r_bad, f_bad = per_class_stats(tp, fp, fn)
    p_good, r_good, f_good = per_class_stats(tn, fn, fp)
    accuracy = (tp + tn) / len(labeled)
    print(f"\nLabeled images: {len(labeled)} | good={tn + fp} | bad={tp + fn}")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Precision (bad):     {p_bad:.4f}")
    print(f"Recall (bad):        {r_bad:.4f}")
    print(f"F1 (bad):            {f_bad:.4f}")
    print(f"Macro precision:     {(p_bad + p_good) / 2:.4f}")
    print(f"Macro recall:        {(r_bad + r_good) / 2:.4f}")
    print(f"Macro F1:            {(f_bad + f_good) / 2:.4f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print("              pred good  pred bad")
    print(f"  true good   {tn:>9}  {fp:>8}")
    print(f"  true bad    {fn:>9}  {tp:>8}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n document quality classification: good vs bad")
    parser.add_argument("--source", required=True, help="A photo, image directory, or labeled good/bad folders")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Predict bad when P(bad) >= threshold (0-1)")
    parser.add_argument("--device", default="cpu", help="cpu or e.g. 0 for first CUDA GPU")
    parser.add_argument("--batch", type=int, default=16, help="Prediction batch size")
    parser.add_argument("--out", default="predictions.csv", help="CSV output path")
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.batch < 1:
        parser.error("--batch must be a positive integer")
    paths = find_images(args.source)
    if not paths:
        raise SystemExit(f"No supported images found in {args.source}")
    weights = Path(args.weights).expanduser()
    if not weights.is_file():
        raise SystemExit(f"Weights file not found: {weights}")

    model = YOLO(str(weights))
    if getattr(model, "task", "classify") != "classify":
        raise SystemExit("Checkpoint must be a YOLO classification model, not an object detector.")
    classes = model.names.items() if hasattr(model.names, "items") else enumerate(model.names)
    labels = {str(value).lower().strip(): int(index) for index, value in classes}
    if not {"good", "bad"}.issubset(labels):
        raise SystemExit(f"Model classes do not contain good and bad: {model.names}")
    bad_idx = labels["bad"]
    print(f"Images: {len(paths)} | device: {args.device} | threshold: {args.threshold} | classes: {model.names}")

    rows = []
    for i in range(0, len(paths), args.batch):
        batch = paths[i:i + args.batch]
        results = model.predict([str(p) for p in batch], imgsz=224, device=args.device, verbose=False)
        if len(results) != len(batch):
            raise RuntimeError("Model output count does not match input count")
        for path, result in zip(batch, results):
            if result.probs is None:
                raise RuntimeError("Expected class probabilities; check loaded model")
            prob_bad = float(result.probs.data[bad_idx])
            rows.append({"path": str(path), "prob_bad": f"{prob_bad:.6f}",
                         "prediction": "bad" if prob_bad >= args.threshold else "good",
                         "true_label": ground_truth(path)})

    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "prob_bad", "prediction", "true_label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output.resolve()}")
    print_metrics(rows)
    print("IMPORTANT: This script only predicts and evaluates. It never trains or modifies the checkpoint.")


if __name__ == "__main__":
    main()
