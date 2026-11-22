import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import joblib
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from doctype import features

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "5_practice/data/quality_dataset")
    parser.add_argument("--split", type=Path, default=ROOT / "6_practice/common/split_70_15_15.jsonl")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.split.read_text(encoding="utf-8").splitlines()]
    groups = {split: set() for split in ("train", "validation", "test")}
    data = {split: ([], []) for split in groups}
    for record in records:
        split = record["split"]
        groups[split].add(record["source_id"])
        if record["quality_label"] == "good":
            vectors, labels = data[split]
            with Image.open(args.dataset / record["image_path"]) as image:
                vectors.append(features(image))
            labels.append(record["safe_class"].upper())
    assert sum(map(len, groups.values())) == len(set.union(*groups.values()))
    model = LogisticRegression(max_iter=500, random_state=24500003)
    model.fit(*data["train"])
    metrics = {"classes": model.classes_.tolist(), "splits": {}}
    for split, (vectors, labels) in data.items():
        predicted = model.predict(vectors)
        metrics["splits"][split] = {
            "counts": dict(Counter(labels)),
            "accuracy": accuracy_score(labels, predicted),
            "classification_report": classification_report(labels, predicted, output_dict=True),
            "confusion_matrix": confusion_matrix(labels, predicted, labels=model.classes_).tolist(),
        }
    timings = []
    samples = [row for row in records if row["split"] == "test" and row["quality_label"] == "good"]
    for row in samples:
        started = perf_counter()
        with Image.open(args.dataset / row["image_path"]) as image:
            model.predict_proba([features(image)])
        timings.append((perf_counter() - started) * 1000)
    metrics["test_latency_ms_mean"] = round(mean(timings), 3)
    (BASE / "models").mkdir(exist_ok=True)
    joblib.dump(model, BASE / "models/doctype.joblib", compress=3)
    (BASE / "doctype_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print({name: result["accuracy"] for name, result in metrics["splits"].items()})


if __name__ == "__main__":
    main()
