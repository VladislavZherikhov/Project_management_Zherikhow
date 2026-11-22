import argparse
from collections import Counter
import json
from pathlib import Path

from pipeline import BASE, DocumentPipeline


def run_batch(pipeline, directory, output):
    predictions = [pipeline.run(path) for path in sorted(Path(directory).iterdir())
                   if path.is_file()]
    counts = Counter(item["status"] for item in predictions)
    report = {"counts": {key: counts[key] for key in ("accepted", "rejected", "error")},
              "predictions": predictions}
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=BASE / "samples")
    parser.add_argument("--output", type=Path, default=BASE / "predictions.json")
    args = parser.parse_args()
    print(run_batch(DocumentPipeline(), args.source, args.output)["counts"])
