"""
Group task: evaluate a classmate's YOLOv8n-cls quality gate on my dataset.

Input:  outputs/peer_predictions_full.csv, produced by the classmate's own script:
        python yolov8n_quality_gate/predict.py --source ../../5_practice/data/quality_dataset
                                               --out outputs/peer_predictions_full.csv
Output: outputs/peer_evaluation.json, outputs/*.csv, outputs/*.png

The classmate's model never saw any of my 1800 images, so the full dataset is a
fair external test set. For a head-to-head comparison with my YOLOv8n-cls, the
same metrics are also computed on my 270-image test split (my model has not seen
these images either).
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from quality_common import (DATASET_DIR, CLASS_TO_LABEL, load_split, compute_metrics,  # noqa: E402
                            breakdown, acceptance_check, model_size_mb, cpu_latency_ms)

OUT = HERE / "outputs"
PEER_WEIGHTS = HERE / "yolov8n_quality_gate" / "yolov8n_quality_gate.pt"
MY_TEST_PREDICTIONS = HERE.parent / "task3" / "outputs" / "yolov8n_cls_test_predictions.csv"
MY_TEST_METRICS = HERE.parent / "task3" / "outputs" / "test_metrics.json"


def read_peer_predictions():
    with open(OUT / "peer_predictions_full.csv", encoding="utf-8") as f:
        return {Path(r["path"]).stem: float(r["prob_bad"]) for r in csv.DictReader(f)}


def read_my_test_predictions():
    with open(MY_TEST_PREDICTIONS, encoding="utf-8") as f:
        return {r["image_id"]: float(r["prob_bad"]) for r in csv.DictReader(f)}


def evaluate(records, probs):
    y_true = [CLASS_TO_LABEL[r["quality_label"]] for r in records]
    metrics = compute_metrics(y_true, probs)
    parts = breakdown(records, probs)
    return metrics, parts


def round_dict(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def peer_cost():
    """Latency and size with the same protocol as Tasks 1-4 (CPU, batch 1, 224, fp32)."""
    from ultralytics import YOLO
    net = YOLO(str(PEER_WEIGHTS)).model.float()
    mean, std = cpu_latency_ms(net)
    return {"latency_ms": round(mean, 1), "latency_std_ms": round(std, 1),
            "size_mb_state_dict": round(model_size_mb(net), 1),
            "file_size_mb": round(PEER_WEIGHTS.stat().st_size / 1024 ** 2, 1),
            "params_m": round(sum(p.numel() for p in net.parameters()) / 1e6, 2)}


def plot_confusions(items, path):
    fig, axes = plt.subplots(1, len(items), figsize=(4.2 * len(items), 3.8))
    for ax, (title, cm) in zip(axes, items):
        cm = np.array(cm)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["good", "bad"])
        ax.set_yticks([0, 1], ["good", "bad"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_per_group(peer_full, peer_test, mine_test, path):
    groups = [("defect: " + k, "recall_bad_per_defect", k) for k in peer_full["recall_bad_per_defect"]]
    groups += [("good: " + k, "accuracy_good_per_augmentation", k)
               for k in peer_full["accuracy_good_per_augmentation"]]
    labels = [g[0] for g in groups]
    series = [("classmate model, full 1800", peer_full, "#4C72B0"),
              ("classmate model, my test 270", peer_test, "#88A9D6"),
              ("my YOLOv8n-cls, my test 270", mine_test, "#DD8452")]
    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for i, (name, parts, color) in enumerate(series):
        vals = [parts[key][k] for _, key, k in groups]
        ax.bar(x + (i - 1) * w, vals, w, label=name, color=color)
    ax.axhline(0.85, color="red", ls="--", lw=1, label="Q5 threshold 0.85")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("share correct")
    ax.set_title("Recall of bad per defect / accuracy of good per augmentation")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_gallery(errors, path, n=12):
    errors = errors[:n]
    cols = 6
    rows = (len(errors) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.9 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, e in zip(np.ravel(axes), errors):
        ax.imshow(Image.open(DATASET_DIR / e["image_path"]).convert("RGB").resize((256, 256)))
        kind = e["defect_type"] if e["quality_label"] == "bad" else e["good_augmentation"]
        ax.set_title(f"true {e['quality_label']} ({kind})\nP(bad)={e['prob_bad']:.2f}  {e['class_name']}",
                     fontsize=8, color="red")
    fig.suptitle("Most confident errors of the classmate model on my dataset", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    splits = load_split()
    all_records = [r for s in splits.values() for r in s]
    test_records = splits["test"]
    peer = read_peer_predictions()
    mine = read_my_test_predictions()
    assert len(peer) == len(all_records) == 1800, "predictions must cover the full dataset"

    peer_full_m, peer_full_p = evaluate(all_records, [peer[r["image_id"]] for r in all_records])
    peer_test_m, peer_test_p = evaluate(test_records, [peer[r["image_id"]] for r in test_records])
    mine_test_m, mine_test_p = evaluate(test_records, [mine[r["image_id"]] for r in test_records])

    cost = peer_cost()
    my_cost = json.loads(MY_TEST_METRICS.read_text(encoding="utf-8"))
    acceptance = acceptance_check(peer_full_m, peer_full_p, cost["latency_ms"], cost["size_mb_state_dict"])

    # Threshold sweep: diagnostic only (chosen on the evaluation data itself, so not a fair result)
    y_full = [CLASS_TO_LABEL[r["quality_label"]] for r in all_records]
    p_full = [peer[r["image_id"]] for r in all_records]
    sweep = []
    for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = compute_metrics(y_full, p_full, threshold=t)
        sweep.append({"threshold": t, "accuracy": round(m["accuracy"], 4),
                      "precision_bad": round(m["precision_bad"], 4),
                      "recall_bad": round(m["recall_bad"], 4), "f1_bad": round(m["f1_bad"], 4),
                      "fn": m["confusion_matrix"][1][0], "fp": m["confusion_matrix"][0][1]})

    # Error list (full dataset), most confident first
    errors = []
    for r in all_records:
        p = peer[r["image_id"]]
        if (p >= 0.5) != (r["quality_label"] == "bad"):
            errors.append({**{k: r[k] for k in ("image_id", "image_path", "class_name", "quality_label",
                                                "defect_type", "good_augmentation", "split")},
                           "prob_bad": round(p, 4),
                           "confidence_of_error": round(p if p >= 0.5 else 1 - p, 4)})
    errors.sort(key=lambda e: -e["confidence_of_error"])
    with open(OUT / "peer_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(errors[0]))
        w.writeheader()
        w.writerows(errors)

    # Error counts by document class x kind
    by_group = {}
    for e in errors:
        kind = e["defect_type"] if e["quality_label"] == "bad" else "good/" + (
            "slight_rotation" if e["good_augmentation"].startswith("slight_rotation") else e["good_augmentation"])
        key = f"{e['class_name']} | {kind}"
        by_group[key] = by_group.get(key, 0) + 1

    # Head-to-head on my test split
    head_to_head = []
    for r in test_records:
        pp, mp = peer[r["image_id"]], mine[r["image_id"]]
        truth = r["quality_label"] == "bad"
        if (pp >= 0.5) != truth or (mp >= 0.5) != truth:
            head_to_head.append({"image_id": r["image_id"], "true": r["quality_label"],
                                 "defect_or_aug": r["defect_type"] if truth else r["good_augmentation"],
                                 "peer_prob_bad": round(pp, 4), "peer_correct": (pp >= 0.5) == truth,
                                 "my_prob_bad": round(mp, 4), "my_correct": (mp >= 0.5) == truth})
    with open(OUT / "test_split_disagreements.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(head_to_head[0]))
        w.writeheader()
        w.writerows(head_to_head)

    # False alarms on slightly rotated good images, by document and |angle|
    angle_bins = {}
    for r in all_records:
        aug = r["good_augmentation"]
        if r["quality_label"] == "good" and aug.startswith("slight_rotation"):
            a = abs(float(aug.rsplit("_", 1)[-1]))
            b = "3-5 deg" if a < 5 else ("5-6.5 deg" if a < 6.5 else "6.5-8 deg")
            cell = angle_bins.setdefault(r["class_name"], {}).setdefault(b, [0, 0])
            cell[0] += peer[r["image_id"]] >= 0.5
            cell[1] += 1
    angle_bins = {d: {b: f"{v[0]}/{v[1]}" for b, v in sorted(bins.items())} for d, bins in sorted(angle_bins.items())}

    result = {
        "peer_model": {"weights": "yolov8n_quality_gate/yolov8n_quality_gate.pt", "classes": {0: "bad", 1: "good"},
                       "train_args_from_checkpoint": {"base": "yolov8n-cls.pt", "epochs": 15, "patience": 4,
                                                      "batch": 16, "imgsz": 224, "optimizer": "Adam",
                                                      "lr0": 0.0001, "dropout": 0.3, "seed": 2473986,
                                                      "degrees": 0.0, "fliplr": 0.0, "flipud": 0.0,
                                                      "hsv_s": 0.7, "hsv_v": 0.4, "scale": 0.5,
                                                      "device": "GPU (Colab)"},
                       "reported_val_accuracy_on_own_data": 0.99259},
        "full_dataset_1800": {"metrics": round_dict(peer_full_m), "breakdown": peer_full_p,
                              "acceptance_criteria": acceptance},
        "my_test_split_270": {"peer": {"metrics": round_dict(peer_test_m), "breakdown": peer_test_p},
                              "mine_yolov8n_cls": {"metrics": round_dict(mine_test_m), "breakdown": mine_test_p}},
        "cost": {"peer": cost, "mine_yolov8n_cls": {
            "latency_ms_task4_session": 9.5, "size_mb_state_dict": 5.6,
            "train_time_s": my_cost["results"]["yolov8n_cls"]["train_time_s"]}},
        "threshold_sweep_full_dataset_diagnostic": sweep,
        "errors_by_document_and_kind": dict(sorted(by_group.items(), key=lambda kv: -kv[1])),
        "false_alarms_slight_rotation_by_angle": angle_bins,
        "n_errors": len(errors),
    }
    (OUT / "peer_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    plot_confusions([("Classmate model\nmy full dataset (1800)", peer_full_m["confusion_matrix"]),
                     ("Classmate model\nmy test split (270)", peer_test_m["confusion_matrix"]),
                     ("My YOLOv8n-cls\nmy test split (270)", mine_test_m["confusion_matrix"])],
                    OUT / "confusion_matrices.png")
    plot_per_group(peer_full_p, peer_test_p, mine_test_p, OUT / "per_defect_comparison.png")
    plot_gallery(errors, OUT / "peer_failure_gallery.png")

    print(json.dumps({k: result[k] for k in ("full_dataset_1800", "my_test_split_270", "cost",
                                             "threshold_sweep_full_dataset_diagnostic",
                                             "errors_by_document_and_kind")}, indent=1))


if __name__ == "__main__":
    main()
