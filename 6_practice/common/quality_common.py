"""
Shared code for Practice 6 (Tasks 2-4): split, data loading, evaluation,
latency protocol and plots. Every model is evaluated with the same functions,
so the comparison between models is fair.

Labels: good = 0, bad = 1 (bad is the positive class, see Task 1 memo).
"""

import io
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

PRACTICE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PRACTICE_DIR.parent
DATASET_DIR = PROJECT_DIR / "5_practice" / "data" / "quality_dataset"
SPLIT_FILE = PRACTICE_DIR / "common" / "split_70_15_15.jsonl"

SEED = 24500003  # student ID
SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = (0.70, 0.15, 0.15)
CLASS_TO_LABEL = {"good": 0, "bad": 1}
LABEL_TO_CLASS = {0: "good", 1: "bad"}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 70 / 15 / 15 split by source image, stratified by document class
# ============================================================

def make_split(dataset_dir=DATASET_DIR, out_path=SPLIT_FILE, seed=SEED):
    """
    Re-assign the split so that it is exactly 70/15/15 inside every document
    class. The unit of the split is the source document (source_id): all
    good/bad variants of one source stay in the same split (no leakage).
    """
    records = [json.loads(line) for line in
               (Path(dataset_dir) / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]

    sources_by_class = defaultdict(set)
    for r in records:
        sources_by_class[r["class_name"]].add(r["source_id"])

    rng = random.Random(seed)
    source_split = {}
    for class_name in sorted(sources_by_class):
        sources = sorted(sources_by_class[class_name])
        rng.shuffle(sources)
        n_train = round(len(sources) * SPLIT_RATIOS[0])
        n_val = round(len(sources) * SPLIT_RATIOS[1])
        for i, source in enumerate(sources):
            source_split[source] = ("train" if i < n_train
                                    else "validation" if i < n_train + n_val
                                    else "test")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            r = dict(r, split=source_split[r["source_id"]], split_seed=seed)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path


def load_split(split_file=SPLIT_FILE):
    records = [json.loads(line) for line in
               Path(split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {s: [r for r in records if r["split"] == s] for s in SPLITS}


def check_split(splits):
    """Print the split table and assert that there is no source leakage."""
    owner = {}
    for name, recs in splits.items():
        for r in recs:
            assert owner.setdefault(r["source_id"], name) == name, \
                f"leakage: {r['source_id']} in {owner[r['source_id']]} and {name}"
    total = sum(len(v) for v in splits.values())
    for name, recs in splits.items():
        q = Counter(r["quality_label"] for r in recs)
        c = Counter(r["class_name"] for r in recs)
        print(f"{name:<10} {len(recs):>5} ({len(recs) / total:.0%})  "
              f"good={q['good']} bad={q['bad']}  "
              + " ".join(f"{k}={c[k]}" for k in sorted(c)))
    print("No source_id appears in two splits: OK")


# ============================================================
# Data: images are decoded once and cached as uint8 tensors
# ============================================================

def load_images(records, size=224, dataset_dir=DATASET_DIR):
    """RGB, resized to size x size, uint8 tensor [N, 3, H, W] + labels [N]."""
    images = torch.empty((len(records), 3, size, size), dtype=torch.uint8)
    for i, r in enumerate(records):
        with Image.open(Path(dataset_dir) / r["image_path"]) as im:
            im = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        images[i] = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
    labels = torch.tensor([CLASS_TO_LABEL[r["quality_label"]] for r in records],
                          dtype=torch.float32)
    return images, labels


def imagenet_normalize(batch_uint8):
    """uint8 [B,3,H,W] -> float, ImageNet mean/std normalisation."""
    x = batch_uint8.float() / 255.0
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def random_rot90(batch, generator):
    """
    Rotate every image by a random multiple of 90 degrees (both classes).
    In the dataset only GOOD images are rotated, so without this the model
    could learn the shortcut "rotated = good" (see Task 1 memo, risks).
    """
    ks = torch.randint(0, 4, (batch.shape[0],), generator=generator)
    return torch.stack([torch.rot90(img, int(k), dims=(1, 2)) for img, k in zip(batch, ks)])


# ============================================================
# Evaluation
# ============================================================

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_bad": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_bad": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_bad": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_good": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_good": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),  # rows = true [good, bad], cols = predicted
    }


def breakdown(records, y_prob, threshold=0.5):
    """Recall of BAD per defect, error rate of GOOD per augmentation, accuracy per class."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    per_defect, per_aug, per_class = defaultdict(list), defaultdict(list), defaultdict(list)
    for r, p in zip(records, y_pred):
        correct = int(p == CLASS_TO_LABEL[r["quality_label"]])
        per_class[r["class_name"]].append(correct)
        if r["quality_label"] == "bad":
            per_defect[r["defect_type"]].append(correct)
        else:
            aug = r["good_augmentation"]
            aug = "slight_rotation" if aug.startswith("slight_rotation") else aug
            per_aug[aug].append(correct)
    mean = lambda d: {k: round(float(np.mean(v)), 4) for k, v in sorted(d.items())}
    return {
        "recall_bad_per_defect": mean(per_defect),
        "accuracy_good_per_augmentation": mean(per_aug),
        "accuracy_per_document_class": mean(per_class),
    }


def acceptance_check(metrics, parts, latency_ms, size_mb):
    """Acceptance criteria fixed in the Task 1 memo (section 5)."""
    checks = {
        "Q1 accuracy >= 0.92": metrics["accuracy"] >= 0.92,
        "Q2 F1(bad) >= 0.92": metrics["f1_bad"] >= 0.92,
        "Q3 recall(bad) >= 0.93": metrics["recall_bad"] >= 0.93,
        "Q4 precision(bad) >= 0.88": metrics["precision_bad"] >= 0.88,
        "Q5 recall per defect >= 0.85": min(parts["recall_bad_per_defect"].values()) >= 0.85,
        "C1 CPU latency <= 50 ms": latency_ms <= 50,
        "C2 size <= 25 MB": size_mb <= 25,
    }
    return checks


# ============================================================
# Cost metrics: same protocol as the Task 1 benchmark
# ============================================================

def model_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 1024 ** 2


def cpu_latency_ms(model, size=224, warmup=10, runs=100):
    """batch 1, fp32, CPU: warm-up runs, then mean/std of timed runs."""
    model = model.to("cpu").eval()
    x = torch.randn(1, 3, size, size)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(times), statistics.stdev(times)


# ============================================================
# Plots
# ============================================================

def plot_history(history, title, out_path):
    import matplotlib.pyplot as plt
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, name in ((axes[0], "loss", "Loss (BCE)"), (axes[1], "acc", "Accuracy")):
        ax.plot(epochs, [h[f"train_{key}"] for h in history], "o-", label="train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], "s-", label="validation")
        ax.set_xlabel("epoch"); ax.set_title(name); ax.grid(alpha=0.3); ax.legend()
    best = min(history, key=lambda h: h["val_loss"])["epoch"]
    for ax in axes:
        ax.axvline(best, color="gray", ls="--", lw=1)
    fig.suptitle(f"{title} (dashed line = best epoch {best}, restored)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    return fig


def plot_confusion(cm, title, out_path):
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["good", "bad"]); ax.set_yticks([0, 1], ["good", "bad"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    return fig
