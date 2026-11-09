# YOLOv8n-cls Document Photo Quality Gate

A binary classifier that decides whether a photo of a document is **good** (readable) or **bad** (blur, glare, strong skew, occlusion, bad exposure). It was selected as the best model in Practice 6 (author: Vladislav Zherikhov).

## Files

| File | Content |
|---|---|
| `yolov8n_quality_gate.pt` | Trained Ultralytics YOLOv8n-cls weights (2.8 MB) |
| `predict.py` | Runs the model on an image or a folder, saves a CSV, and computes metrics if labels are known |

## Quick start

```bash
pip install ultralytics
python predict.py --source path/to/your/images
```

### Run with labels to get metrics

Put your images into `good/` and `bad/` sub-folders, e.g.

```
my_data/
├── good/  ...jpg
└── bad/   ...jpg
```

```bash
python predict.py --source my_data --out my_predictions.csv
```

The script prints accuracy, precision, recall, F1 and the confusion matrix, with **bad as the positive class**. It also writes `path, prob_bad, prediction, true_label` for every image to the CSV.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--source` | (required) | image, folder, or folder with `good/` and `bad/` (any depth) |
| `--threshold` | 0.5 | an image is predicted **bad** if P(bad) ≥ threshold. Lower it (e.g. 0.3) to catch more bad photos at the cost of more false alarms |
| `--device` | cpu | `0` to use the first GPU |
| `--out` | predictions.csv | output CSV |

### Use from Python

```python
from ultralytics import YOLO
model = YOLO("yolov8n_quality_gate.pt")      # classes: {0: 'bad', 1: 'good'}
r = model.predict("photo.jpg", imgsz=224, verbose=False)[0]
print(model.names[r.probs.top1], float(r.probs.data[0]))   # label, P(bad)
```

## Model details

| | |
|---|---|
| Architecture | Ultralytics YOLOv8n-cls, ImageNet-pretrained, fully fine-tuned |
| Classes | `0 = bad`, `1 = good` (softmax) |
| Input | RGB image of any size; resized to **224×224** internally by Ultralytics |
| Training data | 1260 synthetic images of Russian documents (passport, INN certificate, SNILS), 50 % good / 50 % bad |
| Bad defects in training | Gaussian blur, glare (white ellipses), strong skew 15–35°, occlusion (dark rectangles), bad exposure (dark / bright / low contrast) |
| Good variants in training | clean, slight rotation ±3–8°, rotated 90° / 180° / 270° (a rotated but sharp document is **good**) |
| Training | 224×224, AdamW lr 1e-3, batch 32, early stopping, seed 24500003; no flips, no random erasing |

## Results on our test split (270 synthetic images, never seen in training)

| Accuracy | Precision (bad) | Recall (bad) | F1 (bad) | Missed bad / false alarms | CPU latency (i7-10750H) | Size |
|---|---|---|---|---|---|---|
| 0.996 | 1.000 | 0.993 | 0.996 | 1 / 0 | ~10 ms per image | 2.8 MB file |

## Limitations (please keep in mind)

- **Trained only on synthetic data.** Expect lower accuracy on real phone photos; a 3–6 % drop is typical for synthetic → real. Measuring this is exactly why it is useful to run it on your data.
- **Document types:** passport main page, INN certificate, SNILS. Other documents or backgrounds may behave differently.
- **Defect definitions:** a document that is rotated but sharp counts as **good**. Glare and over-exposure on very white forms are the weakest cases.
- **What would help:** after the run, please send back `my_predictions.csv` or the printed metrics, ideally with a few examples of misclassified images.
