# Dataset Card — Document Image Quality (GOOD / BAD)

> All source documents are synthetic (Week 2–3 generators) and carry the watermark "SYNTHETIC TRAINING SAMPLE - NOT A REAL DOCUMENT". No real personal data.

## 1. Goal

A balanced dataset for a binary image-quality classifier: **GOOD** — an OCR engine can read the document (even if the phone was held at an angle), **BAD** — the text is destroyed or hidden.

## 2. Project structure

```
4 практика/
├── build_good_bad_dataset.py           augmentation pipeline (provided, with CLI args added)
├── build_good_bad_dataset_original.py  original script from the course resource pack
├── audit_quality_dataset.py            audit: files vs metadata, balance, split leakage, contact sheets
├── data/
│   ├── raw/                            clean source images (Step 1)
│   │   ├── inn/        100 images
│   │   ├── snils/      100 images
│   │   └── passport/   100 images
│   ├── demo_quality_dataset/           in-class demo: 5 per class
│   └── quality_dataset_300/            homework: 300 per class
└── DATASET_CARD.md
```

Each built dataset contains `good/`, `bad/`, `preprocessed/`, `metadata.jsonl`, `summary.json` and `audit/` (contact sheets).

## 3. Source data (Step 1)

| Class | Count | Source | Original size |
|---|---|---|---|
| inn | 100 | Week 2 INN certificate generator (`новая лаба проект 2/lab/inn`) | 786 × 1024 |
| snils | 100 | Week 2 SNILS generator (`новая лаба проект 2/lab/snils`) | 471 × 320 |
| passport | 100 | Week 3 passport generator (`3 практика/generated_passports_100`, seed 20260001) | 608 × 759 |

## 4. Augmentation rules (Step 2)

| Class | Definition | Augmentations used |
|---|---|---|
| GOOD | Text is sharp and readable | none (clean), slight rotation ±3…8°, upside down 180°, rotation 90° / 270° |
| BAD | Text is destroyed or hidden | Gaussian blur (r 3–6), glare (1–3 white ellipses, alpha 150–230), strong skew (±15…35°), occlusion (1–2 dark rectangles), bad exposure (dark / bright / low contrast) |

No horizontal or vertical flipping: mirrored text creates fake languages and unrealistic layouts. All images are resized to 640 × 640.

## 5. How it was built (Step 3)

```
# in-class demo: 5 images per class
python build_good_bad_dataset.py

# homework: 100 raw images per class -> 300 per class (3 variants per source)
python build_good_bad_dataset.py --output data/quality_dataset_300 --target 300 --variants 3
```

Seed: **2026** (script default). For every source image the pipeline makes one GOOD and one BAD variant (×`variants`). BAD defects are cycled in a fixed order, so all 5 defects are equally represented.

Changes to the provided script:
- Added command-line arguments (`--input --output --target --variants --seed`). The defaults are the same as in the SESSION SETTINGS block, so running without arguments gives the demo.
- In `summary.json`, slight rotations are grouped into one `slight_rotation` key. Before, each angle was its own key, which hid the balance. The exact angle is still stored in `metadata.jsonl`.

## 6. Results and audit (Step 4)

`python audit_quality_dataset.py <dataset>` checks that every metadata line has an image and every image has a metadata line, checks the balance, checks for split leakage, and saves contact sheets.

| | demo_quality_dataset | quality_dataset_300 |
|---|---|---|
| Total images | 30 | 1800 |
| good / bad | 15 / 15 | 900 / 900 |
| Per class (good = bad) | 5 each | 300 each |
| Defects (bad) | 3 of each of 5 | 180 of each of 5 (60 per class) |
| Good augmentations | none 5, slight_rotation 3, rotate_90 3, rotate_270 4 | none 240, slight_rotation 292, upside_down 126, rotate_90 120, rotate_270 122 |
| Split by images (train / val / test) | 26 / 2 / 2 | 1314 / 222 / 264 |
| Source images | 15 | 300 |
| Split leakage | 0 sources | 0 sources |
| Missing files | 0 | 0 |

**Visual audit** (`data/*/audit/sheet_good.png`, `sheet_bad.png`):
- GOOD: all documents are readable; rotated and upside-down documents are present. In the 5-image demo there is no upside-down example by chance, but the 300-per-class set has 126.
- BAD: all 5 defects are clearly visible on passports and SNILS.
- `summary.json`: classes and good/bad are perfectly balanced.
- `metadata.jsonl`: the split is assigned once per **source image**. All GOOD and BAD variants of one source share the same split, so there is no data leakage between train, validation and test.

## 7. Problems found during the audit

| Problem | Impact | Suggested fix |
|---|---|---|
| BAD images are never rotated 90/180/270°, while ~40% of GOOD images are | Shortcut: the model can learn "rotated = good" instead of judging quality | Apply the same orientation augmentations to BAD images before adding the defect |
| Glare and "bright" exposure are weak on the white INN form | Some BAD INN images are still readable, which causes label noise | Make glare/exposure strength depend on background brightness, or check with OCR that the defect really breaks reading |
| Skew is ±15…35° while the guide says strong skew ±30° | Skew around 15° is close to the GOOD ±8° limit | Use a 20…35° range |
| Resize to 640 × 640 ignores the aspect ratio (SNILS 471×320 gets stretched vertically) | Text shapes are distorted differently per class | Letterbox: resize keeping the aspect ratio and pad to 640 × 640 |
| Random split with few sources (demo: 13/1/1 sources) | Validation and test are too small in the demo | Use a stratified split by class with exact 70/15/15 proportions |
| Passport sources come from one template | Low variety inside the class | Add more templates and backgrounds |

## 8. Ethics

- All documents are synthetic, generated with Faker; faces on passports are synthetic portraits.
- The watermark "SYNTHETIC TRAINING SAMPLE - NOT A REAL DOCUMENT" stays on every image. Augmentations may partly cover it (occlusion, glare), but they never remove it from the source.
- The dataset is intended only for training and testing document-quality and OCR systems.

## 9. Next steps

- Fix the orientation shortcut (see section 7) and rebuild.
- Add realistic defects: motion blur, perspective transform (photo of a document on a table), JPEG compression, shadows, fingers instead of rectangles.
- Validate labels automatically with an OCR engine (e.g. Tesseract): GOOD must be readable, BAD must not be.
- Train a baseline classifier (e.g. a fine-tuned ResNet-18 / MobileNet) on `quality_dataset_300` and report per-defect accuracy.
