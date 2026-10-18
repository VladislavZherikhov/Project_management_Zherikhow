# Homework 4: Dataset Audit and Data Loading

## Project structure

```
5 практика/
├── scripts/
│   ├── audit_quality_dataset.py   dataset audit (provided, extended)
│   └── quality_dataloader.py      PyTorch Dataset + DataLoader (provided, extended)
├── data/
│   ├── quality_dataset/           Homework 3 dataset (300 per class), copied from 4 практика/data/quality_dataset_300
│   └── demo_quality_dataset/      in-class demo dataset (5 per class)
├── reports/
│   ├── dataset_audit_report.json        audit of data/quality_dataset (main report)
│   ├── demo_dataset_audit_report.json   audit of the demo dataset
│   ├── broken_demo_audit_report.json    audit of a deliberately broken copy (self-test)
│   ├── dataloader_output.txt            DataLoader test, quality_dataset
│   └── demo_dataloader_output.txt       DataLoader test, demo dataset
└── REPORT.md
```

Commands:

```
python scripts/audit_quality_dataset.py --dataset_dir data/quality_dataset --output_report reports/dataset_audit_report.json
python scripts/quality_dataloader.py --dataset_dir data/quality_dataset
python scripts/quality_dataloader.py --dataset_dir data/demo_quality_dataset --split all --batch_size 4
```

## Dataset

Folder: `data/quality_dataset` (`good/`, `bad/`, `metadata.jsonl`, `summary.json`). It is the Homework 3 dataset: synthetic INN, SNILS and passport images, 640 × 640, seed 2026.

## Total records

- Total valid records: **1800**
- Total errors: **0**

## Quality balance

- good: 900
- bad: 900

Bad defects: blur 180, glare 180, skew 180, occlusion 180, exposure 180.
Good augmentations: none 240, slight_rotation 292, upside_down 126, rotate_90 120, rotate_270 122.

## Class balance

- inn: 600
- snils: 600
- passport: 600

## Split distribution

| Split | Images | good / bad | Source images | inn | snils | passport |
|---|---|---|---|---|---|---|
| train | 1314 (73%) | 657 / 657 | 219 | 456 | 426 | 432 |
| validation | 222 (12%) | 111 / 111 | 37 | 66 | 72 | 84 |
| test | 264 (15%) | 132 / 132 | 44 | 78 | 102 | 84 |

Every split has exactly the same number of good and bad images, because each source image produces both a good and a bad variant in the same split.

## Errors

- No missing metadata fields.
- No missing image files.
- No corrupted images (every file passes `verify()` and a full pixel decode).
- No wrong image sizes (all 640 × 640).
- No invalid quality labels, split values or class names.
- No good/bad folder vs label mismatch.
- No duplicate image IDs and no duplicate image paths.
- No image files without a metadata record.

## Warnings

None (`"warnings": []`). The demo dataset also has no warnings: 30 records, 0 errors, all three splits non-empty (26 / 2 / 2).

Observations that are not audit warnings:
- The split is random by source, so the ratio is 73 / 12 / 15 instead of an exact 70 / 15 / 15, and the class mix inside validation/test is a little uneven (inn 66 vs passport 84 in validation). For the next version a stratified split by class is recommended.
- Known from the Homework 3 visual audit: BAD images are never rotated 90/180/270°, so a model could learn "rotated = good". The audit script cannot detect this; it has to be fixed in the augmentation pipeline.

## Data leakage

**No data leakage found.** All 300 source images are assigned to exactly one split, so no source image appears in two splits (`"source_id_leakage": {}`).

## DataLoader test

```
train       1314  (good=657, bad=657)
validation   222  (good=111, bad=111)
test         264  (good=132, bad=132)

Batch shape: torch.Size([16, 3, 224, 224])
Labels shape: torch.Size([16])
Example labels (train): [0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1]
                        good=0, bad=1
```

Transform: resize 640 → 224, `ToTensor`, ImageNet normalization (ready for pretrained CNNs). The training loader is shuffled; the validation and test loaders keep a fixed order.

## Changes to the provided scripts

`audit_quality_dataset.py`:
- `class_name` added to the required fields and validated (`inn`, `snils`, `passport`).
- New check: the image folder must match the label (`good/…` ↔ `good`).
- Corrupted-image check now runs `verify()` before the full decode.
- New check: image files on disk without a metadata record (`orphan_files`).
- Leakage is counted only for valid split values (an invalid split is reported as an error, not as leakage).
- Slight rotations are grouped in the summary (the exact angle stays in metadata).

`quality_dataloader.py`:
- Without `--split`, it loads train, validation and test separately and prints the size, good/bad count, one batch shape and example labels for each split.
- `--split all` loads every record (for small demo datasets, see Common Error 4). The old `choices=[None, …]` value could not be passed from the command line.
- Default batch size 16, fixed seed for shuffling, `num_workers=0` (safe on Windows).

**Audit self-test.** To check that the audit really finds problems, a temporary copy of the demo dataset was broken on purpose: a truncated JPEG, a deleted file, a 500 × 500 image, `quality_label = "medium"`, `split = "training"`, a duplicate `image_id`, a missing `source_id`, a source moved to another split, and an extra unlisted file. The audit found all 9 problems: 6 errors, plus the duplicate ID, the leakage and the orphan files as warnings. The result is in `reports/broken_demo_audit_report.json`.

## Conclusion

The dataset is balanced (900 good / 900 bad, 600 per document class, 180 per defect type), clean (0 errors, 0 warnings), has no data leakage between splits, and loads correctly into PyTorch with batches of shape `[16, 3, 224, 224]`. **The dataset is ready for model training.** Recommended improvements before the final version: a stratified 70/15/15 split and rotated BAD samples to remove the orientation shortcut.
