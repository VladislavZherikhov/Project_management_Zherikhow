# Dataset Card — Synthetic Russian Passport Main Pages

> SYNTHETIC TRAINING SAMPLE — NOT A REAL DOCUMENT. Educational dataset for document recognition tasks.

## 1. Project structure

```
project/
├── passport_generator.py           generator (CONFIG filled in for the template)
├── passport_generator_original.py  original code from the course resource pack
├── coordinate_finder.py            coordinate measuring tool (click → (x, y))
├── audit_metadata.py               metadata.jsonl audit (homework)
├── templates/passport_template.jpg
├── portraits/                      200 synthetic faces
├── signs/                          4 signature PNGs with transparency
├── generated_passports/            5 images + metadata.jsonl (main task)
├── generated_passports_100/        100 images + metadata.jsonl (homework)
└── experiments/                    broken coordinate: broken_… / fixed_…
```

## 2. Template

- File: `templates/passport_template.jpg`, 608 × 759 px, spread layout (issue page + photo page).
- Source: *[state where the template came from — link / "provided by the instructor"]*.
- The template is cleared: no personal data, all fields are blank, the photo window is white. The stamp on the upper page is part of the blank image; the generator adds no security features.

## 3. How the coordinates were measured

1. `python coordinate_finder.py templates/passport_template.jpg` — click on the start of every field; points are printed and saved to `coordinates.txt`.
2. Clicks were refined with a brightness profile: the blank lines are pixel rows darker than the background. Baselines found: y = 106, 138, 170, 200, 336 (upper page) and 492, 525, 557, 589, 620, 653, 684, 716 (lower page). Text is placed 14 px above its line (Pillow coordinates are the top-left corner of the text).
3. The photo window was detected as the white area: x 38…193, y 518…716.

| Item | x | y | Note |
|---|---|---|---|
| police_department | 122 | 92 | after "Passport issued by" |
| oblast | 50 | 124 | 2nd line of the issuing authority |
| given_date | 110 | 186 | "Date of issue" |
| dep_number | 335 | 186 | "Department code" |
| surname | 255 | 478 | "Surname" |
| name | 235 | 543 | "Name" |
| patronymic | 255 | 575 | "Patronymic" |
| gender | 228 | 606 | "Sex" |
| birth_date | 345 | 606 | "Date of birth" |
| city | 255 | 639 | "Place of birth", line 1 |
| district | 205 | 670 | line 2 |
| region | 205 | 702 | line 3 |
| photo window (top-left) | 38 | 518 | |
| photo window (bottom-right) | 193 | 716 | width = 155, height = 198 |
| rotated number position 1 | 574 | 105 | right edge of the upper page |
| rotated number position 2 | 574 | 482 | right edge of the lower page |
| sign position 1 | 375 | 205 | official's signature next to the stamp |
| sign position 2 | 400 | 281 | above the "Personal signature" line |

Signature box: 110 × 52 px.

## 4. Generated fields and how they are produced

- **Full name, sex** — Faker `ru_RU`: male or female first name, surname and patronymic depending on sex.
- **Date of birth** — age 18 to 90. **Date of issue** — not before the holder turned 14, not older than 30 years, not after today.
- **REGION_PACKS** — consistent geography: issuing authority + oblast + region + city + district always come from the same pack (Irkutsk Oblast, Moscow Oblast, Novosibirsk Oblast, Khabarovsk Krai); the district always belongs to the chosen city.
- **Department code** — `RR-NNN`, where RR is the pack's region code (38, 50, 54, 27).
- **Series and number** — `OO YY NNNNNN`: OO is the OKATO region code (25, 46, 50, 08), YY is the last two digits of the year of issue, the number is random.
- **Portrait** — taken in order from `portraits/`, center-cropped to the photo window aspect ratio (no stretching).
- **Signatures** — 2 random files from `signs/`; transparent margins are cropped and the signature is fitted into the box keeping its aspect ratio.
- **Number along the page edge** — rotated −90°, drawn digit by digit with gaps, twice (once per page).
- **Watermark** "SYNTHETIC TRAINING SAMPLE - NOT A REAL DOCUMENT" — on every image.

Every image gets a line in `metadata.jsonl`: all fields, portrait source, signatures used, seed, `synthetic: true`.

## 5. Seed

`--seed 20260001` (default value). *[The task requires the student ID: `python passport_generator.py --n 5 --seed <ID>`. If you rerun with another seed, update this value.]*

## 6. Verification

- The 5 images in `generated_passports/` were checked manually: text sits on its lines, the portrait is inside the window and not stretched, both signatures are in their areas, the number runs along the edge and does not cover text, the red watermark is visible.
- Metadata check: record `passport_00001` — ЛУКИН, 04.03.1957, number 08 26 615375 — matches the image.
- `python audit_metadata.py generated_passports_100` → 100 records, 100 images, no errors.

**Broken coordinate experiment.** `PHOTO_POSITION` was changed from (38, 518) to (238, 518). The portrait covered the "Name", "Patronymic", "Sex" and "Place of birth" fields, and the photo window stayed empty (`experiments/broken_passport_00001.jpg`). After restoring (38, 518) the defect disappeared (`experiments/fixed_passport_00001.jpg`). Conclusion: one wrong coordinate breaks both its own element and the neighbouring fields, and the error is only visible on visual inspection.

## 7. Problems and how they were solved

| Problem | Solution |
|---|---|
| The original code had coordinates for a different template, and the guide's table requires oblast / city / district / region fields that the code did not have | Fields and consistent REGION_PACKS were added, CONFIG was filled in from scratch |
| The original rotation drew digits in a column and rotated the whole column, so the number came out horizontal | Digits are drawn in a row and the row is rotated −90°, giving a vertical strip along the edge |
| Square signature PNGs were squashed by `resize` to 150×90, and transparent margins wasted space | Crop by the alpha channel + `thumbnail` keeping the aspect ratio |
| The owner's signature touched the "Personal signature" label | Box reduced to 110×52 |
| The photo window aspect (155×198) differs from the source images (819×1024) | Center crop in `fit_portrait`, the face is not distorted |
| Long values (e.g. "ГОР. КОМСОМОЛЬСК-НА-АМУРЕ") | `fit_font` shrinks the font to fit `FIELD_MAX_WIDTH` |
| **Not solved:** sex in the data is not linked to the portrait (portraits have no sex labels) | Portraits need sex labels so the portrait can be chosen by sex |

## 8. Ethics statement

- All faces are synthetic; no photos of real people are used.
- All personal data is fictional (Faker); any match with real people is coincidental.
- The template is cleared and has no real security features.
- Every image carries the watermark "SYNTHETIC TRAINING SAMPLE - NOT A REAL DOCUMENT"; removing it is forbidden. Metadata contains `synthetic: true`.
- The dataset is intended only for training and testing document recognition systems.

## 9. What to improve next

- **Defect augmentations** (proposal for the next stage):
  1. **Blur**: Gaussian / motion blur (`ImageFilter.GaussianBlur(r)`, r = 0.5–2.5) simulates defocus and shake of a phone photo, `defect_type: "blur"`.
  2. **Glare**: a semi-transparent white ellipse with a radial gradient placed at a random position (often over fields), `defect_type: "glare"`.
- Perspective and rotation: a photo of the document on a table, `ImagePerspective` / OpenCV `warpPerspective`.
- Noise, JPEG artifacts, uneven lighting, shadows.
- **YOLO labels**: save the bbox of every field, the photo and the signatures (`draw.textbbox`) in YOLO format `class cx cy w h` to train a field detector.
- Link sex to the portrait; vary fonts and ink colour.
