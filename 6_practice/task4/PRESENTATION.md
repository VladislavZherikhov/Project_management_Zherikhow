# 5-Minute Presentation — Quality Gate Model Selection

Slide plan with speaker notes and timings (total ≈ 5 min), followed by prepared answers to likely questions. All figures are in `task4/outputs/` and `task3/outputs/`.

---

## Slide 1 — Problem and constraints (0:00–0:40)

**On the slide**
- Telegram bot: photo → **quality gate** → document detection → OCR
- Gate = binary classifier: good / bad (blur, glare, skew, occlusion, exposure)
- Constraints: runs on every photo, CPU server, small dataset (1800 synthetic images)
- Acceptance criteria fixed in advance: F1 ≥ 0.92, recall(bad) ≥ 0.93, ≤ 50 ms, ≤ 25 MB

**Say:** "The gate is the first step of the bot, so every millisecond is added to every answer. A missed bad photo is the expensive error, because OCR then returns wrong data. That's why our main metric is recall on the bad class, and we fixed numeric thresholds before training anything."

---

## Slide 2 — Data and fair protocol (0:40–1:20)

**On the slide**
- 1800 images: INN / SNILS / passport × good / bad, 900 / 900
- Split **70/15/15 by source document**, stratified by class: 1260 / 270 / 270, no leakage
- Same test set, same metric code, same latency protocol for every model
- Fixed shortcut: only GOOD images were rotated → random 90° rotation of both classes in training
- Seed 24500003

**Say:** "All good and bad variants of one document are in the same split, so the test documents are never seen in training. We also found that in our dataset only good images were rotated, so a model could learn 'rotated means good'. We removed that shortcut with rotation augmentation of both classes."

---

## Slide 3 — Six models, one table (1:20–2:30)

**On the slide:** the comparison table + `accuracy_vs_speed.png`

| Model | F1 (bad) | FN / FP | Latency | Size |
|---|---|---|---|---|
| MobileNetV3-Small (frozen) | 0.977 | 5 / 1 | 10.4 ms | 3.9 MB |
| MobileNetV2 (frozen) | 0.985 | 2 / 2 | 20.7 ms | 9.3 MB |
| EfficientNet-B0 (frozen) | 0.993 | 2 / 0 | 31.0 ms | 16.2 MB |
| **YOLOv8n-cls** | **0.996** | **1 / 0** | **9.5 ms** | 5.6 MB |
| YOLOv8s-cls | 1.000 | 0 / 0 | 18.5 ms | 19.5 MB |
| MobileNetV3-Small FT (bonus) | 0.996 | 1 / 0 | 10.7 ms | 3.9 MB |

**Say:** "All six models pass the acceptance criteria. The frozen MobileNets are fast but miss more bad photos. EfficientNet-B0 is the most accurate frozen model but three times slower. The YOLO models, fine-tuned end to end, are the most accurate: nano makes one error out of 270 at under 10 milliseconds."

---

## Slide 4 — Key insight: the regime matters more than the architecture (2:30–3:20)

**On the slide**
- MobileNetV3-Small, frozen → fine-tune last 3 blocks, lr 1e-4
- Accuracy 0.978 → **0.996**, recall 0.963 → **0.993**, FN 5 → **1**, FP 1 → **0**
- Same latency, same size

**Say:** "Why was YOLO so much better than our baseline? To test it, I fine-tuned the last blocks of MobileNetV3-Small with a ten times lower learning rate. It jumped to exactly YOLOv8n's quality. So the gap came from adapting the features to document defects, not from the architecture."

---

## Slide 5 — Error analysis (3:20–4:10)

**On the slide:** `failure_gallery.png` (top row) + defect table

| Hardest defects | Mean recall | Misses |
|---|---|---|
| occlusion | 0.967 | 5 |
| exposure | 0.976 | 4 |
| glare | 0.988 | 2 |

**Say:** "Only 11 of 270 test images are wrong for any model. The hardest defect here is occlusion, not glare as expected. Our glare is very strong, except on the white INN form, where white-on-white glare almost disappears; both glare misses are INN. The image missed by three models has a black block over an empty corner, so it's actually readable: label noise. These findings go back into the data generator."

---

## Slide 6 — Decision, alternatives, risks (4:10–5:00)

**On the slide**
- **Decision: YOLOv8n-cls**: 9.5 ms, 5.6 MB, recall 0.993, 0 false alarms
- Pre-registered rule: fastest group within 5 ms → highest recall → tie with MobileNetV3-Small FT → simplest production path (one-stage training, one-command ONNX export, same toolchain as a future YOLO detector)
- Fallback: MobileNetV3-Small FT (equal quality, BSD licence)
- Risks: synthetic-to-real gap (validate on real photos), small test set, AGPL licence

**Say:** "I recommend YOLOv8n-cls. It's in the fastest group and has the highest recall there. It ties with fine-tuned MobileNetV3-Small, and I break the tie by deployment simplicity. YOLOv8s is perfect on test, but its advantage is one image at double the latency. The main risk is the gap between synthetic and real photos, so the next step is a real-photo test set."

---

## Prepared answers to likely questions

**Where does the model sit in the pipeline, and why does latency matter?**
First, before document detection and OCR; it runs on 100 % of photos. The user waits for the reply in the chat, and a bad photo should be rejected instantly instead of after a slow, failing OCR attempt.

**In the frozen regime, what does the model learn and what not?**
Only the head (74–164 k weights): how to combine fixed ImageNet features into good/bad. The convolution filters do not change, so it cannot learn new defect-specific features. That is why the frozen models missed occlusion and exposure, and fine-tuning fixed it.

**How does the synthetic-to-real gap affect the thresholds?**
Real photos typically cost 3–6 pp of accuracy, so the synthetic thresholds were set with a margin (0.92 → ≥ 0.86 on real photos in the worst case). Our models reach 0.99+, so even a 6 pp drop keeps them above the real-world minimum, but this must be confirmed on real photos.

**Isn't comparing fine-tuned YOLO with frozen MobileNet unfair?**
Yes, the regimes differ. That's exactly why the bonus experiment fine-tunes MobileNetV3-Small. Under the same regime, both models reach the same quality.

**Why is YOLOv8n fast despite more FLOPs than MobileNetV3-Small?**
Dense 3×3 convolutions use CPU vector units efficiently. Depthwise convolutions, squeeze-and-excitation and hardswish are memory-bound and less efficient, so FLOPs do not predict CPU latency well.

**How is data leakage prevented?**
The split unit is the source document; validation is used only for early stopping and best-epoch selection; the threshold is fixed at 0.5; the test set is evaluated once.

**How do you read the Ultralytics training output?**
`train/loss` and `val/loss` are cross-entropy; `metrics/accuracy_top1` is validation accuracy (top-5 is always 1.0 with 2 classes); `best.pt` is the epoch with the best validation top-1; training stops after 8 epochs without improvement.
