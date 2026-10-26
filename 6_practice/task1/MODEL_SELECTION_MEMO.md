# Task 1 — Model Selection Memo: Document Photo Quality Gate

| | |
|---|---|
| Author | Vladislav Zherikhov |
| Date | 26.10.2026 (written **before** any training experiment) |
| Seed for all experiments | `24500003` |
| Dataset | `5_practice/data/quality_dataset` — 1800 synthetic images (INN, SNILS, passport), 900 good / 900 bad, 640×640 |
| Hardware | MSI GL75 Leopard: Intel Core i7-10750H (6C/12T), NVIDIA GTX 1650 Ti 4 GB, 16 GB RAM |
| Framework | PyTorch 2 + torchvision (MobileNetV2, EfficientNet, ResNet), Ultralytics (YOLOv8-cls) |

## 1. Problem and where the model sits in the pipeline

The project is a Telegram bot that reads Russian documents (passport, INN, SNILS) from phone photos:

```
user sends photo ──► [1] QUALITY GATE ──good──► [2] document type / field detection ──► [3] OCR ──► [4] validation ──► answer
                         (this model)
                              │ bad
                              └──► instant reply: "The photo is blurred / has glare, please retake it"
```

The quality model is the **first gate**. It runs on every incoming photo, before the expensive OCR stages, and answers one binary question: is the photo **good** (readable) or **bad** (blur, glare, strong skew, occlusion, bad exposure)?

This position defines the requirements:
- **Speed.** It runs on 100% of requests, so its latency is added to every answer. If a photo is bad, the user should get "retake the photo" within a second, not after a slow OCR attempt that fails anyway. A slow gate makes the whole bot feel slow and wastes OCR compute on unusable photos.
- **Size.** The gate will run on a CPU server next to the OCR models. A small model loads fast, uses little memory and can serve several users in parallel without a GPU.
- **Error cost.**
  - A **missed bad photo** (bad → predicted good) goes to OCR and produces wrong or empty fields. That is the worst outcome, because the user may receive incorrect document data.
  - A **false rejection** (good → predicted bad) only costs the user one retake.
  - Therefore **recall on the bad class** matters more than precision. In all metrics below, **bad is the positive class**.
- **Data.** Only 1800 synthetic images, 300 source documents. This is far too little to train a CNN from scratch, so **transfer learning** from ImageNet is required, and models with fewer parameters are less likely to overfit.

## 2. Candidate architectures

### 2.1 Short technical description

**MobileNetV2** [1]. Built from *inverted residual blocks with linear bottlenecks*:
- each block expands the channels with a 1×1 convolution, filters them with a cheap 3×3 **depthwise** convolution, and projects back to a narrow bottleneck with a linear (no ReLU) 1×1 convolution;
- a residual connection links the narrow layers;
- depthwise-separable convolutions cut computation about 8–9× compared with standard 3×3 convolutions.

It was designed for mobile and embedded inference. The output is a 1280-channel feature map, followed by global average pooling and a classifier.

**EfficientNet-B0** [3]:
- a baseline found by neural architecture search, built from MBConv blocks (the MobileNetV2 inverted residual) plus **squeeze-and-excitation** channel attention and the SiLU activation;
- the paper's main idea is **compound scaling**: depth, width and resolution are scaled together with one coefficient (B0 → B7);
- B0 gives the best ImageNet accuracy among the small models here, but SE blocks and SiLU make it slower on CPU than its FLOPs suggest.

**ResNet-50** [2]:
- a 50-layer network of *bottleneck residual blocks* (1×1 → 3×3 → 1×1 convolutions with an identity shortcut);
- residual learning solved the degradation problem of very deep networks;
- it is the standard, well-understood reference backbone with strong transfer-learning results, but it has about 10× more parameters and FLOPs than MobileNetV2.

**YOLOv8-cls (n / s)** [4]:
- the classification variant of Ultralytics YOLOv8: the YOLOv8 CSP backbone (C2f blocks: convolutions with cross-stage partial connections) followed by a classification head (1×1 conv → global pooling → linear → softmax);
- the nano (n) and small (s) variants differ in width and depth;
- it comes with a complete training/validation/export toolkit (`yolo classify train`) that uses a simple `train/val/test/<class>/` folder layout and has built-in augmentation.

### 2.2 Comparison table

Columns **A** come from the official sources (ImageNet-1k, 1000-class head). Columns **B** were measured on this laptop with a 1-output binary head: `task1/benchmark_candidates.py`, CPU fp32, batch 1, 224×224, 10 warm-up runs, then the mean of 100 runs. Raw output is in `task1/benchmark_results.json`.

| Model | Input | A: Params (M) | A: GFLOPs | A: ImageNet top-1 | A: Weights file | B: Params, binary head (M) | B: Size fp32 (MB) | B: CPU latency (ms) |
|---|---|---|---|---|---|---|---|---|
| MobileNetV2 [1][5] | 224 | 3.5 | 0.30 | 71.9 % | 13.6 MB | 2.23 | 8.7 | **20.3 ± 2.0** |
| MobileNetV3-Large | 224 | 5.4 | 0.22 | 74.0 % | 21.4 MB | 4.10 | 16.4 | **15.2 ± 1.5** |
| MobileNetV3-Small | 224 | 2.5 | 0.06 | 67.4 % | 10.1 MB | 1.25 | 5.0 | **7.4 ± 0.8** |
| EfficientNet-B0 [3][5] | 224 | 5.3 | 0.39 | 77.7 % | 20.5 MB | 4.01 | 15.6 | 29.8 ± 3.3 |
| ResNet-50 [2][5] | 224 | 25.6 | 4.09 | 76.1 % | 97.8 MB | 23.51 | 90.0 | 71.0 ± 5.1 |
| YOLOv8n-cls [4] | 224 | 2.7 | 4.3* | 69.0 % | ~5.5 MB | 1.44 | 5.6 | **9.7 ± 1.1** |
| YOLOv8s-cls [4] | 224 | 6.4 | 13.5* | 73.8 % | ~12.5 MB | 5.08 | 19.5 | 19.0 ± 1.9 |

\* Ultralytics reports FLOPs for its classification models at a larger evaluation size, so these numbers are not directly comparable with the torchvision FLOPs. The measured latency (column B) is the fair comparison.

### 2.3 Pros and cons for this project

| Model | Pros | Cons |
|---|---|---|
| MobileNetV2 | Small (8.7 MB), 20 ms on CPU. Standard, simple transfer-learning head. Few parameters, so low overfitting risk on 1800 images. Course baseline, so results are comparable across students | Lowest ImageNet accuracy of the torchvision models. Its features may be less sensitive to subtle defects (weak glare) |
| EfficientNet-B0 | Best ImageNet accuracy per parameter (77.7 %). Still small (15.6 MB). SE attention can help with local defects such as glare spots | ~50 % slower than MobileNetV2 on CPU. More sensitive to hyperparameters during fine-tuning |
| ResNet-50 | Strongest classic backbone, very well studied, robust features | 90 MB and 71 ms on CPU, which fails the size and latency budget below. 23.5 M parameters on 1800 images is a high overfitting risk |
| YOLOv8n-cls | Fastest (9.7 ms) and smallest (5.6 MB). Full training toolkit with augmentation and export (ONNX/TFLite) | Its own preprocessing and training loop (softmax over 2 classes, no ImageNet mean/std), so the setup differs from the torchvision models. Lowest ImageNet accuracy |
| YOLOv8s-cls | Better accuracy than n at MobileNetV2 speed | 2× slower and 3.5× larger than n |

## 3. Transfer learning and the two training regimes

**Transfer learning.** A network pretrained on ImageNet (1.28 M photos, 1000 classes [7]) has already learned general visual features:
- early layers: edges, gradients, textures;
- middle layers: shapes, text-like patterns;
- late layers: object parts.

Yosinski et al. [6] showed that early-layer features are general and transfer well, while the last layers are task-specific. For our defects (blur removes edges, glare saturates regions, occlusion adds dark blocks) these generic features are exactly what is needed. We keep the pretrained backbone, remove the ImageNet classifier, and add our own head:

```
backbone (pretrained) → GlobalAveragePooling → Dense(128, ReLU) → Dropout(0.3) → Dense(1) → sigmoid  →  P(bad)
```

**Regime 1 — frozen base (feature extraction).**
- All backbone weights are frozen (`requires_grad = False`, BatchNorm in eval mode); only the new head is trained with a normal learning rate (e.g. Adam 1e-3).
- **What the model learns:** only the head weights, i.e. how to combine the fixed 1280 ImageNet features into a good/bad decision.
- **What it does NOT learn:** the convolution filters themselves. It cannot create new features specific to glare or synthetic document texture; it can only reuse what ImageNet taught it.
- Advantages: fast (a few minutes on the GTX 1650 Ti), very few trainable parameters (~164 k for MobileNetV2), so overfitting on 1800 images is unlikely.
- This is the Task 2 baseline.

**Regime 2 — full fine-tuning (or fine-tuning the last blocks).**
- After the head has converged, the last N blocks (or the whole backbone) are unfrozen and trained together with the head at a **10–100× lower learning rate** (e.g. 1e-5).
- **What the model learns:** the features themselves adapt to the task, for example becoming more sensitive to bright saturated spots (glare) or to loss of fine text strokes (blur).
- Risks: on a small synthetic dataset the network can overfit and memorise synthetic artefacts (template, fonts, watermark), and a high learning rate destroys the pretrained features ("catastrophic forgetting").
- Mitigations: low LR, unfreezing only the last blocks, early stopping on validation loss, keeping BatchNorm statistics frozen.
- The head must be trained first; otherwise the random head sends large gradients into the pretrained layers [8].

## 4. Decision for the experiments

| Role | Model | Why |
|---|---|---|
| **Baseline and primary candidate (Task 2)** | **MobileNetV2**, frozen base | Meets the speed and size budget with a large margin (20 ms, 8.7 MB). Fewest trainable parameters for a small synthetic dataset. Designed for exactly this kind of lightweight first-stage inference |
| Speed challenger (YOLOv8 task) | **YOLOv8n-cls** and **YOLOv8s-cls** | n is 2× faster than MobileNetV2; if its accuracy is close, it is the better gate. s checks whether extra capacity buys accuracy |
| Accuracy challenger (third model, Task 3) | **EfficientNet-B0** | Highest ImageNet accuracy within the latency budget. Tests whether better features improve the hardest defect (glare) |
| Reference only, not trained | ResNet-50 | 71 ms and 90 MB already break the latency and size criteria below, so it cannot be deployed as the first gate whatever its accuracy |

The final model is chosen **after** the experiments, as the fastest model that passes all acceptance criteria in section 5. If no model passes, the one with the highest F1 that passes the cost criteria is taken to fine-tuning (regime 2).

## 5. Success criteria and acceptance thresholds (defined before experiments)

All quality metrics are computed **once, on the test split only**, with **bad as the positive class**. Validation is used for early stopping and for choosing the decision threshold (default 0.5); test data is never used for training, model selection or threshold tuning.

| # | Criterion | Threshold | Reason |
|---|---|---|---|
| Q1 | Accuracy (test) | **≥ 0.92** | Overall correctness, kept with a margin for the synthetic-to-real gap (see below) |
| Q2 | F1, bad class (test) | **≥ 0.92** | Balanced quality measure; accuracy alone is not accepted |
| Q3 | Recall, bad class (test) | **≥ 0.93** | A missed bad photo goes to OCR and gives wrong data, which is the most expensive error |
| Q4 | Precision, bad class (test) | **≥ 0.88** | At most ~1 in 8 rejections is unnecessary, so users are not asked to retake good photos too often |
| Q5 | Recall per defect type (blur, glare, skew, occlusion, exposure) | **each ≥ 0.85** | The gate must not be blind to one defect; glare is expected to be the hardest |
| Q6 | Confusion matrix | must be reported | Required to see which error type dominates |
| C1 | CPU latency per image (i7-10750H, batch 1, fp32, 10 warm-up + mean of 100) | **≤ 50 ms** | The gate should use ≤ 5 % of a ~1 s bot response budget and leave headroom for a weaker server CPU |
| C2 | Model file size (fp32) | **≤ 25 MB** | Loads in under a second, fits next to the OCR models in server memory, and can be exported to mobile later |
| C3 | Training time (GTX 1650 Ti) | **≤ 30 min** | Lets us repeat experiments with different seeds and regimes in class |
| R1 | Reproducibility | seed = student ID, all hyperparameters logged | Course rule: missing seed or hyperparameters caps the task at Satisfactory |

**Synthetic-to-real gap.** Models trained on synthetic data typically lose **3–6 percentage points** of accuracy on real phone photos (real lighting, paper texture, camera noise and compression are never fully simulated). The real-world minimum we need is about **0.86–0.88**. That is why the synthetic test thresholds are set **higher**: 0.92 − 0.06 = 0.86 in the worst case.
- A model that only just reaches 0.88 on synthetic test would probably fall to ~0.82–0.85 on real photos, which is not acceptable.
- Before production, the thresholds must be re-checked on a small labelled set of **real** photos.

**Decision rule.** A model is *accepted* if it passes Q1–Q5 and C1–C2. Among accepted models, the fastest wins; if the latency difference is below 5 ms, the one with the higher bad-class recall wins.

## 6. Experiment protocol (applies to every model, so the comparison is fair)

- **Same data.** Same images and the **same 70 / 15 / 15 split by source document** (stratified by document class), so no source document appears in two splits. The current dataset split is 73/12/15, so it will be rebuilt to exactly 70/15/15 before Task 2.
- **Input preprocessing.** RGB, resized to 224×224.
  - torchvision models: ImageNet mean/std normalisation;
  - YOLOv8: its own normalisation (0–1 scaling), as the family requires.
- **Same evaluation script.** Accuracy, precision, recall, F1, confusion matrix, per-defect recall.
- **Same cost protocol.** CPU latency and size measured as in section 2.2.
- **Logging.** Every run logs seed, epochs, batch size, optimizer, learning rate, image size, training time and final metrics in an experiment log.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Synthetic-to-real gap is larger than 6 % | Collect 50–100 real photos (own documents only, never shared) for a final sanity check; add realistic augmentations (perspective, noise, JPEG) |
| Shortcut learning: in our dataset, BAD images are never rotated 90/180/270°, so the model can learn "rotated = good" | Fix in the augmentation pipeline before Task 2 (apply orientation augmentations to BAD as well); check the per-augmentation error rate |
| Weak glare/exposure on the white INN form makes some BAD labels actually readable (label noise) | Error analysis in Task 3; adjust defect strength |
| Overfitting during full fine-tuning | Frozen-base baseline first, low LR, early stopping on validation |

## 8. Final decision for comparison 

For the **first experiment**, the **MobileNetV3-Small**, **EfficientNet-B0**, and **YOLOv8n-cls** models will be officially selected. The primary objective of this initial phase is to thoroughly evaluate and compare their **speed metrics** (latency and throughput), **computational complexity** (parameter count and FLOPs), and **accuracy parameters** on the specific dataset. Based on the quantitative results of this benchmark, **one final model** that demonstrates the optimal balance of efficiency and predictive performance will be selected for deployment.
| Model | Primary Focus | Speed & Efficiency Metrics | Accuracy & Quality Metrics | Selection Criterion |
| :--- | :--- | :--- | :--- | :--- |
| **MobileNetV3-Small** | Ultra-low resource usage and high-speed CPU inference. | • Latency (ms)<br>• Throughput (FPS)<br>• Model Size (MB) | • Top-1 / Top-5 Accuracy<br>• Precision<br>• Recall / F1-Score | **One best model** will be selected based on the highest performance-to-speed ratio after testing on the validation dataset. |
| **EfficientNet-B0** | Optimal balance between feature extraction capability and parameters. | • Parameter Count (M)<br>• FLOPs (B)<br>• Inference Time (ms) | • Top-1 / Top-5 Accuracy<br>• Precision<br>• Recall / F1-Score |
| **YOLOv8n-cls** | Native real-time classification within the YOLO ecosystem. | • Latency (ms)<br>• Throughput (FPS)<br>• Export Flexibility (ONNX/TensorRT) | • Top-1 / Top-5 Accuracy<br>• Precision<br>• Recall / F1-Score |

## 9. Sources

1. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018. arXiv:1801.04381.
2. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016. arXiv:1512.03385.
3. M. Tan, Q. V. Le. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019. arXiv:1905.11946.
4. Ultralytics. *YOLOv8 — Image Classification* (official documentation and model table). https://docs.ultralytics.com/tasks/classify/
5. PyTorch. *torchvision.models — Classification* (official model weights, ImageNet accuracy, parameters, GFLOPs, file sizes). https://pytorch.org/vision/stable/models.html
6. J. Yosinski, J. Clune, Y. Bengio, H. Lipson. *How transferable are features in deep neural networks?* NeurIPS 2014. arXiv:1411.1792.
7. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei. *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR 2009.
8. F. Chollet. *Transfer learning & fine-tuning.* Keras official guide. https://keras.io/guides/transfer_learning/
