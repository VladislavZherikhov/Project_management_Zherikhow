"""
Shared torchvision models and training loop for Practice 6.

The model and training loop are the same as in the Task 2 notebook, so every
torchvision model (MobileNetV3-Small, MobileNetV2, EfficientNet-B0) is
trained under one protocol. Task 4 also uses this module to reload the
Task 2 weights.
"""

import time

import numpy as np
import torch
from torch import nn
from torchvision import models

import quality_common as qc

# name: (constructor with ImageNet weights, channels of .features output)
BACKBONES = {
    "mobilenet_v3_small": (lambda: models.mobilenet_v3_small(weights="IMAGENET1K_V1"), 576),
    "mobilenet_v2":       (lambda: models.mobilenet_v2(weights="IMAGENET1K_V1"), 1280),
    "efficientnet_b0":    (lambda: models.efficientnet_b0(weights="IMAGENET1K_V1"), 1280),
}

DEFAULT_HP = {
    "seed": qc.SEED,
    "image_size": 224,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "max_epochs": 30,
    "patience": 5,
    "hidden_units": 128,
    "dropout": 0.3,
    "threshold": 0.5,
}


class QualityNet(nn.Module):
    """Pretrained backbone + GAP -> Linear(C,128) -> ReLU -> Dropout -> Linear(128,1)."""

    def __init__(self, arch, hidden_units=128, dropout=0.3):
        super().__init__()
        build, channels = BACKBONES[arch]
        self.backbone = build().features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, hidden_units), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, 1),
        )
        self.trainable_blocks = 0
        self.freeze_backbone()

    def freeze_backbone(self, trainable_blocks=0):
        """Freeze the backbone, leaving the last `trainable_blocks` blocks trainable."""
        self.trainable_blocks = trainable_blocks
        blocks = list(self.backbone.children())
        for i, block in enumerate(blocks):
            trainable = i >= len(blocks) - trainable_blocks
            for p in block.parameters():
                p.requires_grad = trainable

    def forward(self, x):
        return self.head(self.backbone(x))

    def train(self, mode=True):
        super().train(mode)
        # BatchNorm statistics stay as learned on ImageNet (small dataset)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def predict(model, x_uint8, device="cpu", batch_size=64):
    """P(bad) for uint8 images [N,3,H,W]."""
    model.eval()
    probs = []
    with torch.inference_mode():
        for i in range(0, len(x_uint8), batch_size):
            xb = qc.imagenet_normalize(x_uint8[i:i + batch_size].to(device))
            probs.append(torch.sigmoid(model(xb)).squeeze(1).cpu())
    return torch.cat(probs).numpy()


def bce(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_model(model, data, hp, device="cpu", log=print, tag=""):
    """
    Adam + BCE, random 90-degree rotations of both classes, early stopping on
    validation loss, best weights restored. Only parameters with
    requires_grad=True are optimised.
    """
    qc.set_seed(hp["seed"])
    gen = torch.Generator().manual_seed(hp["seed"])
    model = model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hp["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()
    (x_tr, y_tr), (x_va, y_va) = data["train"], data["validation"]

    history, best, best_state, bad_epochs = [], float("inf"), None, 0
    t_start = time.time()
    for epoch in range(1, hp["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(x_tr), generator=gen)
        run_loss, run_correct = 0.0, 0
        for i in range(0, len(order), hp["batch_size"]):
            idx = order[i:i + hp["batch_size"]]
            xb = qc.imagenet_normalize(qc.random_rot90(x_tr[idx], gen).to(device))
            yb = y_tr[idx].to(device)
            logits = model(xb).squeeze(1)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(idx)
            run_correct += ((logits > 0).float() == yb).sum().item()

        p_va = predict(model, x_va, device)
        rec = {
            "epoch": epoch,
            "train_loss": run_loss / len(x_tr), "train_acc": run_correct / len(x_tr),
            "val_loss": bce(y_va.numpy(), p_va),
            "val_acc": float(((p_va >= 0.5) == y_va.numpy()).mean()),
        }
        history.append(rec)
        log(f"[{tag}] epoch {epoch:2d}  train loss {rec['train_loss']:.4f} acc {rec['train_acc']:.3f} | "
            f"val loss {rec['val_loss']:.4f} acc {rec['val_acc']:.3f}")

        if rec["val_loss"] < best - 1e-4:
            best, bad_epochs = rec["val_loss"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= hp["patience"]:
                log(f"early stopping: no val_loss improvement for {hp['patience']} epochs")
                break

    train_time = time.time() - t_start
    model.load_state_dict(best_state)
    best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
    log(f"best epoch {best_epoch}, training time {train_time:.0f} s")
    return model, history, train_time, best_epoch
