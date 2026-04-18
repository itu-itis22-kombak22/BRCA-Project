"""Patch-level tumor probability inference.

Loads a ResNet-18 fine-tuned on PCam (trained in ``colab/train_pcam.ipynb``)
and scores 96×96 RGB patches on the MacBook's MPS backend.

This is a SKELETON. Full loading + test happens once
``models/resnet18_pcam.pth`` is in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


IM_MEAN = [0.485, 0.456, 0.406]
IM_STD = [0.229, 0.224, 0.225]
PCAM_PATCH = 96


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _transform() -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(IM_MEAN, IM_STD),
    ])


def load_model(weights_path: str | Path,
               device: torch.device | None = None) -> nn.Module:
    """Rebuild the timm ResNet-18 (num_classes=1) and load fine-tuned weights."""
    import timm  # local import keeps import-time light when only importing signatures

    device = device or _device()
    model = timm.create_model("resnet18", pretrained=False, num_classes=1)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_patch(model: nn.Module, patch: Image.Image,
                  device: torch.device | None = None) -> float:
    """Return P(tumor) for a single 96×96 RGB patch."""
    device = device or next(model.parameters()).device
    x = _transform()(patch.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x).squeeze(-1)
    return float(torch.sigmoid(logits).cpu().item())


@torch.no_grad()
def predict_batch(model: nn.Module,
                  patches: Sequence[Image.Image] | torch.Tensor,
                  device: torch.device | None = None,
                  batch_size: int = 128) -> np.ndarray:
    """Return an (N,) array of P(tumor) for a sequence of 96×96 RGB patches.

    ``patches`` may be:
      - a sequence of PIL images, or
      - a pre-built ``(N, 3, 96, 96)`` float tensor already normalised.
    """
    device = device or next(model.parameters()).device

    if isinstance(patches, torch.Tensor):
        tensor = patches
    else:
        tf = _transform()
        tensor = torch.stack([tf(p.convert("RGB")) for p in patches])

    probs = np.empty(tensor.shape[0], dtype=np.float32)
    for i in range(0, tensor.shape[0], batch_size):
        batch = tensor[i:i + batch_size].to(device)
        logits = model(batch).squeeze(-1)
        probs[i:i + batch.shape[0]] = torch.sigmoid(logits).float().cpu().numpy()
    return probs


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Quick sanity check of the classifier.")
    p.add_argument("--weights", default="models/resnet18_pcam.pth")
    p.add_argument("--patch", help="Optional path to a 96x96 image for a smoke test.")
    args = p.parse_args()

    dev = _device()
    print(f"[inference] device: {dev}")
    model = load_model(args.weights, device=dev)
    print(f"[inference] loaded weights from {args.weights}")

    if args.patch:
        img = Image.open(args.patch)
        print(f"[inference] P(tumor) = {predict_patch(model, img, dev):.4f}")
