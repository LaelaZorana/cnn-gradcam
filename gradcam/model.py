"""
Model loading and image preprocessing for the two demo modes.

Mode A: a ResNet18 pretrained on ImageNet (1000 classes). Works on any image.
Mode B: the same backbone with a fresh 2-class head, fine-tuned on ants vs. bees
        (the classic torchvision transfer-learning dataset). Trained weights ship in
        weights/ so the demo classifies without retraining.

torchvision is imported lazily inside functions so the package can be imported (and the
app can build-test) on machines where torchvision is not installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# ImageNet normalization, shared by both modes (Mode B fine-tuned on the same stats).
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224

FINETUNE_CLASSES: List[str] = ["ants", "bees"]
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "antbee_resnet18.pt"


def preprocess(image_rgb: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 RGB -> normalized (1, 3, 224, 224) float tensor.

    Center-crops to a square then resizes, matching standard ImageNet eval transforms.
    """
    from PIL import Image  # lazy

    pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    w, h = pil.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    pil = pil.crop((left, top, left + side, top + side)).resize(
        (INPUT_SIZE, INPUT_SIZE), Image.BILINEAR
    )
    arr = np.asarray(pil).astype(np.float32) / 255.0          # (224, 224, 3)
    arr = (arr - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    return tensor


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Invert preprocess() for one (1,3,H,W) tensor -> (H, W, 3) uint8 RGB (for overlays)."""
    arr = tensor[0].cpu().numpy().transpose(1, 2, 0)
    arr = arr * np.array(STD, np.float32) + np.array(MEAN, np.float32)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def build_resnet18(num_classes: int = 1000, pretrained: bool = False) -> torch.nn.Module:
    """Construct a ResNet18, optionally with a resized final layer for fine-tuning."""
    from torchvision import models  # lazy

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    if num_classes != 1000:
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    net.eval()
    return net


def load_pretrained() -> torch.nn.Module:
    """Mode A: ResNet18 with ImageNet weights (1000-way classifier)."""
    return build_resnet18(num_classes=1000, pretrained=True)


def load_finetuned() -> torch.nn.Module:
    """Mode B: ResNet18 with the shipped ants-vs-bees head loaded from weights/."""
    net = build_resnet18(num_classes=len(FINETUNE_CLASSES), pretrained=False)
    if WEIGHTS_PATH.exists():
        state = torch.load(WEIGHTS_PATH, map_location="cpu")
        net.load_state_dict(state)
    net.eval()
    return net


def target_layer(net: torch.nn.Module) -> torch.nn.Module:
    """The last residual block of a ResNet, the conventional Grad-CAM target."""
    return net.layer4[-1]


def imagenet_labels() -> List[str]:
    """Human-readable ImageNet class names (from torchvision's bundled metadata)."""
    from torchvision import models  # lazy

    return list(models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"])


def topk(probs: np.ndarray, labels: List[str], k: int = 5) -> List[Tuple[str, float]]:
    """Return the k highest-probability (label, prob) pairs."""
    idx = np.argsort(probs)[::-1][:k]
    return [(labels[i], float(probs[i])) for i in idx]
