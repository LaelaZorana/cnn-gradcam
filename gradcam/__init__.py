"""Grad-CAM on a CNN, with a real transfer-learning fine-tune (ants vs. bees).

Public surface:
    GradCAM, overlay_heatmap          (gradcam.gradcam)
    preprocess, denormalize, ...      (gradcam.model)
"""
from __future__ import annotations

from gradcam.gradcam import GradCAM, overlay_heatmap
from gradcam.model import (
    FINETUNE_CLASSES,
    build_resnet18,
    denormalize,
    imagenet_labels,
    load_finetuned,
    load_pretrained,
    preprocess,
    target_layer,
    topk,
)

__all__ = [
    "GradCAM",
    "overlay_heatmap",
    "preprocess",
    "denormalize",
    "build_resnet18",
    "load_pretrained",
    "load_finetuned",
    "target_layer",
    "imagenet_labels",
    "topk",
    "FINETUNE_CLASSES",
]
