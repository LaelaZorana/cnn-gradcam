"""
Grad-CAM: gradient-weighted class activation mapping.

The idea, in plain terms: a convolutional network's last conv layer still holds a
spatial grid of feature maps (e.g. 7x7 for ResNet on 224x224 input). Each map fires
for some visual pattern. To see WHERE the network looked when it predicted class c,
Grad-CAM asks: how much does the score for class c change if each feature map gets a
little stronger? That sensitivity is the gradient of the class score with respect to
each feature map. Average each map's gradient into a single weight, take the weighted
sum of the maps, keep the positive part (ReLU), and you get a coarse heatmap over the
image: bright where the evidence for class c lives.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization" (ICCV 2017).

This implementation registers a forward hook (to capture the conv activations) and a
full backward hook (to capture the gradients flowing back into them), runs one forward
and one backward pass, then computes the map. Hooks are always removed afterward, even
on error, so repeated calls never leak handles.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Compute Grad-CAM heatmaps for a CNN against a chosen target conv layer.

    Args:
        model: a torch CNN in eval mode.
        target_layer: the module whose activations/gradients to hook. For ResNet this
            is typically ``model.layer4[-1]`` (the last residual block).
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

    def _save_activation(self, module, inp, out) -> None:
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out) -> None:
        # grad_out[0] is the gradient of the loss w.r.t. this layer's output.
        self._gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None):
        """Run Grad-CAM for one image.

        Args:
            input_tensor: shape (1, C, H, W), already normalized for the model.
            class_idx: which class to explain. If None, uses the model's top prediction.

        Returns:
            (cam, class_idx, probs) where ``cam`` is an (H, W) float32 numpy array in
            [0, 1] upsampled to the input's spatial size, ``class_idx`` is the explained
            class, and ``probs`` is the (num_classes,) softmax probability vector.
        """
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError("input_tensor must have shape (1, C, H, W)")

        self.model.eval()
        self._activations = None
        self._gradients = None

        h1 = self.target_layer.register_forward_hook(self._save_activation)
        h2 = self.target_layer.register_full_backward_hook(self._save_gradient)
        try:
            logits = self.model(input_tensor)            # (1, num_classes)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())

            self.model.zero_grad(set_to_none=True)
            score = logits[0, class_idx]
            score.backward()

            if self._activations is None or self._gradients is None:
                raise RuntimeError("hooks did not fire; check the target layer")

            acts = self._activations[0]                  # (K, h, w)
            grads = self._gradients[0]                   # (K, h, w)
            weights = grads.mean(dim=(1, 2))             # (K,)  global-average-pooled grads
            cam = (weights[:, None, None] * acts).sum(dim=0)   # (h, w)
            cam = F.relu(cam)                            # keep evidence FOR the class
        finally:
            h1.remove()
            h2.remove()

        # Upsample the coarse map to the input resolution and scale to [0, 1].
        cam = cam[None, None]                            # (1, 1, h, w)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear",
                            align_corners=False)[0, 0]
        cam = cam.cpu().numpy().astype(np.float32)
        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)                     # flat map -> no signal
        return cam, int(class_idx), probs


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a [0,1] heatmap over an RGB uint8 image using a jet-style colormap.

    Args:
        image_rgb: (H, W, 3) uint8.
        cam: (H, W) float in [0, 1], same spatial size as the image.
        alpha: heatmap opacity in [0, 1].

    Returns:
        (H, W, 3) uint8 overlay.
    """
    from matplotlib import colormaps  # lazy: only needed when rendering

    heat = colormaps["jet"](cam)[..., :3]               # (H, W, 3) float in [0,1]
    heat = (heat * 255).astype(np.float32)
    base = image_rgb.astype(np.float32)
    out = (1 - alpha) * base + alpha * heat
    return np.clip(out, 0, 255).astype(np.uint8)
