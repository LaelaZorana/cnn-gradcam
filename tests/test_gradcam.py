"""
Tests for the Grad-CAM implementation.

The "prove it works" element here is a LOCALIZATION test: we build a tiny CNN whose
output deterministically depends on one corner of the feature map, run Grad-CAM, and
assert the resulting heatmap actually peaks in that corner. A broken chain rule (wrong
gradients) or a broken weighting would scatter the heat elsewhere and the test fails.
We also check the heatmap invariants (shape, [0,1] range), that hooks are removed after
every call (no leaks), and that target-class selection changes the explanation.

These run with no network access and no torchvision (a small hand-built CNN), so the
core algorithm is verified in isolation. A separate test exercises the real ResNet path
only if torchvision is importable.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from gradcam.gradcam import GradCAM, overlay_heatmap


class CornerNet(nn.Module):
    """A tiny CNN with a known answer.

    The conv layer is hand-set so that channel 0 responds to bright pixels. After the
    conv there is a (named) target layer, then global pooling -> 2 logits. Class 1's
    weight reads only the pooled channel-0 activation, so the gradient of class 1's
    score w.r.t. the conv output is concentrated on channel 0 wherever it fired -> the
    Grad-CAM heatmap should be brightest where the input was bright.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.features.weight.zero_()
            self.features.weight[0, 0, 1, 1] = 1.0   # channel 0 = identity (bright -> high)
            self.features.weight[1, 0, 1, 1] = 0.0   # channel 1 = dead
        self.target = nn.ReLU()                      # the layer we hook
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, 2)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))  # class1 <- chan0
            self.fc.bias.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.target(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _bright_corner_image(size: int = 8) -> torch.Tensor:
    """A (1,1,size,size) image that is bright in the top-left quadrant, dark elsewhere."""
    img = torch.zeros(1, 1, size, size)
    img[..., : size // 2, : size // 2] = 1.0
    return img


def test_heatmap_invariants():
    net = CornerNet().eval()
    cam_fn = GradCAM(net, net.target)
    img = _bright_corner_image()
    cam, cls, probs = cam_fn(img, class_idx=1)

    assert cam.shape == (8, 8)                       # upsampled to input size
    assert cam.dtype == np.float32
    assert cam.min() >= 0.0 and cam.max() <= 1.0     # normalized to [0,1]
    assert abs(probs.sum() - 1.0) < 1e-5             # probs are a distribution
    assert cls == 1


def test_localization_peaks_on_evidence():
    """The core correctness check: heat concentrates where the class evidence is."""
    net = CornerNet().eval()
    cam_fn = GradCAM(net, net.target)
    img = _bright_corner_image()
    cam, _, _ = cam_fn(img, class_idx=1)

    tl = cam[:4, :4].mean()                          # bright quadrant
    rest = (cam.sum() - cam[:4, :4].sum()) / (cam.size - 16)
    assert tl > rest * 3, f"heat not localized: tl={tl:.3f} rest={rest:.3f}"
    assert int(np.unravel_index(cam.argmax(), cam.shape)[0]) < 4  # peak in top half


def test_hooks_are_removed_after_call():
    """No hook leakage: repeated calls must not accumulate handles on the layer."""
    net = CornerNet().eval()
    cam_fn = GradCAM(net, net.target)
    img = _bright_corner_image()
    for _ in range(3):
        cam_fn(img, class_idx=1)
    # _forward_hooks / _backward_hooks are the registries torch keeps per module.
    assert len(net.target._forward_hooks) == 0
    assert len(net.target._backward_hooks) == 0


def test_default_class_is_argmax():
    net = CornerNet().eval()
    cam_fn = GradCAM(net, net.target)
    img = _bright_corner_image()        # bright -> class 1 should win
    _, cls, probs = cam_fn(img, class_idx=None)
    assert cls == int(np.argmax(probs))


def test_rejects_bad_input_shape():
    net = CornerNet().eval()
    cam_fn = GradCAM(net, net.target)
    with pytest.raises(ValueError):
        cam_fn(torch.zeros(3, 1, 8, 8))     # batch != 1


def test_overlay_shape_and_dtype():
    img = (np.random.default_rng(0).random((8, 8, 3)) * 255).astype(np.uint8)
    cam = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    out = overlay_heatmap(img, cam, alpha=0.5)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8


def test_flat_map_is_zero():
    """A model whose target gradient is zero everywhere yields an all-zero (safe) map."""
    net = CornerNet().eval()
    cam_fn = GradCAM(net, net.target)
    img = torch.zeros(1, 1, 8, 8)        # no bright pixels -> chan0 dead -> flat
    cam, _, _ = cam_fn(img, class_idx=1)
    assert np.allclose(cam, 0.0)


# --- Optional: exercise the real ResNet/torchvision path if it is installed. ---

torchvision = pytest.importorskip("torchvision")


def test_finetuned_clears_accuracy_bar():
    """Prove the shipped fine-tune actually works: held-out accuracy must clear a bar.

    Skips if the weights or the val data are not present locally, so the suite stays fast
    and offline by default. When both are present (e.g. after `python -m gradcam.train`),
    this guards against shipping a regressed or corrupted checkpoint.
    """
    from gradcam.model import WEIGHTS_PATH
    from gradcam.train import DATA_DIR

    if not WEIGHTS_PATH.exists():
        pytest.skip("fine-tuned weights not present")
    if not (DATA_DIR / "val").exists():
        pytest.skip("validation data not downloaded")

    from gradcam.evaluate import evaluate_val

    acc, per_class, wrong = evaluate_val(verbose=False)
    assert acc >= 0.85, f"held-out accuracy regressed: {acc:.3f}"
    assert all(c >= 0.80 for c in per_class), f"a class regressed: {per_class}"


def test_resnet_gradcam_smoke():
    from gradcam.model import build_resnet18, preprocess, target_layer

    net = build_resnet18(num_classes=1000, pretrained=False).eval()
    cam_fn = GradCAM(net, target_layer(net))
    img = (np.random.default_rng(1).random((224, 224, 3)) * 255).astype(np.uint8)
    cam, cls, probs = cam_fn(preprocess(img))
    assert cam.shape == (224, 224)
    assert 0 <= cls < 1000
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert abs(float(probs.sum()) - 1.0) < 1e-4
