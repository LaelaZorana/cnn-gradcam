"""
Gradio demo for CNN + Grad-CAM.

Two modes, both running the real package code (gradcam/):
  A) Pretrained ResNet18 (ImageNet, 1000 classes) — upload ANY photo, get the top-5
     predictions and a Grad-CAM heatmap showing where the network looked.
  B) A real transfer-learning fine-tune (ImageNet ResNet18 -> ants vs. bees), with the
     trained weights shipped in weights/. Same Grad-CAM, on a model I actually trained.

Grad-CAM is the bridge between building and evaluating: it does not just predict, it
shows the evidence, so you can check whether the model looked at the right thing.

Run locally:   pip install -r requirements.txt && python app.py
On Hugging Face Spaces this file is the entry point (app_file: app.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import gradio as gr

from gradcam import (
    FINETUNE_CLASSES,
    GradCAM,
    imagenet_labels,
    load_finetuned,
    load_pretrained,
    overlay_heatmap,
    preprocess,
    target_layer,
    topk,
)

ACCENT = "#ea580c"  # orange

# Load models once at startup (CPU). Heavy, so do it lazily-but-cached.
_PRE = None
_PRE_CAM = None
_FT = None
_FT_CAM = None
_LABELS = None


def _pretrained():
    global _PRE, _PRE_CAM, _LABELS
    if _PRE is None:
        _PRE = load_pretrained()
        _PRE_CAM = GradCAM(_PRE, target_layer(_PRE))
        _LABELS = imagenet_labels()
    return _PRE, _PRE_CAM, _LABELS


def _finetuned():
    global _FT, _FT_CAM
    if _FT is None:
        _FT = load_finetuned()
        _FT_CAM = GradCAM(_FT, target_layer(_FT))
    return _FT, _FT_CAM


def explain_imagenet(image: np.ndarray):
    """Mode A: any image -> top-5 ImageNet labels + Grad-CAM overlay."""
    if image is None:
        return None, {}
    net, cam_fn, labels = _pretrained()
    x = preprocess(image)
    cam, cls, probs = cam_fn(x)
    # Resize the original to the model's 224 square so the overlay lines up.
    from PIL import Image

    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    w, h = pil.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    base = np.asarray(pil.crop((left, top, left + side, top + side)).resize((224, 224)))
    overlay = overlay_heatmap(base, cam, alpha=0.5)
    return overlay, {lab: p for lab, p in topk(probs, labels, k=5)}


def explain_antbee(image: np.ndarray):
    """Mode B: fine-tuned ants-vs-bees model -> prediction + Grad-CAM overlay."""
    if image is None:
        return None, {}
    net, cam_fn = _finetuned()
    x = preprocess(image)
    cam, cls, probs = cam_fn(x)
    from PIL import Image

    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    w, h = pil.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    base = np.asarray(pil.crop((left, top, left + side, top + side)).resize((224, 224)))
    overlay = overlay_heatmap(base, cam, alpha=0.5)
    return overlay, {FINETUNE_CLASSES[i]: float(probs[i]) for i in range(len(FINETUNE_CLASSES))}


CSS = """
:root { --accent: %s; }
.gradio-container { max-width: 1080px !important; }
#hero { background: linear-gradient(135deg, var(--accent), #1c1917);
        color:#fff; border-radius:18px; padding:26px 30px; margin-bottom:6px; }
#hero h1 { margin:0 0 8px 0; font-size:1.7rem; font-weight:800; letter-spacing:-.01em; }
#hero p { margin:0; opacity:.93; font-size:1.0rem; line-height:1.5; max-width:780px; }
#hero .pill { display:inline-block; background:rgba(255,255,255,.16); border-radius:999px;
        padding:3px 11px; font-size:.74rem; font-weight:700; margin-bottom:12px; letter-spacing:.04em; }
.footer { margin-top:20px; padding-top:14px; border-top:1px solid rgba(128,128,128,.25);
        font-size:.88rem; text-align:center; opacity:.92; }
.footer a { text-decoration:none; font-weight:700; color:var(--accent); }
""" % ACCENT

FOOTER = """
<div class="footer">
🔥 CNN + Grad-CAM (PyTorch, real transfer-learning fine-tune) by <b>Laela Zorana</b> &nbsp;·&nbsp;
<a href="https://github.com/LaelaZorana/cnn-gradcam">Source on GitHub</a> &nbsp;·&nbsp;
see also:
🧠 <a href="https://huggingface.co/spaces/LaelaZ/nn-from-scratch">NN From Scratch</a> ·
🔍 <a href="https://huggingface.co/spaces/LaelaZ/ai-agent-scenario-qc">Scenario QC</a> ·
⚖️ <a href="https://huggingface.co/spaces/LaelaZ/rlhf-pairwise-rater">RLHF Rater</a> ·
📦 <a href="https://huggingface.co/spaces/LaelaZ/scorm-qa-validator">SCORM QA</a>
</div>
"""

_EX_DIR = Path("examples")
ANTBEE_EXAMPLES = sorted(str(p) for p in _EX_DIR.glob("antbee_*.jpg")) if _EX_DIR.exists() else []
IMAGENET_EXAMPLES = sorted(str(p) for p in _EX_DIR.glob("imagenet_*.jpg")) if _EX_DIR.exists() else []

theme = gr.themes.Soft(primary_hue="orange", neutral_hue="stone",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

with gr.Blocks(title="CNN + Grad-CAM", theme=theme, css=CSS) as demo:
    gr.HTML(
        '<div id="hero"><span class="pill">DEEP LEARNING · VISION · INTERPRETABILITY</span>'
        "<h1>🔥 CNN + Grad-CAM</h1>"
        "<p>A convolutional network does not have to be a black box. Grad-CAM uses the "
        "gradients flowing into the last conv layer to draw a heatmap of where the model "
        "looked. Mode A runs a pretrained ResNet18 on any image you upload. Mode B runs a "
        "model I actually fine-tuned (ImageNet to ants vs. bees) on its own data. Both show "
        "the prediction and the evidence, so you can check the model looked at the right "
        "thing.</p></div>"
    )

    with gr.Tab("🖼️ Mode A — any image (ImageNet)"):
        with gr.Row():
            with gr.Column():
                in_a = gr.Image(label="Upload an image", type="numpy", height=300)
                btn_a = gr.Button("Classify + explain", variant="primary")
                if IMAGENET_EXAMPLES:
                    gr.Examples(IMAGENET_EXAMPLES, inputs=in_a, label="Try an example")
            with gr.Column():
                out_a_img = gr.Image(label="Grad-CAM (where the model looked)", height=300)
                out_a_lab = gr.Label(num_top_classes=5, label="Top-5 ImageNet prediction")
        btn_a.click(explain_imagenet, inputs=in_a, outputs=[out_a_img, out_a_lab])

    with gr.Tab("🐜🐝 Mode B — ants vs. bees (fine-tuned)"):
        gr.Markdown("*This model started as ImageNet ResNet18, then I fine-tuned it on the "
                    "hymenoptera dataset (~240 training images). The shipped weights do the "
                    "classifying. Grad-CAM shows whether it keys on the insect, not the background.*")
        with gr.Row():
            with gr.Column():
                in_b = gr.Image(label="Upload an ant or a bee", type="numpy", height=300)
                btn_b = gr.Button("Classify + explain", variant="primary")
                if ANTBEE_EXAMPLES:
                    gr.Examples(ANTBEE_EXAMPLES, inputs=in_b, label="Try an example")
            with gr.Column():
                out_b_img = gr.Image(label="Grad-CAM (where the model looked)", height=300)
                out_b_lab = gr.Label(num_top_classes=2, label="Prediction")
        btn_b.click(explain_antbee, inputs=in_b, outputs=[out_b_img, out_b_lab])

    gr.HTML(FOOTER)
    gr.Markdown("*Runs the actual package (`gradcam/`) — the same Grad-CAM code whose "
                "localization is verified in `tests/`. Architecture: ResNet18, Grad-CAM on "
                "`layer4[-1]`.*")


if __name__ == "__main__":
    demo.launch()
