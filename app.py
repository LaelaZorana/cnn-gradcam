"""
Gradio demo for CNN + Grad-CAM.

Two modes, both running the real package code (gradcam/):
  A) Pretrained ResNet18 (ImageNet, 1000 classes) -- upload ANY photo, get the top-5
     predictions and a Grad-CAM heatmap showing where the network looked.
  B) A real transfer-learning fine-tune (ImageNet ResNet18 -> ants vs. bees), with the
     trained weights shipped in weights/. Same Grad-CAM, on a model I actually trained.

The UI is bespoke, not the stock label-bars: the prediction renders as a hero card naming
the top class with its confidence, the full distribution renders as an animated custom bar
chart, and the Grad-CAM overlay sits beside the original as a clean side-by-side. All of the
prediction HTML is drawn by the Python function into a single gr.HTML panel, so the design is
fully controlled rather than themed defaults. Inference runs the real package (gradcam/).

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


def _render(pairs) -> str:
    """Render an ordered list of (label, prob) as a hero card + animated bar chart (HTML)."""
    ordered = sorted(pairs, key=lambda kv: kv[1], reverse=True)
    top_label, top_p = ordered[0]
    conf = round(top_p * 100)

    hero = f"""
    <div class="gc-hero">
      <div class="gc-hero-icon">&#128065;</div>
      <div class="gc-hero-body">
        <div class="gc-hero-kicker">Top prediction</div>
        <div class="gc-hero-label">{top_label}</div>
        <div class="gc-hero-sub">{conf}% confident &middot; the heatmap shows the evidence</div>
      </div>
      <div class="gc-hero-ring" style="--p:{top_p:.4f}">
        <span>{conf}<small>%</small></span>
      </div>
    </div>
    """

    rows = []
    for label, p in ordered:
        pct = round(p * 100)
        rows.append(f"""
        <div class="gc-row">
          <div class="gc-row-name">{label}</div>
          <div class="gc-track">
            <div class="gc-fill" style="width:{p*100:.2f}%"></div>
          </div>
          <div class="gc-pct">{pct}%</div>
        </div>
        """)
    chart = f'<div class="gc-chart">{"".join(rows)}</div>'
    return f'<div class="gc-result">{hero}{chart}</div>'


def _crop_square(image: np.ndarray):
    """Center-crop the original to the model's 224 square so the overlay lines up."""
    from PIL import Image

    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    w, h = pil.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return np.asarray(pil.crop((left, top, left + side, top + side)).resize((224, 224)))


EMPTY_A = """
<div class="gc-empty">
  <div class="gc-empty-icon">&#128247;</div>
  <div class="gc-empty-text">Upload any photo and I'll name the top-5 and show you where I looked.</div>
  <div class="gc-empty-sub">Pretrained ResNet18, 1000 ImageNet classes.</div>
</div>
"""

EMPTY_B = """
<div class="gc-empty">
  <div class="gc-empty-icon">&#128027;</div>
  <div class="gc-empty-text">Upload an ant or a bee and I'll classify it and show the evidence.</div>
  <div class="gc-empty-sub">A ResNet18 I fine-tuned on the hymenoptera dataset.</div>
</div>
"""


def explain_imagenet(image: np.ndarray):
    """Mode A: any image -> top-5 ImageNet labels + Grad-CAM overlay. Returns (image, html)."""
    if image is None:
        return None, EMPTY_A
    net, cam_fn, labels = _pretrained()
    x = preprocess(image)
    cam, cls, probs = cam_fn(x)
    base = _crop_square(image)
    overlay = overlay_heatmap(base, cam, alpha=0.5)
    pairs = [(lab, float(p)) for lab, p in topk(probs, labels, k=5)]
    return overlay, _render(pairs)


def explain_antbee(image: np.ndarray):
    """Mode B: fine-tuned ants-vs-bees model -> prediction + Grad-CAM overlay. Returns (image, html)."""
    if image is None:
        return None, EMPTY_B
    net, cam_fn = _finetuned()
    x = preprocess(image)
    cam, cls, probs = cam_fn(x)
    base = _crop_square(image)
    overlay = overlay_heatmap(base, cam, alpha=0.5)
    pairs = [(FINETUNE_CLASSES[i], float(probs[i])) for i in range(len(FINETUNE_CLASSES))]
    return overlay, _render(pairs)


CSS = """
:root {
  --gc-accent:%s; --gc-bg1:#fff7ed; --gc-bg2:#fef2f2; --gc-ink:#1c1917; --gc-muted:#78716c;
  --gc-card:#ffffff; --gc-line:rgba(28,25,23,.08);
  --gc-font:'Plus Jakarta Sans','Inter',system-ui,sans-serif;
}

/* Light lock: HF Spaces default to dark mode, but this UI is designed light.
   Override Gradio's dark theme variables so it renders light everywhere. */
:root, .dark, gradio-app.dark {
  color-scheme: light !important;
  --body-background-fill:#ffffff !important;
  --background-fill-primary:#ffffff !important;
  --background-fill-secondary:#f6f6fb !important;
  --block-background-fill:#ffffff !important;
  --block-label-background-fill:#ffffff !important;
  --input-background-fill:#ffffff !important;
  --border-color-primary:rgba(20,16,40,.12) !important;
  --body-text-color:#16131f !important;
  --body-text-color-subdued:#6b6880 !important;
  --block-title-text-color:#16131f !important;
  --block-info-text-color:#6b6880 !important;
}
html, body, gradio-app, .dark { background:#ffffff !important; }

.gradio-container { max-width: 1080px !important; background:
  radial-gradient(1200px 500px at 15%% -10%%, var(--gc-bg1), transparent 60%%),
  radial-gradient(1000px 500px at 110%% 10%%, var(--gc-bg2), transparent 55%%) !important; }
.gradio-container, .gradio-container * { font-family: var(--gc-font); }

/* Header */
#gc-head { text-align:center; padding: 18px 8px 6px; }
#gc-head .gc-pill { display:inline-block; background:var(--gc-ink); color:#fff; border-radius:999px;
  padding:5px 13px; font-size:.7rem; font-weight:700; letter-spacing:.08em; margin-bottom:14px; }
#gc-head h1 { margin:0; font-size:2.05rem; font-weight:800; letter-spacing:-.02em; color:var(--gc-ink);
  background:linear-gradient(90deg,#ea580c,#f59e0b,#dc2626); -webkit-background-clip:text;
  background-clip:text; -webkit-text-fill-color:transparent; }
#gc-head p { margin:10px auto 0; max-width:640px; color:var(--gc-muted); font-size:1.02rem; line-height:1.55; }

/* Buttons */
.gc-go { border-radius:14px !important; font-weight:800 !important; font-size:1rem !important;
  background:linear-gradient(135deg,#ea580c,#f59e0b) !important; border:none !important; color:#fff !important;
  box-shadow:0 10px 26px rgba(234,88,12,.32) !important; transition:transform .12s ease, box-shadow .12s ease !important; }
.gc-go:hover { transform:translateY(-1px); box-shadow:0 14px 32px rgba(234,88,12,.42) !important; }

/* Mode note */
.gc-note { background:var(--gc-card); border:1px solid var(--gc-line); border-radius:16px;
  padding:14px 18px; box-shadow:0 8px 24px rgba(28,25,23,.05); color:var(--gc-muted);
  font-size:.95rem; line-height:1.5; }
.gc-note b { color:var(--gc-ink); }

/* Side-by-side image cards */
.gc-shot { background:var(--gc-card); border:1px solid var(--gc-line); border-radius:20px;
  padding:14px 14px 6px; box-shadow:0 10px 30px rgba(28,25,23,.06); }
.gc-shot .gc-cap { text-align:center; font-weight:700; color:var(--gc-ink); font-size:.92rem; margin-bottom:4px; }
.gc-shot .gc-cap small { display:block; font-weight:600; color:var(--gc-muted); font-size:.78rem; margin-top:2px; }
.gc-shot .gradio-image, .gc-shot .image-container { border-radius:14px !important; overflow:hidden; }

/* Results panel */
.gc-result { animation: gc-fade .35s ease both; }
@keyframes gc-fade { from{opacity:0; transform:translateY(8px)} to{opacity:1; transform:none} }

.gc-hero { display:flex; align-items:center; gap:18px; padding:22px 24px; border-radius:20px;
  background:var(--gc-card); border:1px solid var(--gc-line); position:relative; overflow:hidden;
  box-shadow:0 18px 44px rgba(234,88,12,.18); }
.gc-hero::before { content:""; position:absolute; inset:0; opacity:.10;
  background:radial-gradient(420px 160px at 8%% 0%%, var(--gc-accent), transparent 70%%); }
.gc-hero-icon { font-size:2.8rem; line-height:1; filter:drop-shadow(0 6px 12px rgba(234,88,12,.35)); }
.gc-hero-body { flex:1; }
.gc-hero-kicker { font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--gc-accent); }
.gc-hero-label { font-size:1.7rem; font-weight:800; text-transform:capitalize; color:var(--gc-ink); letter-spacing:-.01em; line-height:1.15; }
.gc-hero-sub { color:var(--gc-muted); font-size:.95rem; margin-top:3px; }
.gc-hero-ring { width:64px; height:64px; border-radius:50%%; display:grid; place-items:center; flex-shrink:0;
  background:conic-gradient(var(--gc-accent) calc(var(--p)*360deg), color-mix(in srgb,var(--gc-accent) 16%%, #fff) 0); }
.gc-hero-ring span { width:50px; height:50px; border-radius:50%%; background:var(--gc-card); display:grid; place-items:center;
  font-weight:800; color:var(--gc-ink); font-size:1.02rem; }
.gc-hero-ring small { font-size:.62rem; font-weight:700; color:var(--gc-muted); }

.gc-chart { margin-top:14px; padding:18px 22px; border-radius:18px; background:var(--gc-card);
  border:1px solid var(--gc-line); box-shadow:0 10px 30px rgba(28,25,23,.05); }
.gc-row { display:flex; align-items:center; gap:14px; padding:7px 0; }
.gc-row-name { width:160px; text-transform:capitalize; font-weight:700; color:var(--gc-ink); font-size:.9rem;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.gc-track { flex:1; height:13px; border-radius:999px; background:#fdebd9; overflow:hidden; }
.gc-fill { height:100%%; border-radius:999px; transform-origin:left;
  background:linear-gradient(90deg,#ea580c,#f59e0b);
  animation: gc-grow .65s cubic-bezier(.2,.8,.2,1) both; }
@keyframes gc-grow { from{transform:scaleX(0)} to{transform:scaleX(1)} }
.gc-pct { width:42px; text-align:right; font-variant-numeric:tabular-nums; font-weight:700; color:var(--gc-muted); font-size:.9rem; }

/* Empty state */
.gc-empty { text-align:center; padding:42px 20px; border-radius:20px; background:var(--gc-card);
  border:1px dashed var(--gc-line); }
.gc-empty-icon { font-size:2.6rem; }
.gc-empty-text { margin-top:10px; font-weight:700; color:var(--gc-ink); font-size:1.05rem; }
.gc-empty-sub { margin-top:4px; color:var(--gc-muted); font-size:.92rem; }

/* Footer */
.gc-footer { margin-top:22px; padding-top:16px; border-top:1px solid var(--gc-line);
  text-align:center; font-size:.88rem; color:var(--gc-muted); line-height:1.9; }
.gc-footer a { text-decoration:none; font-weight:700; color:var(--gc-accent); }
.gc-meta { text-align:center; color:var(--gc-muted); font-size:.82rem; margin-top:10px; }
""" % ACCENT

FOOTER = """
<div class="gc-footer">
&#128293; CNN + Grad-CAM (PyTorch, real transfer-learning fine-tune) by <b>Laela Zorana</b><br>
<a href="https://github.com/LaelaZorana/cnn-gradcam">Source on GitHub</a> &middot;
<a href="https://huggingface.co/spaces/LaelaZ/nn-from-scratch">NN From Scratch</a> &middot;
<a href="https://huggingface.co/spaces/LaelaZ/ai-agent-scenario-qc">Scenario QC</a> &middot;
<a href="https://huggingface.co/spaces/LaelaZ/rlhf-pairwise-rater">RLHF Rater</a> &middot;
<a href="https://huggingface.co/spaces/LaelaZ/scorm-qa-validator">SCORM QA</a>
</div>
"""

_EX_DIR = Path("examples")
ANTBEE_EXAMPLES = sorted(str(p) for p in _EX_DIR.glob("antbee_*.jpg")) if _EX_DIR.exists() else []
IMAGENET_EXAMPLES = sorted(str(p) for p in _EX_DIR.glob("imagenet_*.jpg")) if _EX_DIR.exists() else []

theme = gr.themes.Soft(
    primary_hue="orange", neutral_hue="stone",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), gr.themes.GoogleFont("Inter"),
          "system-ui", "sans-serif"],
)

with gr.Blocks(title="CNN + Grad-CAM", theme=theme, css=CSS) as demo:
    gr.HTML(
        '<div id="gc-head"><span class="gc-pill">DEEP LEARNING &middot; VISION &middot; INTERPRETABILITY</span>'
        "<h1>CNN + Grad-CAM</h1>"
        "<p>A convolutional network does not have to be a black box. Grad-CAM uses the "
        "gradients flowing into the last conv layer to draw a heatmap of where the model "
        "looked. Mode A runs a pretrained ResNet18 on any image you upload. Mode B runs a "
        "model I actually fine-tuned, from ImageNet to ants vs. bees, on its own data. Both "
        "show the prediction and the evidence, so you can check the model looked at the right "
        "thing.</p></div>"
    )

    with gr.Tab("Mode A -- any image (ImageNet)"):
        with gr.Row():
            with gr.Column():
                in_a = gr.Image(label="Upload an image", type="numpy", height=300)
                btn_a = gr.Button("Classify + explain", elem_classes="gc-go", variant="primary")
                if IMAGENET_EXAMPLES:
                    gr.Examples(IMAGENET_EXAMPLES, inputs=in_a, label="Try an example")
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="gc-cap">What you uploaded'
                                '<small>center-cropped to 224 square</small></div>')
                        view_a = gr.Image(label="", show_label=False, height=240,
                                          elem_classes="gc-shot", interactive=False)
                    with gr.Column():
                        gr.HTML('<div class="gc-cap">Where the model looked'
                                '<small>Grad-CAM overlay</small></div>')
                        out_a_img = gr.Image(label="", show_label=False, height=240,
                                             elem_classes="gc-shot", interactive=False)
                out_a_html = gr.HTML(EMPTY_A)
        btn_a.click(lambda im: im, inputs=in_a, outputs=view_a)
        btn_a.click(explain_imagenet, inputs=in_a, outputs=[out_a_img, out_a_html])

    with gr.Tab("Mode B -- ants vs. bees (fine-tuned)"):
        gr.HTML('<div class="gc-note">This model started as ImageNet ResNet18, then I '
                "fine-tuned it on the hymenoptera dataset (about 240 training images). The "
                "shipped weights do the classifying. Grad-CAM shows whether it keys on the "
                "<b>insect</b>, not the background.</div>")
        with gr.Row():
            with gr.Column():
                in_b = gr.Image(label="Upload an ant or a bee", type="numpy", height=300)
                btn_b = gr.Button("Classify + explain", elem_classes="gc-go", variant="primary")
                if ANTBEE_EXAMPLES:
                    gr.Examples(ANTBEE_EXAMPLES, inputs=in_b, label="Try an example")
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="gc-cap">What you uploaded'
                                '<small>center-cropped to 224 square</small></div>')
                        view_b = gr.Image(label="", show_label=False, height=240,
                                          elem_classes="gc-shot", interactive=False)
                    with gr.Column():
                        gr.HTML('<div class="gc-cap">Where the model looked'
                                '<small>Grad-CAM overlay</small></div>')
                        out_b_img = gr.Image(label="", show_label=False, height=240,
                                             elem_classes="gc-shot", interactive=False)
                out_b_html = gr.HTML(EMPTY_B)
        btn_b.click(lambda im: im, inputs=in_b, outputs=view_b)
        btn_b.click(explain_antbee, inputs=in_b, outputs=[out_b_img, out_b_html])
        with gr.Accordion("How I verified this model (held-out evaluation)", open=False):
            gr.Markdown(
                "I did not just train it, I checked it on the **held-out validation set** "
                "(153 images the model never saw):\n\n"
                "| | accuracy |\n|---|---|\n"
                "| **overall** | **0.928** (142/153) |\n"
                "| ants | 0.914 (64/70) |\n"
                "| bees | 0.940 (78/83) |\n\n"
                "I also look at where it is **confidently wrong** (a loud mistake is worse "
                "than a quiet one): of 11 misses, the worst predicts *bees* on an ant at "
                "100% confidence. Those are exactly the images to open Grad-CAM on, to see "
                "whether the model keyed on the insect or on the background. Reproduce with "
                "`python -m gradcam.evaluate`; the accuracy bar is also a test in the repo."
            )

    gr.HTML(FOOTER)
    gr.HTML('<div class="gc-meta">Runs the actual package (gradcam/), the same Grad-CAM code '
            "whose localization is verified in tests/. Architecture: ResNet18, Grad-CAM on "
            "<code>layer4[-1]</code>.</div>")


if __name__ == "__main__":
    demo.launch()
