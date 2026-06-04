# cnn-gradcam

**🔗 Live demo:** [try it on Hugging Face Spaces](https://huggingface.co/spaces/LaelaZ/cnn-gradcam). Upload an image and see *where* a CNN looked, plus a model I actually fine-tuned.

A small computer-vision project with two halves that tell one story: a CNN that **predicts**,
and **Grad-CAM** that shows you the evidence behind each prediction. I wrote the Grad-CAM from
the algorithm up (forward + backward hooks, gradient-weighted activation maps) rather than
pulling a library, because the point is to understand and verify *why* the model decided what
it decided.

## Two modes

- **Mode A, any image (ImageNet).** A ResNet18 pretrained on ImageNet classifies any photo
  you upload (1000 classes) and Grad-CAM overlays a heatmap of where the evidence was.
- **Mode B, ants vs. bees (real fine-tune).** I took the ImageNet ResNet18, replaced the
  1000-way head with a 2-way head, and fine-tuned on the hymenoptera dataset (~240 training
  images, the classic transfer-learning set). The trained weights are committed, so the demo
  classifies without retraining. Grad-CAM then shows whether the model keys on the insect or
  on the background, which is exactly the kind of check that catches a model "right for the
  wrong reason."

## I built it and I verified it

This is the through-line of my portfolio: build the model *and* the evidence it works.
For Mode B I evaluated the fine-tune on the **held-out validation set** (153 images the
model never trained on):

```
Overall accuracy: 0.928  (142/153)
  ants : 0.914  (64/70)
  bees : 0.940  (78/83)
```

I also surface the model's **confidently wrong** cases, because a model that is wrong
*loudly* is more dangerous than one that is wrong quietly:

```
Confidently wrong (top of 11 misses):
  119785936_dd428e40c3.jpg   true=ants  pred=bees  conf=1.00
  59798110_2b6a3c8031.jpg    true=bees  pred=ants  conf=0.95
  8398478_50ef10c47a.jpg     true=ants  pred=bees  conf=0.90
```

Reproduce with `python -m gradcam.evaluate`. The accuracy bar is also a **test**
(`test_finetuned_clears_accuracy_bar`), so a regressed or corrupted checkpoint fails CI,
not the demo. Grad-CAM is the tool you reach for next: open one of those confident
mistakes and look at *where* the model looked.

## Why this repo is more than "it predicts"

The interesting part is that the Grad-CAM is **verified to localize correctly**.
`tests/test_gradcam.py` builds a tiny CNN whose output depends, by construction, on one
corner of the feature map, runs Grad-CAM, and asserts the heatmap actually peaks in that
corner. A wrong gradient or a wrong weighting would scatter the heat and the test fails. The
suite also checks the heatmap invariants (shape, [0,1] range), that hooks are removed after
every call (no leaks), and the real ResNet path end to end. (This is the same
build-it-and-prove-it discipline I apply to evaluation work.)

```
tests/test_gradcam.py
  test_heatmap_invariants            # shape, dtype, [0,1] range, probs sum to 1
  test_localization_peaks_on_evidence # the core: heat lands where the class evidence is
  test_hooks_are_removed_after_call   # repeated calls do not leak forward/backward hooks
  test_default_class_is_argmax        # explaining "the prediction" picks the top class
  test_rejects_bad_input_shape        # guards the (1, C, H, W) contract
  test_overlay_shape_and_dtype        # colormap overlay returns a valid RGB image
  test_flat_map_is_zero               # no signal -> safe all-zero map, no NaNs
  test_finetuned_clears_accuracy_bar  # shipped fine-tune holds >=85% on held-out val
  test_resnet_gradcam_smoke           # full ResNet18 path (skipped if no torchvision)
```

## How Grad-CAM works (plain version)

A CNN's last conv layer holds a small spatial grid of feature maps (7×7 for ResNet on a
224×224 image). To see where the network looked for class *c*, ask how much *c*'s score would
change if each feature map fired a little stronger, because that sensitivity is the gradient of the
score with respect to each map. Average each map's gradient into one weight, take the weighted
sum of the maps, keep the positive part (ReLU), upsample to the image, and you get a heatmap:
bright where the evidence for *c* lives. (Selvaraju et al., ICCV 2017.)

## Run it

```bash
pip install -r requirements.txt
pytest -q                              # runs the localization check + other tests
python -m gradcam.train --epochs 8     # (optional) re-run the fine-tune; saves weights/
python app.py                          # launches the two-mode Grad-CAM demo locally
```

## Layout

```
gradcam/
  gradcam.py   # GradCAM (hooks + gradient-weighted maps) + overlay_heatmap
  model.py     # ResNet18 loaders (pretrained / fine-tuned), preprocessing, labels
  train.py     # transfer-learning fine-tune: ImageNet -> ants vs. bees
  evaluate.py  # held-out accuracy, per-class, and confidently-wrong cases
tests/         # localization check + invariants + hook-leak + accuracy bar + ResNet smoke
app.py         # Gradio demo (Mode A any image / Mode B fine-tuned)
weights/       # shipped fine-tuned weights (antbee_resnet18.pt)
examples/      # a few sample images for the demo
```

Part of my ML portfolio (build + evaluate). License: MIT.
