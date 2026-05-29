---
title: CNN + Grad-CAM
emoji: 🔥
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# CNN + Grad-CAM

A convolutional network does not have to be a black box. **Grad-CAM** uses the gradients
flowing into the last convolutional layer to draw a heatmap of *where the model looked*
when it made a prediction.

Two modes, both running the real package code:

- **Mode A — any image:** a ResNet18 pretrained on ImageNet (1000 classes). Upload any
  photo and see the top-5 prediction plus the Grad-CAM heatmap.
- **Mode B — ants vs. bees:** a genuine transfer-learning fine-tune. I started from the
  ImageNet ResNet18, swapped in a 2-class head, and fine-tuned on the hymenoptera dataset
  (~240 training images). The trained weights ship with the Space.

Grad-CAM is the bridge between **building** a model and **evaluating** it: it does not just
predict, it shows the evidence, so you can check the network keyed on the object and not the
background. The implementation's localization is verified in the test suite.

**Source & full docs:** https://github.com/LaelaZorana/cnn-gradcam
