"""
Evaluate the fine-tuned ants-vs-bees model on the held-out validation set.

This is the "I verified it" half of the project. It does not just report one accuracy
number; it breaks accuracy down per class and surfaces the model's most CONFIDENTLY WRONG
predictions, because a model that is wrong loudly is more dangerous than one that is wrong
quietly. Those are exactly the cases Grad-CAM is useful for inspecting.

Run:  python -m gradcam.evaluate
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from gradcam.model import FINETUNE_CLASSES, load_finetuned
from gradcam.train import DATA_DIR, ensure_data


def _val_items() -> List[Tuple[Path, int]]:
    """List (image_path, true_label) for the validation split, label by folder order."""
    ensure_data()
    items: List[Tuple[Path, int]] = []
    for label, cls in enumerate(FINETUNE_CLASSES):           # ['ants','bees'] -> 0,1
        for p in sorted((DATA_DIR / "val" / cls).glob("*.jpg")):
            items.append((p, label))
    return items


def evaluate_val(verbose: bool = True):
    """Return (accuracy, per_class_acc, confidently_wrong) over the held-out val set."""
    from PIL import Image
    from gradcam.model import preprocess

    net = load_finetuned()
    items = _val_items()

    correct = 0
    per_class_correct = [0, 0]
    per_class_total = [0, 0]
    wrong: List[Tuple[str, str, str, float]] = []   # (file, true, pred, confidence)

    for path, true in items:
        img = np.asarray(Image.open(path).convert("RGB"))
        with torch.no_grad():
            logits = net(preprocess(img))
            probs = F.softmax(logits, dim=1)[0].numpy()
        pred = int(probs.argmax())
        conf = float(probs[pred])
        per_class_total[true] += 1
        if pred == true:
            correct += 1
            per_class_correct[true] += 1
        else:
            wrong.append((path.name, FINETUNE_CLASSES[true], FINETUNE_CLASSES[pred], conf))

    acc = correct / max(len(items), 1)
    per_class = [per_class_correct[i] / max(per_class_total[i], 1) for i in range(2)]
    wrong.sort(key=lambda r: r[3], reverse=True)    # most confident mistakes first

    if verbose:
        print(f"Held-out validation set: {len(items)} images")
        print(f"Overall accuracy: {acc:.3f}  ({correct}/{len(items)})")
        for i, cls in enumerate(FINETUNE_CLASSES):
            print(f"  {cls:5s}: {per_class[i]:.3f}  ({per_class_correct[i]}/{per_class_total[i]})")
        print(f"\nConfidently wrong (top {min(5, len(wrong))} of {len(wrong)} misses):")
        for name, true, pred, conf in wrong[:5]:
            print(f"  {name:32s} true={true:5s} pred={pred:5s} conf={conf:.2f}")
    return acc, per_class, wrong


if __name__ == "__main__":
    evaluate_val(verbose=True)
