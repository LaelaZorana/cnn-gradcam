"""
Transfer-learning fine-tune: ImageNet ResNet18 -> ants vs. bees.

This is a genuine fine-tune, not a toy. It downloads the hymenoptera dataset (the
classic torchvision transfer-learning set, ~240 train / ~150 val images), swaps the
1000-way ImageNet head for a fresh 2-way head, and trains. With the convolutional
backbone kept (only lightly updated) the model reaches ~90%+ validation accuracy in a
handful of CPU epochs. The trained head + backbone are saved to weights/ so the demo
and tests run inference without retraining.

Run:  python -m gradcam.train --epochs 8
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn

from gradcam.model import WEIGHTS_PATH, build_resnet18, INPUT_SIZE

DATA_URL = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "hymenoptera_data"


def ensure_data() -> Path:
    """Download + unzip the hymenoptera dataset if it is not already present."""
    import zipfile

    if DATA_DIR.exists():
        return DATA_DIR
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR.parent / "hymenoptera_data.zip"
    print(f"Downloading {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR.parent)
    zip_path.unlink(missing_ok=True)
    return DATA_DIR


def make_loaders(batch_size: int = 16):
    from torchvision import datasets, transforms

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    root = ensure_data()
    train_ds = datasets.ImageFolder(root / "train", train_tf)
    val_ds = datasets.ImageFolder(root / "val", val_tf)
    # ImageFolder sorts classes alphabetically: ['ants', 'bees'] -> matches FINETUNE_CLASSES.
    print("Classes:", train_ds.classes)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl


def evaluate(net, loader, device) -> float:
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = net(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += y.numel()
    return correct / max(total, 1)


def train(epochs: int = 8, lr: float = 1e-3, batch_size: int = 16, seed: int = 0) -> float:
    torch.manual_seed(seed)
    device = torch.device("cpu")

    # Start from ImageNet weights, swap to a 2-way head: classic transfer learning.
    from torchvision import models

    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net = net.to(device)

    train_dl, val_dl = make_loaders(batch_size)
    criterion = nn.CrossEntropyLoss()
    # Fine-tune the whole network at a small LR; give the fresh head a higher LR.
    optim = torch.optim.AdamW([
        {"params": [p for n, p in net.named_parameters() if not n.startswith("fc.")], "lr": lr * 0.1},
        {"params": net.fc.parameters(), "lr": lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_acc = 0.0
    for ep in range(1, epochs + 1):
        net.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
        sched.step()
        acc = evaluate(net, val_dl, device)
        print(f"epoch {ep:2d}  train_loss {running / len(train_dl.dataset):.4f}  val_acc {acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), WEIGHTS_PATH)

    print(f"\nBest val accuracy: {best_acc:.4f}  ->  saved {WEIGHTS_PATH}")
    return best_acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed)
