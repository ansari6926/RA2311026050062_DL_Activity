"""Train and evaluate a seedling image classifier."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class SeedlingDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, transform=None):
        self.rows = rows.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row.path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row.label)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        blocks = []
        in_channels = 3
        for out_channels in (32, 64, 128, 256):
            blocks.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_images(data_dir: Path):
    classes = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"No class directories found under {data_dir}")
    paths, labels = [], []
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    for label, class_name in enumerate(classes):
        files = sorted(p for p in (data_dir / class_name).rglob("*") if p.suffix.lower() in valid_suffixes)
        paths.extend(str(p) for p in files)
        labels.extend([label] * len(files))
    if not paths:
        raise ValueError(f"No supported image files found under {data_dir}")
    return pd.DataFrame({"path": paths, "label": labels}), classes


def build_model(name: str, num_classes: int):
    if name == "small-cnn":
        return SmallCNN(num_classes)
    weights = models.ResNet34_Weights.DEFAULT
    model = models.resnet34(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_features, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        nn.Dropout(0.2), nn.Linear(256, num_classes),
    )
    return model


def run_epoch(model, loader, loss_fn, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    losses, predictions, targets = [], [], []
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        predictions.extend(logits.argmax(1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())
    if not targets:
        raise ValueError("No batches were processed; increase --max-train-batches")
    return float(np.mean(losses)), accuracy_score(targets, predictions), f1_score(targets, predictions, average="macro", zero_division=0), predictions, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing one subfolder per species")
    parser.add_argument("--model", choices=("resnet34", "small-cnn"), default="resnet34")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    set_seed(args.seed)
    rows, class_names = collect_images(args.data_dir)
    train_rows, remainder = train_test_split(rows, test_size=0.30, stratify=rows.label, random_state=args.seed)
    val_rows, test_rows = train_test_split(remainder, test_size=0.50, stratify=remainder.label, random_state=args.seed)

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180), transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_args = {"batch_size": args.batch_size, "num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(SeedlingDataset(train_rows, train_transform), shuffle=True, **loader_args)
    val_loader = DataLoader(SeedlingDataset(val_rows, eval_transform), shuffle=False, **loader_args)
    test_loader = DataLoader(SeedlingDataset(test_rows, eval_transform), shuffle=False, **loader_args)

    model = build_model(args.model, len(class_names)).to(device)
    lr = args.learning_rate or (4e-4 if args.model == "resnet34" else 5e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / f"{args.model}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    history, best_f1 = [], -1.0
    best_path = run_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer, args.max_train_batches)
        val_loss, val_acc, val_f1, _, _ = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "train_macro_f1": train_f1,
                        "val_loss": val_loss, "val_accuracy": val_acc, "val_macro_f1": val_f1})
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} macro-F1 {train_f1:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model_state": model.state_dict(), "classes": class_names, "model": args.model, "seed": args.seed}, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    _, _, _, test_predictions, test_targets = run_epoch(model, test_loader, loss_fn, device)
    test_accuracy = accuracy_score(test_targets, test_predictions)
    test_macro_f1 = f1_score(test_targets, test_predictions, average="macro", zero_division=0)
    report = classification_report(test_targets, test_predictions, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0)
    matrix = confusion_matrix(test_targets, test_predictions, labels=list(range(len(class_names))))

    pd.DataFrame(history).to_csv(run_dir / "learning_curves.csv", index=False)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(run_dir / "confusion_matrix.csv")
    with (run_dir / "classification_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    config = {**vars(args), "data_dir": str(args.data_dir), "output_dir": str(args.output_dir),
              "classes": class_names, "counts": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
              "device": str(device), "best_validation_macro_f1": best_f1,
              "test_accuracy": test_accuracy, "test_macro_f1": test_macro_f1}
    with (run_dir / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)

    curve = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(curve.epoch, curve.train_loss, label="Train")
    axes[0].plot(curve.epoch, curve.val_loss, label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(curve.epoch, curve.train_macro_f1, label="Train")
    axes[1].plot(curve.epoch, curve.val_macro_f1, label="Validation")
    axes[1].set(title="Macro F1", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "learning_curves.png", dpi=160)
    plt.close(fig)
    print(f"Test accuracy: {test_accuracy:.4f} | test macro F1: {test_macro_f1:.4f}")
    print(f"Artifacts saved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
