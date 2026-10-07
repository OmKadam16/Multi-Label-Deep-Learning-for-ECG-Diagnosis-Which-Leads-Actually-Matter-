"""Train the 1D CNN with a class-weighted loss, checkpointing and early stopping.

    python -m src.train --data-dir data --out-dir outputs
"""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data import load_cached
from src.model import ECG1DCNN, ECGDataset

BASELINE_VAL_AUROC = 0.836  # XGBoost baseline on the validation fold, drawn as a reference line


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, preds, targets = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        total += criterion(out, yb).item() * xb.size(0)
        preds.append(torch.sigmoid(out).cpu().numpy())
        targets.append(yb.cpu().numpy())
    preds, targets = np.concatenate(preds), np.concatenate(targets)
    return total / len(loader.dataset), roc_auc_score(targets, preds, average="macro")


def main(data_dir, out_dir, epochs, patience, batch_size, lr, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    splits = load_cached(data_dir)
    (X_tr, Y_tr), (X_va, Y_va) = splits["train"], splits["val"]
    train_loader = DataLoader(ECGDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ECGDataset(X_va, Y_va), batch_size=batch_size)

    # Class weights: (negatives / positives) per class, so rare classes cost more when missed.
    pos = Y_tr.sum(axis=0)
    pos_weight = torch.tensor((len(Y_tr) - pos) / pos, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = ECG1DCNN(num_classes=Y_tr.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "best_model.pt"

    best, stale = 0.0, 0
    history = {"train_loss": [], "val_loss": [], "val_auroc": []}
    start = time.time()
    for epoch in range(epochs):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        history["val_auroc"].append(va)

        improved = va > best
        if improved:
            best, stale = va, 0
            torch.save(model.state_dict(), ckpt)
        else:
            stale += 1
        print(f"Epoch {epoch + 1}/{epochs} | train {tr:.4f} | val {vl:.4f} | val AUROC {va:.4f}"
              + (" <- best, saved" if improved else ""))
        if stale >= patience:
            print(f"Stopped early: no improvement for {patience} epochs.")
            break

    print(f"Training time {time.time() - start:.1f}s | best val AUROC {best:.4f}")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train loss")
    axes[0].plot(history["val_loss"], label="Val loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss over training")
    axes[0].legend()
    axes[1].plot(history["val_auroc"], color="green")
    axes[1].axhline(BASELINE_VAL_AUROC, color="red", linestyle="--", label="Baseline (XGBoost)")
    axes[1].set(xlabel="Epoch", ylabel="Val macro-AUROC", title="Validation AUROC vs baseline")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.data_dir, a.out_dir, a.epochs, a.patience, a.batch_size, a.lr, a.seed)
