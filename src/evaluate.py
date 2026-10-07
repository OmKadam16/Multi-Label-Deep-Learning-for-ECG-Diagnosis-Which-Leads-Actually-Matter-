"""Score the best checkpoint on the untouched test fold.

    python -m src.evaluate --data-dir data --out-dir outputs
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data import CLASSES, load_cached
from src.model import ECG1DCNN, ECGDataset


@torch.no_grad()
def predict(model, loader, device, zero_lead=None):
    """Return (probabilities, targets). If zero_lead is an index, that lead is set to 0
    for every record before the forward pass (used by the lead ablation)."""
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        if zero_lead is not None:
            xb = xb.clone()
            xb[:, zero_lead, :] = 0.0
        preds.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
        targets.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def load_best(out_dir, device):
    model = ECG1DCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(Path(out_dir) / "best_model.pt", map_location=device))
    return model.eval()


def main(data_dir, out_dir, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = load_cached(data_dir)["test"]
    loader = DataLoader(ECGDataset(X, Y), batch_size=batch_size)

    preds, targets = predict(load_best(out_dir, device), loader, device)
    results = {
        "test_macro_auroc": float(roc_auc_score(targets, preds, average="macro")),
        "test_macro_f1": float(f1_score(targets, (preds > 0.5).astype(int), average="macro")),
        "per_class_auroc": {
            c: float(a) for c, a in zip(CLASSES, roc_auc_score(targets, preds, average=None))
        },
    }
    with open(Path(out_dir) / "cnn_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"TEST macro-AUROC {results['test_macro_auroc']:.4f} | macro-F1 {results['test_macro_f1']:.4f}")
    for c, a in results["per_class_auroc"].items():
        print(f"  {c}: {a:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="outputs")
    a = p.parse_args()
    main(a.data_dir, a.out_dir)
