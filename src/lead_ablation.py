"""Lead ablation: zero out one lead at a time on the test fold and measure the macro-AUROC drop.

    python -m src.lead_ablation --data-dir data --out-dir outputs
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data import LEAD_NAMES, load_cached
from src.evaluate import load_best, predict
from src.model import ECGDataset


def main(data_dir, out_dir, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = load_cached(data_dir)["test"]
    loader = DataLoader(ECGDataset(X, Y), batch_size=batch_size)
    model = load_best(out_dir, device)

    preds, targets = predict(model, loader, device)
    full = roc_auc_score(targets, preds, average="macro")
    print(f"All 12 leads: macro-AUROC {full:.4f}")

    results = {}
    for idx, name in enumerate(LEAD_NAMES):
        preds, targets = predict(model, loader, device, zero_lead=idx)
        auroc = roc_auc_score(targets, preds, average="macro")
        results[name] = {"ablated_auroc": float(auroc), "auroc_drop": float(full - auroc)}
        print(f"Zeroing {name:>3}: AUROC {auroc:.4f} (drop {full - auroc:+.4f})")

    out_dir = Path(out_dir)
    with open(out_dir / "lead_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    ranked = sorted(results.items(), key=lambda kv: kv[1]["auroc_drop"], reverse=True)
    plt.figure(figsize=(10, 5))
    plt.bar([k for k, _ in ranked], [v["auroc_drop"] for _, v in ranked], color="darkorange")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Lead importance: AUROC drop when each lead is zeroed out")
    plt.xlabel("ECG Lead")
    plt.ylabel("AUROC Drop")
    plt.tight_layout()
    plt.savefig(out_dir / "lead_ablation.png", dpi=150)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="outputs")
    a = p.parse_args()
    main(a.data_dir, a.out_dir)
