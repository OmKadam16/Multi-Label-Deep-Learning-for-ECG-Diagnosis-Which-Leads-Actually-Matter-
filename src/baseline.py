"""XGBoost baseline on simple per-lead statistics.

    python -m src.baseline --data-dir data --out-dir outputs
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from src.data import CLASSES, load_cached


def extract_features(X):
    """(N, 1000, 12) -> (N, 72): mean, std, min, max, 25th and 75th percentile for each lead.
    Ordering is lead-major, so lead 0 contributes the first six columns."""
    stats = [
        X.mean(axis=1),
        X.std(axis=1),
        X.min(axis=1),
        X.max(axis=1),
        np.percentile(X, 25, axis=1),
        np.percentile(X, 75, axis=1),
    ]  # each (N, 12)
    return np.stack(stats, axis=2).reshape(len(X), -1)


def score(model, F, Y):
    proba = np.array([est.predict_proba(F)[:, 1] for est in model.estimators_]).T
    return {
        "macro_auroc": float(roc_auc_score(Y, proba, average="macro")),
        "macro_f1": float(f1_score(Y, (proba > 0.5).astype(int), average="macro")),
        "per_class_auroc": {
            c: float(a) for c, a in zip(CLASSES, roc_auc_score(Y, proba, average=None))
        },
    }


def main(data_dir, out_dir):
    splits = load_cached(data_dir)
    feats = {name: extract_features(X) for name, (X, _) in splits.items()}

    base = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="logloss", random_state=42
    )
    model = MultiOutputClassifier(base)  # one independent XGBoost per superclass
    model.fit(feats["train"], splits["train"][1])

    results = {name: score(model, feats[name], splits[name][1]) for name in ("val", "test")}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    for name, r in results.items():
        print(f"{name}: macro-AUROC {r['macro_auroc']:.4f} | macro-F1 {r['macro_f1']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="outputs")
    a = p.parse_args()
    main(a.data_dir, a.out_dir)
