"""Data loading, label engineering and signal preprocessing for PTB-XL (100 Hz).

Run once to build the cache:
    python -m src.data --data-dir data

Expected layout before running (download from PhysioNet, version 1.0.3):
    data/ptbxl_database.csv
    data/scp_statements.csv
    data/records100/...
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

CLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]  # alphabetical, same order MultiLabelBinarizer gives
LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 100  # sampling rate in Hz for the records100 files


def bandpass_filter(signal, fs=FS, lowcut=0.5, highcut=40, order=4):
    """Butterworth bandpass, applied to each lead (time axis = 0)."""
    nyquist = fs / 2
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal, axis=0)


def normalize_signal(signal):
    """Z-score each lead independently (mean 0, std 1)."""
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    std[std == 0] = 1e-8  # avoid divide-by-zero on a flat lead
    return (signal - mean) / std


def load_labels(data_dir):
    """Map the raw SCP codes to the 5 diagnostic superclasses.

    Returns the label table (records with at least one superclass) and the
    multi-hot label matrix Y of shape (n_records, 5), columns ordered as CLASSES.
    """
    from sklearn.preprocessing import MultiLabelBinarizer

    data_dir = Path(data_dir)
    labels_df = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    labels_df["scp_codes"] = labels_df["scp_codes"].apply(ast.literal_eval)

    scp_df = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    scp_df = scp_df[scp_df["diagnostic"] == 1]  # keep diagnostic codes only

    def codes_to_superclasses(scp_dict):
        return list({scp_df.loc[c, "diagnostic_class"] for c in scp_dict if c in scp_df.index})

    labels_df["superclasses"] = labels_df["scp_codes"].apply(codes_to_superclasses)
    labels_df = labels_df[labels_df["superclasses"].apply(len) > 0]

    mlb = MultiLabelBinarizer(classes=CLASSES)
    Y = mlb.fit_transform(labels_df["superclasses"])
    return labels_df, Y


def preprocess_record(data_dir, rel_path):
    import wfdb

    record = wfdb.rdrecord(str(Path(data_dir) / rel_path))
    return normalize_signal(bandpass_filter(record.p_signal))


def build_cache(data_dir):
    """Filter and normalize every record once, then save X, Y and the fold ids."""
    from tqdm import tqdm

    data_dir = Path(data_dir)
    labels_df, Y = load_labels(data_dir)
    X = np.zeros((len(labels_df), 1000, 12), dtype=np.float32)
    for i, rel_path in enumerate(tqdm(labels_df["filename_lr"].tolist())):
        X[i] = preprocess_record(data_dir, rel_path)

    np.save(data_dir / "X_processed.npy", X)
    np.save(data_dir / "Y_labels.npy", Y)
    labels_df[["strat_fold"]].to_csv(data_dir / "fold_assignments.csv")
    with open(data_dir / "classes.json", "w") as f:
        json.dump(CLASSES, f)
    print("X:", X.shape, "Y:", Y.shape)


def load_cached(data_dir):
    """Load the cached arrays and split with PTB-XL's official folds.

    Folds 1-8 train, fold 9 validation, fold 10 test.
    """
    data_dir = Path(data_dir)
    X = np.load(data_dir / "X_processed.npy")
    Y = np.load(data_dir / "Y_labels.npy")
    fold = pd.read_csv(data_dir / "fold_assignments.csv", index_col="ecg_id")["strat_fold"].values

    masks = {"train": fold <= 8, "val": fold == 9, "test": fold == 10}
    return {name: (X[m], Y[m]) for name, m in masks.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    build_cache(parser.parse_args().data_dir)
