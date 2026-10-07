"""Dataset wrapper and the 1D CNN used in the project."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """Wraps (N, 1000, 12) signals and (N, 5) labels. Signals are permuted to (N, 12, 1000)
    because Conv1d expects channels (leads) first."""

    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class ECG1DCNN(nn.Module):
    """Four Conv1d blocks (12 -> 32 -> 64 -> 128 -> 256 channels), each with
    BatchNorm, ReLU and MaxPool(2), then global average pooling, dropout and a
    linear layer. Outputs raw logits, one per class. Sigmoid is applied in the
    loss (BCEWithLogitsLoss) and at prediction time."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)
