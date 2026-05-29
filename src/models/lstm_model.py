"""
PyTorch 2-layer LSTM for 5-class stock prediction.
Input: sliding window of sequence_length trading days × n_features.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, n_classes: int = 5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # last timestep


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert flat feature matrix to (n_samples, seq_len, n_features) sequences."""
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i - seq_len: i])
        ys.append(y[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)


class LSTMModel:
    """PyTorch LSTM wrapper with sklearn-compatible interface."""

    def __init__(
        self,
        symbol: str,
        sequence_length: int = 20,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        random_state: int = 42,
    ):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.net: Optional[_LSTMNet] = None
        self.n_features: int = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> "LSTMModel":
        torch.manual_seed(self.random_state)
        self.n_features = X_train.shape[1]

        X_tr_seq, y_tr_seq = _make_sequences(X_train, y_train, self.sequence_length)
        X_val_seq, y_val_seq = _make_sequences(X_val, y_val, self.sequence_length)

        if len(X_tr_seq) == 0:
            logger.warning("%s LSTM: not enough data for sequences (need > %d rows)", self.symbol, self.sequence_length)
            return self

        train_ds = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq))
        val_ds   = TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val_seq))
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)

        self.net = _LSTMNet(self.n_features, self.hidden_size, self.num_layers, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            self.net.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()

            # Validation
            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_loss += criterion(self.net(xb), yb).item()
            val_loss /= max(len(val_dl), 1)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                logger.debug("%s LSTM epoch %d/%d val_loss=%.4f", self.symbol, epoch + 1, self.epochs, val_loss)

        if best_state:
            self.net.load_state_dict(best_state)
        logger.info("%s LSTM fitted | best_val_loss=%.4f", self.symbol, best_val_loss)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.net is None or len(X) <= self.sequence_length:
            n = max(len(X), 1)
            return np.full((n, 5), 0.2)
        X_seq, _ = _make_sequences(X, np.zeros(len(X)), self.sequence_length)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(X_seq).to(self.device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # Pad front rows that have no sequence with uniform probs
        pad = np.full((self.sequence_length, 5), 0.2)
        return np.vstack([pad, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict() if self.net else None,
            "config": {
                "symbol": self.symbol,
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "n_features": self.n_features,
            },
        }, path)
        logger.info("LSTMModel saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        data = torch.load(path, map_location="cpu", weights_only=False)
        cfg = data["config"]
        obj = cls(
            symbol=cfg["symbol"],
            sequence_length=cfg["sequence_length"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
        obj.n_features = cfg["n_features"]
        if data["state_dict"] and obj.n_features > 0:
            obj.net = _LSTMNet(obj.n_features, obj.hidden_size, obj.num_layers, obj.dropout)
            obj.net.load_state_dict(data["state_dict"])
            obj.net.eval()
        return obj
