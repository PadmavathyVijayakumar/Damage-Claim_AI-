#!/usr/bin/env python3
"""
Train the Stacked-LSTM temperature forecaster and LSTM Autoencoder
for cold-chain excursion prediction.

Usage — forecaster:
    python scripts/train_lstm.py \
        --mode forecaster \
        --data datasets/cold_chain/sensor_logs.csv \
        --output models/weights/lstm_forecaster.pt \
        --epochs 100

Usage — autoencoder:
    python scripts/train_lstm.py \
        --mode autoencoder \
        --data datasets/cold_chain/sensor_logs.csv \
        --output models/weights/lstm_ae.pt \
        --epochs 50

CSV format expected:
    timestamp (ISO-8601),
    zone_id,
    temp_outlet,temp_door_zone,temp_cargo_centre,
    humidity,door_open_flag,compressor_state,
    compressor_current,compressor_rpm
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Sensor column names (8 features matching src/cold_chain/lstm_forecaster.py)
SENSOR_COLS = [
    "temp_outlet", "temp_door_zone", "temp_cargo_centre",
    "humidity", "door_open_flag", "compressor_state",
    "compressor_current", "compressor_rpm",
]
SEQUENCE_LENGTH = 60
FORECAST_HORIZON = 60
AUTOENCODER_LATENT = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM cold-chain models")
    p.add_argument("--mode",    choices=["forecaster", "autoencoder"], required=True)
    p.add_argument("--data",    default="datasets/cold_chain/sensor_logs.csv")
    p.add_argument("--output",  default=None,
                   help="Path to save model weights (.pt); default inferred from mode")
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--batch",   type=int,   default=64)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--hidden",  type=int,   default=128)
    p.add_argument("--layers",  type=int,   default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience",type=int,   default=15)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--device",  default="auto")
    return p.parse_args()


def load_data(csv_path: str) -> np.ndarray:
    """Load and normalise sensor CSV. Returns array (N, 8)."""
    import pandas as pd
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    missing = [c for c in SENSOR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    arr = df[SENSOR_COLS].astype("float32").values
    # Min-max normalise per column
    col_min = arr.min(axis=0)
    col_max = arr.max(axis=0)
    arr = (arr - col_min) / (col_max - col_min + 1e-8)
    return arr


def make_sequences(arr: np.ndarray, seq_len: int, horizon: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Slide window → (X: N×seq×feat, y: N×horizon) using temp_outlet index 0."""
    xs, ys = [], []
    for i in range(len(arr) - seq_len - horizon + 1):
        xs.append(arr[i : i + seq_len])
        ys.append(arr[i + seq_len : i + seq_len + horizon, 0])  # temp_outlet only
    return np.array(xs, dtype="float32"), np.array(ys, dtype="float32")


def make_ae_sequences(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Sliding window for autoencoder (input = output)."""
    xs = [arr[i : i + seq_len] for i in range(len(arr) - seq_len + 1)]
    return np.array(xs, dtype="float32")


# ---------------------------------------------------------------------------
# Forecaster training
# ---------------------------------------------------------------------------

def train_forecaster(args: argparse.Namespace, arr: np.ndarray) -> None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    logger.info(f"Device: {device}")

    X, y = make_sequences(arr, SEQUENCE_LENGTH, FORECAST_HORIZON)
    split = int(len(X) * (1 - args.val_split))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    loader_tr = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=args.batch, shuffle=True,
    )
    loader_val = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=args.batch,
    )

    # Import model from project
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from src.cold_chain.lstm_forecaster import StackedLSTM

    n_features = X.shape[2]
    model = StackedLSTM(
        input_size=n_features,
        hidden_size=args.hidden,
        num_layers=args.layers,
        output_size=FORECAST_HORIZON,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True,
    )
    criterion = nn.HuberLoss(delta=0.5)

    best_val = float("inf")
    patience_count = 0
    output_path = Path(args.output or "models/weights/lstm_forecaster.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_tr)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader_val:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(X_val)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d}/{args.epochs} | "
                        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
            torch.save(model.state_dict(), output_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"Forecaster saved → {output_path}  (best val_loss={best_val:.4f})")


# ---------------------------------------------------------------------------
# Autoencoder training (normal data only → anomaly detection threshold)
# ---------------------------------------------------------------------------

def train_autoencoder(args: argparse.Namespace, arr: np.ndarray) -> None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.error("PyTorch not installed.")
        sys.exit(1)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    X = make_ae_sequences(arr, SEQUENCE_LENGTH)
    split = int(len(X) * (1 - args.val_split))
    X_tr, X_val = X[:split], X[split:]

    loader_tr = DataLoader(
        TensorDataset(torch.tensor(X_tr)),
        batch_size=args.batch, shuffle=True,
    )
    loader_val = DataLoader(
        TensorDataset(torch.tensor(X_val)),
        batch_size=args.batch,
    )

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from src.cold_chain.anomaly_detector import LSTMAutoencoder

    n_features = X.shape[2]
    model = LSTMAutoencoder(
        input_size=n_features,
        latent_size=AUTOENCODER_LATENT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    patience_count = 0
    output_path = Path(args.output or "models/weights/lstm_ae.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for (xb,) in loader_tr:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_tr)

        model.eval()
        val_loss = 0.0
        recon_errors = []
        with torch.no_grad():
            for (xb,) in loader_val:
                xb = xb.to(device)
                recon = model(xb)
                per_sample = ((recon - xb) ** 2).mean(dim=(1, 2))
                recon_errors.extend(per_sample.cpu().numpy().tolist())
                val_loss += criterion(recon, xb).item() * len(xb)
        val_loss /= len(X_val)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d}/{args.epochs} | "
                        f"train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
            torch.save(
                {"state_dict": model.state_dict(),
                 "threshold_95pct": float(np.percentile(recon_errors, 95))},
                output_path,
            )
        else:
            patience_count += 1
            if patience_count >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    threshold = np.percentile(recon_errors, 95)
    logger.info(f"Autoencoder saved → {output_path}  "
                f"(best val_loss={best_val:.5f}, 95th-pct threshold={threshold:.5f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    logger.info(f"Loading sensor data from {data_path} …")
    arr = load_data(str(data_path))
    logger.info(f"Data shape: {arr.shape}")

    if args.mode == "forecaster":
        train_forecaster(args, arr)
    else:
        train_autoencoder(args, arr)


if __name__ == "__main__":
    main()
