#!/usr/bin/env python3
"""
Train the ANN risk-scoring model for cold-chain excursion risk.

Usage:
    python scripts/train_ann.py \
        --data datasets/cold_chain/labeled_excursions.csv \
        --output models/weights/ann_risk.h5 \
        --epochs 200

CSV format expected:
    # 10 feature columns (must match ANN_FEATURES in src/cold_chain/ann_risk_scorer.py):
    temp_mean, temp_std, temp_max, humidity_mean, door_open_total_min,
    compressor_cycle_count, temp_trend_slope, transit_duration_h,
    zone_encoded, time_of_day_sin
    # Binary label:
    excursion_occurred  (0 or 1)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "temp_mean", "temp_std", "temp_max", "humidity_mean",
    "door_open_total_min", "compressor_cycle_count", "temp_trend_slope",
    "transit_duration_h", "zone_encoded", "time_of_day_sin",
]
LABEL_COL = "excursion_occurred"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ANN cold-chain risk model")
    p.add_argument("--data",    default="datasets/cold_chain/labeled_excursions.csv")
    p.add_argument("--output",  default="models/weights/ann_risk.h5")
    p.add_argument("--epochs",  type=int,   default=200)
    p.add_argument("--batch",   type=int,   default=256)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--hidden-layers", nargs="+", type=int, default=[64, 32, 16],
                   dest="hidden_layers")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--l2",      type=float, default=1e-4,
                   help="L2 regularisation weight")
    p.add_argument("--patience",type=int,   default=20)
    p.add_argument("--val-split", type=float, default=0.20)
    p.add_argument("--class-weight-positive", type=float, default=3.0,
                   dest="cw_pos",
                   help="Weight for positive (excursion) class (handles imbalance)")
    return p.parse_args()


def load_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    X = df[FEATURE_COLS].astype("float32").values
    y = df[LABEL_COL].astype("float32").values
    logger.info(f"Dataset: {len(X)} samples | "
                f"positive rate: {y.mean()*100:.1f}%")
    return X, y


def build_model(n_features: int, hidden_layers: list[int], dropout: float, l2: float):
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, regularizers
    except ImportError:
        logger.error("TensorFlow not installed. Run: pip install tensorflow>=2.16")
        sys.exit(1)

    reg = regularizers.l2(l2)
    inputs = tf.keras.Input(shape=(n_features,), name="features")
    x = layers.BatchNormalization()(inputs)
    for units in hidden_layers:
        x = layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="risk_score")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    X, y = load_dataset(str(data_path))

    # Shuffle + split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(len(X) * (1 - args.val_split))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    try:
        import tensorflow as tf
    except ImportError:
        logger.error("TensorFlow not installed.")
        sys.exit(1)

    model = build_model(
        n_features=len(FEATURE_COLS),
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
        l2=args.l2,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=args.patience, restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_path),
            monitor="val_auc", mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, verbose=1,
        ),
    ]

    class_weight = {0: 1.0, 1: args.cw_pos}

    logger.info(f"Training ANN  epochs={args.epochs} batch={args.batch} "
                f"class_weight={class_weight}")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # Evaluate best model
    val_metrics = model.evaluate(X_val, y_val, verbose=0)
    metric_names = ["loss"] + [m.name for m in model.metrics]
    logger.info("--- Best Model Validation Metrics ---")
    for name, val in zip(metric_names, val_metrics):
        logger.info(f"  {name}: {val:.4f}")

    # Save a 95th-percentile calibration threshold alongside model
    preds = model.predict(X_val, verbose=0).flatten()
    threshold_95 = float(np.percentile(preds[y_val == 0], 95))
    logger.info(f"  Calibrated threshold (95th pct negative): {threshold_95:.4f}")

    # Save threshold to companion JSON
    import json
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "feature_cols": FEATURE_COLS,
                "risk_threshold": threshold_95,
                "val_auc": val_metrics[metric_names.index("auc")],
                "hidden_layers": args.hidden_layers,
                "dropout": args.dropout,
            },
            f,
            indent=2,
        )
    logger.info(f"Model saved → {output_path}")
    logger.info(f"Metadata   → {meta_path}")


if __name__ == "__main__":
    main()
