#!/usr/bin/env python3
"""
Fine-tune YOLOv9 / YOLO-World on the Mindwave damage-detection dataset.

Usage:
    python scripts/train_yolo.py \
        --data datasets/damage_detection/data.yaml \
        --epochs 100 \
        --batch 16 \
        --img 640 \
        --weights yolov9c.pt \
        --output models/weights/yolo_damage.pt

Dataset structure expected (YOLO format):
    datasets/damage_detection/
        data.yaml
        images/train/   *.jpg
        images/val/     *.jpg
        labels/train/   *.txt
        labels/val/     *.txt

data.yaml format:
    path: datasets/damage_detection
    train: images/train
    val:   images/val
    nc: 16
    names:
      0: crush
      1: puncture
      2: tear
      3: wet_damage
      ...
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLOv9 for damage detection")
    p.add_argument("--data",    default="datasets/damage_detection/data.yaml")
    p.add_argument("--weights", default="yolov9c.pt",
                   help="Base weights: yolov9c.pt, yolov9e.pt, yolo_world_v2m.pt")
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--batch",   type=int,   default=16)
    p.add_argument("--img",     type=int,   default=640)
    p.add_argument("--workers", type=int,   default=8)
    p.add_argument("--device",  default="0",   help="GPU id(s) or 'cpu'")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name",    default="damage_yolo")
    p.add_argument("--output",  default="models/weights/yolo_damage.pt",
                   help="Copy best.pt to this path after training")
    p.add_argument("--resume",  action="store_true",
                   help="Resume training from last checkpoint")
    p.add_argument("--patience",type=int,   default=50,
                   help="Early stopping patience")
    p.add_argument("--lr0",     type=float, default=0.01)
    p.add_argument("--lrf",     type=float, default=0.01)
    p.add_argument("--mosaic",  type=float, default=1.0,
                   help="Mosaic augmentation probability")
    p.add_argument("--mixup",   type=float, default=0.1)
    p.add_argument("--conf-thres", type=float, default=0.45,
                   dest="conf_thres")
    p.add_argument("--iou-thres",  type=float, default=0.45,
                   dest="iou_thres")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics is not installed. Run: pip install ultralytics>=8.2")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Dataset YAML not found: {data_path}")
        logger.info("Create the dataset structure described in this script's docstring.")
        sys.exit(1)

    logger.info(f"Loading base model: {args.weights}")
    model = YOLO(args.weights)

    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        patience=args.patience,
        lr0=args.lr0,
        lrf=args.lrf,
        mosaic=args.mosaic,
        mixup=args.mixup,
        conf=args.conf_thres,
        iou=args.iou_thres,
        # Augmentation hardened for warehouse imagery
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.01,
        fliplr=0.5,
    )

    logger.info("Starting training …")
    results = model.train(**train_kwargs)

    # Locate best checkpoint
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_pt.exists():
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(best_pt, output)
        logger.info(f"Best model saved → {output}")
    else:
        logger.warning(f"best.pt not found at expected path {best_pt}")

    # Print final metrics
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        logger.info("--- Validation Metrics ---")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
