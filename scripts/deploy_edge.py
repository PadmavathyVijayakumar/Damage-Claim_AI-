#!/usr/bin/env python3
"""
Export trained models to edge-deployable formats: ONNX and TensorRT.

Supported models:
  yolo      — YOLOv9 damage detector (PyTorch → ONNX → TensorRT)
  classifier — EfficientNet-B4 damage classifier (PyTorch → ONNX → TensorRT)
  lstm      — LSTM temperature forecaster (PyTorch → ONNX)

Usage:
    # Export YOLO to ONNX (all edge targets)
    python scripts/deploy_edge.py \
        --model yolo \
        --weights models/weights/yolo_damage.pt \
        --format onnx

    # Export YOLO to TensorRT FP16 (requires NVIDIA GPU + TensorRT)
    python scripts/deploy_edge.py \
        --model yolo \
        --weights models/weights/yolo_damage.pt \
        --format tensorrt \
        --fp16

    # Export EfficientNet-B4 classifier
    python scripts/deploy_edge.py \
        --model classifier \
        --weights models/weights/efficientnet_damage.pt \
        --format onnx \
        --img 224
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export models to ONNX / TensorRT for edge deployment")
    p.add_argument("--model",   choices=["yolo", "classifier", "lstm"], required=True)
    p.add_argument("--weights", required=True,
                   help="Path to trained .pt weights")
    p.add_argument("--format",  choices=["onnx", "tensorrt"],
                   default="onnx")
    p.add_argument("--output",  default=None,
                   help="Output path; default: same directory as weights")
    p.add_argument("--img",     type=int, default=640,
                   help="Input image size (YOLO / classifier only)")
    p.add_argument("--batch",   type=int, default=1,
                   help="Static batch size for ONNX export")
    p.add_argument("--fp16",    action="store_true",
                   help="Use FP16 precision for TensorRT")
    p.add_argument("--int8",    action="store_true",
                   help="Use INT8 precision for TensorRT (requires calibration data)")
    p.add_argument("--dynamic", action="store_true",
                   help="Enable dynamic axes in ONNX (batch / image size)")
    p.add_argument("--opset",   type=int, default=17,
                   help="ONNX opset version")
    p.add_argument("--simplify",action="store_true", default=True,
                   help="Run onnx-simplifier after export")
    p.add_argument("--calib-data", default=None,
                   help="Path to INT8 calibration images directory")
    p.add_argument("--workspace-gb", type=float, default=4.0,
                   help="TensorRT builder workspace in GB")
    return p.parse_args()


# ---------------------------------------------------------------------------
# YOLO export
# ---------------------------------------------------------------------------

def export_yolo(args: argparse.Namespace) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        sys.exit(1)

    model = YOLO(args.weights)
    kwargs: dict = {
        "imgsz": args.img,
        "batch": args.batch,
        "opset": args.opset,
        "simplify": args.simplify,
        "dynamic": args.dynamic,
    }
    if args.format == "tensorrt":
        kwargs.update({"format": "engine", "half": args.fp16, "int8": args.int8,
                       "workspace": int(args.workspace_gb * 1024)})
        logger.info("Exporting YOLO → TensorRT …")
        path = model.export(**kwargs)
    else:
        kwargs["format"] = "onnx"
        logger.info("Exporting YOLO → ONNX …")
        path = model.export(**kwargs)

    return Path(str(path))


# ---------------------------------------------------------------------------
# EfficientNet-B4 classifier export
# ---------------------------------------------------------------------------

def export_classifier(args: argparse.Namespace) -> Path:
    try:
        import torch
        import torch.onnx
        from torchvision import models
    except ImportError:
        logger.error("torch / torchvision not installed.")
        sys.exit(1)

    logger.info("Loading EfficientNet-B4 …")
    # Rebuild model architecture
    num_classes = 7  # damage severity classes
    model = models.efficientnet_b4(weights=None)
    import torch.nn as nn
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    state_dict = torch.load(args.weights, map_location="cpu")
    # Strip DataParallel wrappers if present
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(args.batch, 3, args.img, args.img)
    output_path = Path(args.output or Path(args.weights).with_suffix(".onnx"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    logger.info(f"Exporting EfficientNet-B4 → ONNX opset {args.opset} …")
    torch.onnx.export(
        model, dummy, str(output_path),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    if args.simplify:
        try:
            import onnx
            import onnxsim
            logger.info("Simplifying ONNX graph …")
            m = onnx.load(str(output_path))
            m_simplified, ok = onnxsim.simplify(m)
            if ok:
                onnx.save(m_simplified, str(output_path))
                logger.info("ONNX simplified OK")
        except ImportError:
            logger.warning("onnx-simplifier not installed; skipping simplification")

    if args.format == "tensorrt":
        output_path = _onnx_to_trt(output_path, args)

    return output_path


# ---------------------------------------------------------------------------
# LSTM export
# ---------------------------------------------------------------------------

def export_lstm(args: argparse.Namespace) -> Path:
    try:
        import torch
    except ImportError:
        logger.error("torch not installed.")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from src.cold_chain.lstm_forecaster import StackedLSTM

    # Reconstruct with defaults; override hidden/layers via CLI if needed
    model = StackedLSTM(input_size=8, hidden_size=128, num_layers=3, output_size=60)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # LSTM requires (batch, seq, features) → (batch, forecast_horizon)
    dummy = torch.randn(args.batch, 60, 8)
    output_path = Path(args.output or Path(args.weights).with_suffix(".onnx"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting LSTM forecaster → ONNX opset {args.opset} …")
    torch.onnx.export(
        model, dummy, str(output_path),
        opset_version=args.opset,
        input_names=["sensor_sequence"],
        output_names=["temperature_forecast"],
        dynamic_axes={"sensor_sequence": {0: "batch"}} if args.dynamic else None,
        do_constant_folding=True,
    )
    return output_path


# ---------------------------------------------------------------------------
# ONNX → TensorRT helper
# ---------------------------------------------------------------------------

def _onnx_to_trt(onnx_path: Path, args: argparse.Namespace) -> Path:
    try:
        import tensorrt as trt
    except ImportError:
        logger.error("TensorRT Python bindings not installed.")
        sys.exit(1)

    trt_path = onnx_path.with_suffix(".engine")
    logger.info(f"Building TensorRT engine → {trt_path} …")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(args.workspace_gb * 1024 ** 3),
    )
    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 precision enabled")
    if args.int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        logger.warning("INT8 calibration not configured; expect degraded accuracy")

    engine_bytes = builder.build_serialized_network(network, config)
    with open(trt_path, "wb") as f:
        f.write(engine_bytes)
    logger.info(f"TensorRT engine saved → {trt_path}")
    return trt_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights file not found: {weights_path}")
        sys.exit(1)

    if args.model == "yolo":
        out = export_yolo(args)
    elif args.model == "classifier":
        out = export_classifier(args)
    else:
        out = export_lstm(args)

    logger.info(f"Export complete → {out}")
    logger.info(f"Model size: {out.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
