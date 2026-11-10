# Mindwave AI — Damage Claim Intelligence Platform

End-to-end AI system for warehouse damage claims prevention, evidence capture, and dispute automation. Covers inbound dock inspection, cold-chain excursion prediction, outbound shipment verification, claims intelligence, and analytics.

## Overview

| Pillar | Approach | Typical ROI |
|---|---|---|
| **Damage prevention** | Computer vision + forklift telemetry + predictive cold-chain | 6–12 months |
| **Claims defensibility** | Evidence assembly + dispute automation | 3–6 months |

```
Inbound dock ──► Storage ──► Outbound dock ──► Carrier handover
     ↓               ↓               ↓                ↓
  Vision AI       IoT+LSTM      Vision AI        BOL/Seal OCR
     ↓               ↓               ↓                ↓
                Claims Intelligence Engine
                         ↓
              Dispute package generator
                         ↓
                 ERP financial close
```

## Architecture

Five-stage pipeline:

1. **Inbound inspection** — YOLO + ResNet classifier + OCR at receiving dock
2. **Storage monitoring** — LSTM forecasting + Isolation Forest + ANN risk scoring
3. **Outbound inspection** — Dock-door vision, pallet count verification, seal OCR
4. **Claims intelligence** — Evidence assembly, claim classification, inconsistency detection
5. **Analytics & risk** — Root-cause analysis, forklift risk scoring, executive dashboards

## Tech Stack

| Layer | Technology |
|---|---|
| Object detection | YOLOv9 / YOLO-World |
| Damage classification | ResNet-50 / EfficientNet-B4 |
| Label & OCR | PaddleOCR / AWS Textract |
| Edge-case review | GPT-4o Vision / LLaMA 3.2 Vision |
| Temp forecasting | LSTM (PyTorch) |
| Anomaly detection | LSTM Autoencoder + Isolation Forest |
| Proactive risk | ANN (Keras) |
| Edge runtime | NVIDIA Jetson Orin / ONNX / TensorRT |
| API layer | FastAPI |
| Event bus | MQTT / Kafka |
| WMS connectors | Manhattan Active, SAP EWM, Blue Yonder, Oracle WMS |
| TMS connectors | Oracle TMS, SAP TM |
| ERP connectors | SAP S/4HANA, Oracle Fusion, MS D365 |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/PadmavathyVijayakumar/Damage-Claim_AI-
cd Damage-Claim_AI-

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp config/.env.example config/.env
# Edit config/.env with your credentials

# 5. Start services
docker-compose up -d

# 6. Run the API
uvicorn src.api.main:app --reload --port 8000
```

API docs at `http://localhost:8000/docs`

## Project Structure

```
damage-claim_ai/
├── src/
│   ├── vision/           # 4-layer dock camera AI stack
│   ├── cold_chain/       # Temperature excursion prediction models
│   ├── evidence/         # Evidence assembly & comparison engine
│   ├── claims/           # Claim classification & dispute logic
│   ├── integrations/     # WMS / TMS / ERP connectors
│   ├── analytics/        # Root-cause, risk scoring, dashboards
│   └── api/              # FastAPI endpoints
├── tests/                # Pytest suites per module
├── notebooks/            # EDA & model training notebooks
├── docs/                 # Architecture & integration guides
└── scripts/              # Training & deployment scripts
```

## Accuracy Benchmarks

| Model | Metric | Value |
|---|---|---|
| Dock camera AI (pre-tuning) | Accuracy | 85–90% |
| Dock camera AI (after 4–8 wk tuning) | Accuracy | 94–97% |
| LSTM temperature forecast | MAE | ≤ 0.40°C |
| ANN cold-chain disruption prevention | Rate | 82%+ |
| Excursion detection latency | Response | ≤ 90 sec |

## Integration Reference

See [`docs/integration_guide.md`](docs/integration_guide.md) for per-platform connector specs covering Manhattan Active, SAP EWM, Blue Yonder, Oracle WMS, and the three-layer integration architecture.

## Licence

Proprietary — EZ Innovation / Mindwave AI. All rights reserved.
