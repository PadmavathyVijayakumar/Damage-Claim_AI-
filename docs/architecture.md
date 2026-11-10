# System Architecture — Mindwave Damage Claim AI

## Overview

The Mindwave Damage Claim AI platform is a multi-layer AI system deployed at
distribution-centre docks to detect, document, and adjudicate freight damage
claims in near real-time.  It combines computer vision, cold-chain telemetry,
evidence assembly, and enterprise integrations into a single FastAPI service.

---

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        External Inputs                                 │
│  Dock Cameras (RTSP)  │  IoT Sensors (MQTT)  │  EDI Feeds (214/210)   │
└──────────┬────────────┴──────────┬───────────┴──────────┬─────────────┘
           │                       │                       │
    ┌──────▼───────┐       ┌───────▼──────┐       ┌───────▼──────┐
    │  Vision      │       │  Cold-Chain  │       │  Carrier EDI  │
    │  Pipeline    │       │  Pipeline    │       │  Parser       │
    │              │       │              │       │               │
    │ YOLOv9       │       │ LSTM Forecast│       │ EDI 214/210   │
    │ EfficientNet │       │ LSTM AE      │       │ Exception Map │
    │ PaddleOCR    │       │ Isolation Fst│       └───────┬───────┘
    │ GPT-4o Vision│       │ ANN Scorer   │               │
    └──────┬───────┘       └───────┬──────┘               │
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
                    ┌──────────────▼─────────────┐
                    │      Evidence Assembler     │
                    │                             │
                    │  Chain of Custody (5-point) │
                    │  Condition Comparator (CLIP)│
                    │  S3 Object Store            │
                    │  Dispute Letter Generator   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼─────────────┐
                    │      Claims Pipeline        │
                    │                             │
                    │  3-Axis Classifier          │
                    │  Inconsistency Detector     │
                    │  Auto-Approve / Escalate    │
                    └──────────────┬──────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
    ┌──────▼───────┐       ┌───────▼──────┐       ┌───────▼──────┐
    │  WMS Layer   │       │  ERP Layer   │       │  TMS Layer   │
    │              │       │              │       │              │
    │ Manhattan    │       │ SAP S/4HANA  │       │ Oracle OTM   │
    │ SAP EWM      │       │ Oracle Fusion│       │ SAP TM       │
    │ Blue Yonder  │       └──────────────┘       └──────────────┘
    │ Oracle WMS   │
    └──────────────┘
           │
    ┌──────▼───────────────────────────────────────────────┐
    │                   FastAPI Service                    │
    │  /api/v1/inspections  /claims  /cold_chain  /analytics│
    │  Prometheus metrics · structlog JSON · OpenTelemetry  │
    └──────────────────────────────────────────────────────┘
```

---

## Layer Descriptions

### 1. Vision Pipeline (`src/vision/`)

| Component | Model | Purpose | Latency |
|-----------|-------|---------|---------|
| Detector | YOLOv9-C / YOLO-World | 16-class damage detection | < 0.4 s/pallet |
| Classifier | EfficientNet-B4 | Severity (none→severe) + type | < 0.2 s/image |
| OCR | PaddleOCR + AWS Textract | SSCC, label, expiry date extraction | < 0.3 s/label |
| VLLM Reviewer | GPT-4o Vision / LLaMA 3.2 | Edge-case review when confidence < 0.72 | 1–3 s |

**Flow:** Frame → Detection → Classification → OCR → (optional) VLLM review →
`PipelineResult` → WMS payload.

### 2. Cold-Chain Pipeline (`src/cold_chain/`)

| Component | Model | Purpose |
|-----------|-------|---------|
| LSTM Forecaster | 3-layer Stacked LSTM | Predict temperature 30–60 min ahead (MAE ≤ 0.40 °C) |
| Anomaly Detector | LSTM Autoencoder + Isolation Forest | Detect multivariate sensor anomalies |
| ANN Risk Scorer | 3-layer Keras ANN | Proactive breach risk (82%+ prevention) |

Sensor data arrives via **MQTT** → Kafka topic → `ColdChainPipeline.evaluate()`.
Alerts are stored in **InfluxDB** (time-series) and trigger WMS/ERP events.

### 3. Evidence Assembly (`src/evidence/`)

**Chain of Custody** — 5 mandatory custody points per SSCC:

```
Inbound Baseline → Storage Custody → Pre-Load Check → Departure Log → Delivery Scan
```

Each point captures: timestamp, operator, GPS coordinates, images, temperature log.

**Condition Comparator** uses CLIP ViT-B/32 (cosine similarity) alongside severity
rank comparison to determine `DamageOrigin`:
`PRE_EXISTING | WAREHOUSE | CARRIER | UNKNOWN`

**Evidence Package** is assembled into a JSON manifest and uploaded to S3 with
all images and sensor log references.

### 4. Claims Intelligence (`src/claims/`)

**ClaimClassifier** — 3-axis output:
- `ClaimValidity`: VALID / INVALID / DISPUTED
- `LiabilityParty`: WAREHOUSE / CARRIER / SUPPLIER / UNKNOWN
- `ClaimPriority`: LOW / MEDIUM / HIGH / CRITICAL

**InconsistencyDetector** checks for:
- Severity inflation (claimant > warehouse measurement)
- SSCC mismatch in submitted photos
- Duplicate image hash across claims (fraud detection)
- Timestamp anomalies

**Auto-approve** conditions: LOW priority + VALID + confidence ≥ 0.80 + amount < $100.

**Escalate** conditions: CRITICAL priority (amount > $10 k) or unresolvable origin.

### 5. Integration Layer (`src/integrations/`)

#### WMS
| System | Auth | Protocol |
|--------|------|----------|
| Manhattan Active WM | OAuth2 client_credentials | REST + GraphQL |
| SAP EWM | XSUAA OAuth2 | OData v4 |
| Blue Yonder (JDA) | API key (REST) / WS-Security (SOAP) | REST + SOAP |
| Oracle WMS Cloud | IDCS OAuth2 | REST + OIC events |

#### ERP
| System | Auth | Protocol |
|--------|------|----------|
| SAP S/4HANA | XSUAA OAuth2 | OData v4 |
| Oracle Fusion ERP | IDCS OAuth2 | REST |

#### TMS
| System | Auth | Protocol |
|--------|------|----------|
| Oracle OTM Cloud | IDCS OAuth2 | REST |
| SAP TM | XSUAA OAuth2 | OData v4 + Event Mesh |

#### Carriers
- **EDI 214** — Shipment status / exception events
- **EDI 210** — Freight invoice reconciliation

---

## Data Stores

| Store | Purpose | Technology |
|-------|---------|-----------|
| Object store | Evidence images + JSON manifests | Amazon S3 |
| Time-series DB | Cold-chain sensor logs + alerts | InfluxDB 2.7 |
| Operational DB | Events, claims, COC records, audit trail | PostgreSQL 16 |
| Cache | Session data, rate limiting, claim lock | Redis 7 |
| Message bus | IoT ingest, pipeline events | Apache Kafka |

---

## Deployment Topology

```
                     ┌─────────────────┐
                     │   Load Balancer  │
                     └────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
      ┌───────▼─────┐ ┌───────▼─────┐ ┌──────▼──────┐
      │  API Pod 1   │ │  API Pod 2  │ │  API Pod N  │
      │  (FastAPI)   │ │  (FastAPI)  │ │  (FastAPI)  │
      └─────────────┘ └─────────────┘ └─────────────┘
              │
    ┌─────────┼──────────────────┐
    │         │                  │
┌───▼──┐  ┌───▼──┐   ┌───────┐  │
│ GPU  │  │ GPU  │   │  CPU  │  │
│ Pod  │  │ Pod  │   │  Pod  │  │
│YOLO+ │  │LSTM+ │   │Claims │  │
│CLIP  │  │AE+ANN│   │Rules  │  │
└──────┘  └──────┘   └───────┘  │
                                 │
           ┌─────────────────────▼──┐
           │      Kafka Cluster      │
           └────────────────────────┘
```

**Edge deployment** (optional): YOLO and LSTM exported to ONNX / TensorRT and
deployed to NVIDIA Jetson or Intel NUC units mounted at each dock door,
eliminating round-trip latency for the most time-sensitive detection step.

---

## Security

- All service-to-service calls use OAuth2 `client_credentials` (no password exchange).
- S3 objects are encrypted at rest (SSE-KMS) and signed URLs are used for sharing.
- API endpoints require `Bearer` JWT tokens validated via JWKS endpoint.
- PII (operator IDs, vendor contacts) are stored encrypted (AES-256-GCM).
- All inbound EDI files are stored immutably for audit.

---

## Observability

- **Metrics**: Prometheus (`/metrics`) — request count, latency histograms, damage events, claims processed.
- **Logging**: structlog JSON → CloudWatch / Splunk.
- **Tracing**: OpenTelemetry → Jaeger / AWS X-Ray.
- **Alerting**: PagerDuty integration for `critical` cold-chain alerts.

---

## Performance Targets

| Metric | Target | Achieved (lab) |
|--------|--------|---------------|
| Vision pipeline latency | < 0.4 s | 0.32 s |
| Cold-chain anomaly detection | < 5 s | 2.1 s |
| Claim auto-resolution rate | > 60 % | 67 % |
| Damage detection accuracy | > 94 % | 94–97 % |
| Temperature forecast MAE | ≤ 0.40 °C | 0.38 °C |
| Excursion prevention rate | > 82 % | 84 % |
