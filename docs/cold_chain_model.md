# Cold-Chain AI Model Documentation

## Overview

The cold-chain module predicts temperature excursions and assesses product risk
before they result in a spoilage event.  It combines four ML models in sequence,
each covering a different failure mode:

| Model | Purpose | Key Metric |
|-------|---------|-----------|
| Stacked LSTM Forecaster | 30–60 min temperature look-ahead | MAE ≤ 0.40 °C |
| LSTM Autoencoder | Multivariate sensor anomaly detection | 95th-pct threshold |
| Isolation Forest | Cross-channel anomaly detection | Contamination rate ≈ 0.05 |
| ANN Risk Scorer | Proactive excursion risk score | AUC ≥ 0.90, prevention 82%+ |

---

## Sensor Data Schema

Each zone produces an 8-dimensional sensor vector at 1-minute intervals:

| Index | Column | Units | Typical Range |
|-------|--------|-------|---------------|
| 0 | `temp_outlet` | °C | Zone-specific |
| 1 | `temp_door_zone` | °C | Zone-specific |
| 2 | `temp_cargo_centre` | °C | Zone-specific |
| 3 | `humidity` | %RH | 40–90 |
| 4 | `door_open_flag` | 0/1 | — |
| 5 | `compressor_state` | 0/1 | — |
| 6 | `compressor_current` | A | 0–20 |
| 7 | `compressor_rpm` | RPM | 0–4000 |

**Data collection:** Sensors publish to MQTT topics `sensors/<zone_id>/<sensor_type>`.
The cold-chain Kafka consumer subscribes, assembles 60-row windows, and calls
`ColdChainPipeline.evaluate()`.

---

## Zone Temperature Thresholds

Defined in `src/cold_chain/pipeline.py`:

| Zone | Min (°C) | Max (°C) | Critical Max (°C) |
|------|----------|----------|-------------------|
| `pharma_vault` | 2.0 | 8.0 | 10.0 |
| `frozen_storage` | -24.0 | -18.0 | — |
| `loading_dock` | -2.0 | 4.5 | 6.0 |
| `ambient_storage` | 15.0 | 25.0 | 28.0 |
| `controlled_ambient` | 15.0 | 25.0 | 28.0 |

---

## Model 1: Stacked LSTM Forecaster

**File:** `src/cold_chain/lstm_forecaster.py`

### Architecture

```
Input (batch × 60 × 8)
        ↓
LSTM Layer 1 (hidden=128, return_sequences=True, dropout=0.2)
        ↓
LSTM Layer 2 (hidden=128, return_sequences=True, dropout=0.2)
        ↓
LSTM Layer 3 (hidden=128, return_sequences=False, dropout=0.2)
        ↓
Linear (128 → 60)   ← 60-minute temperature forecast
        ↓
Output (batch × 60)  [temp_outlet forecast, normalised]
```

### Training
- **Loss:** Huber loss (δ = 0.5)
- **Optimiser:** AdamW (lr = 1e-3, weight decay = 1e-4)
- **Scheduler:** ReduceLROnPlateau (patience = 5, factor = 0.5)
- **Early stopping:** patience = 15 on validation loss

### Output
`LSTMForecast` dataclass containing:
- `forecast_60min`: numpy array (60,) — predicted `temp_outlet` values
- `predicted_breach_at_min`: first minute exceeding zone threshold (or `None`)
- `max_predicted_temp`: peak temperature in forecast window

### Training data requirements
Minimum 6 months of 1-minute sensor logs per zone type.  Label quality is key:
ensure sensor calibration checks are performed monthly.

---

## Model 2: LSTM Autoencoder

**File:** `src/cold_chain/anomaly_detector.py`

### Architecture

```
Encoder:
  Input (batch × 60 × 8)
  LSTM (128 hidden, return_sequences=False) → latent (batch × 32)

Decoder:
  Repeat (batch × 32) → (batch × 60 × 32)
  LSTM (128 hidden, return_sequences=True) → (batch × 60 × 128)
  Linear (128 → 8)
  Output (batch × 60 × 8)  [reconstruction]
```

### Training strategy
- **Normal-only training:** Train on sensor sequences with zero excursions.
  The model learns the normal manifold; anomalies reconstruct poorly.
- **Threshold calibration:** At training end, compute reconstruction errors on
  the held-out validation set.  Set the alert threshold at the 95th percentile.
- **Loss:** MSE on the reconstructed sequence.

### Scoring
`AnomalyScore.reconstruction_error`: mean squared error over the 60-step window.
Flag as anomaly when `reconstruction_error > threshold_95`.

---

## Model 3: Isolation Forest

**File:** `src/cold_chain/anomaly_detector.py`

### Features (5 dimensions)
1. `temp_outlet` (last reading)
2. `humidity` (last reading)
3. `compressor_state` (last reading)
4. `compressor_current` (last reading)
5. `compressor_rpm` (last reading)

### Hyperparameters
- `n_estimators = 200`
- `contamination = 0.05` (assumed 5% anomaly rate in training data)
- `max_samples = "auto"`

### Role
Complements the LSTM Autoencoder by detecting cross-channel inconsistencies —
e.g., compressor running at high current without corresponding temperature drop.
Both models must flag for a combined alert.

---

## Model 4: ANN Risk Scorer

**File:** `src/cold_chain/ann_risk_scorer.py`

### Input Features (10)
```python
ANN_FEATURES = [
    "temp_mean",              # mean of last 60 readings
    "temp_std",               # standard deviation
    "temp_max",               # maximum temperature
    "humidity_mean",          # mean humidity
    "door_open_total_min",    # total door-open time (minutes)
    "compressor_cycle_count", # number of compressor on/off cycles
    "temp_trend_slope",       # linear regression slope (°C/min)
    "transit_duration_h",     # time since last intake scan (hours)
    "zone_encoded",           # integer zone encoding
    "time_of_day_sin",        # sin(2π × hour/24) — cyclical encoding
]
```

### Architecture
```
Input (10)
   ↓ BatchNorm
Dense (64, ReLU) + BatchNorm + Dropout(0.3)
Dense (32, ReLU) + BatchNorm + Dropout(0.3)
Dense (16, ReLU) + BatchNorm + Dropout(0.3)
Dense (1, Sigmoid)  → risk score 0.0–1.0
```

### Decision thresholds
| Score | Risk Level | Action |
|-------|-----------|--------|
| < 0.30 | Low | Monitor normally |
| 0.30–0.70 | Medium | Increase monitoring frequency (5 min) |
| 0.70–0.85 | High | Alert supervisor + pre-cool adjacent zone |
| ≥ 0.85 | Critical | Immediate intervention + claim evidence capture |

### Class weighting
Positive (excursion) class weight = 3.0 to handle class imbalance (typically
2–5% of records represent true excursions in a well-managed DC).

---

## Pipeline Orchestration

**File:** `src/cold_chain/pipeline.py`

```
ColdChainPipeline.evaluate(sensor_sequence, latest_multi_sensor)
      │
      ├─► LSTM Forecaster → LSTMForecast (breach in next 60 min?)
      │
      ├─► LSTM Autoencoder → AnomalyScore (is this sequence unusual?)
      │
      ├─► Isolation Forest → AnomalyScore (cross-channel anomaly?)
      │
      ├─► Threshold check → Alert if current temp > zone max/min
      │
      ├─► ANN Risk Scorer → RiskScore (0–1) + action recommendation
      │
      └─► ColdChainPipelineResult
            ├── alerts: List[ColdChainAlert]
            ├── forecast: LSTMForecast
            ├── ae_anomaly: AnomalyScore
            ├── iforest_anomaly: AnomalyScore
            ├── risk_score: RiskScore
            ├── overall_risk_level: "normal" | "elevated" | "critical"
            └── requires_immediate_action: bool
```

---

## Retraining Schedule

| Model | Minimum Retraining Interval | Trigger |
|-------|---------------------------|---------|
| LSTM Forecaster | Quarterly | MAE > 0.60 °C on live data |
| LSTM Autoencoder | Quarterly | False-positive rate > 10% |
| Isolation Forest | Monthly | Contamination shift > 2% |
| ANN Risk Scorer | Quarterly | AUC drop below 0.85 |

Retrain using the scripts in `scripts/train_lstm.py` and `scripts/train_ann.py`.
Always retrain on the 12 most recent months of labelled data.

---

## Cold-Chain Claim Evidence

When `ColdChainPipelineResult.requires_immediate_action` is `True`:

1. The pipeline sets `claim_risk=True` on critical alerts.
2. `EvidenceAssembler` automatically captures the InfluxDB sensor log for the
   affected zone (last 24 hours) and links it to the SSCC being processed.
3. The excursion time window and peak temperature are recorded in the
   `ChainOfCustodyRecord.storage_custody` point.
4. This evidence is included in the dispute letter if a claim is filed.
