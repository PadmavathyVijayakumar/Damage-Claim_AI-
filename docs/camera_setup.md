# Dock Camera Installation & Configuration Guide

This guide covers hardware selection, physical mounting, network configuration,
and software integration for the inbound dock camera system.

---

## Camera Requirements

### Minimum Specification

| Parameter | Requirement |
|-----------|-------------|
| Resolution | 2 MP (1920 × 1080) minimum; 4 MP (2560 × 1440) recommended |
| Frame rate | 30 fps at full resolution |
| Dynamic range | ≥ 120 dB (WDR / HDR) for mixed dock lighting |
| Low-light sensitivity | F1.6 or better aperture; IR illumination |
| Exposure control | Auto + manual override |
| Interface | RTSP over 10/100/1000 Ethernet; PoE+ (802.3at) |
| IP rating | IP66 or better (dock dust / fork wash-down zones) |
| Operating temperature | -10 °C to 55 °C |
| Compression | H.264 / H.265 (RTSP stream) + JPEG snapshot endpoint |

### Recommended Cameras

- **Axis P1448-LE** (8 MP, 180° FOV, IR, PoE+)
- **Hikvision DS-2CD2T47G2-L** (4 MP, AcuSense, WDR)
- **Bosch FLEXIDOME 5100i** (2 MP, IR, IVA analytics)

> **NVIDIA Jetson edge option:** Any USB3 or MIPI-CSI industrial camera with
> GigE Vision interface can be used when running the YOLO model on a Jetson AGX
> Orin co-located with the camera.

---

## Physical Mounting

### Overhead Dock Door Mount (Recommended)

```
         ┌─────────────────────┐
         │      Dock Door       │
         └─────────────────────┘
              ↑ 3.5–4.5 m
         ┌────────────┐
         │   Camera   │  ← Mounted on structural beam, centred on dock door
         └────────────┘
              ↓ 20–35° tilt
  ┌───────────────────────────────────┐
  │       Pallet in staging zone       │
  └───────────────────────────────────┘
  ← ─────────────── 4–6 m ─────────── →
```

**Mounting rules:**
1. Centre the camera horizontally above the dock door.
2. Mount height: 3.5–4.5 m above floor level.
3. Down-tilt: 20–35° to cover the full pallet footprint.
4. Ensure full pallet coverage (1.2 m × 1.0 m CHEP standard) at capture distance.
5. Avoid back-lighting: orient so dock door opens away from the camera's field of view.

### Side-Mount (Alternative)

Use when overhead mounting is structurally impractical.  Mount at 2.5 m height,
angled 45° inward.  Accuracy may be 5–8% lower for bottom-pallet-tier damage.

---

## Camera Network Setup

### IP Address Assignment

Assign static IPs from a dedicated VLAN (e.g., `10.20.30.0/24`):

| Camera | Dock Door | Static IP |
|--------|-----------|-----------|
| CAM-B1 | Door 1 | 10.20.30.11 |
| CAM-B2 | Door 2 | 10.20.30.12 |
| … | … | … |
| CAM-BN | Door N | 10.20.30.1N |

### RTSP Stream URLs

Most cameras expose the primary stream at:
```
rtsp://<user>:<password>@<camera_ip>:554/stream1
```

JPEG snapshot endpoint (for single-frame capture):
```
http://<camera_ip>/jpg/image.jpg
```

### Firewall Rules

```
SOURCE      DEST         PORT   PROTOCOL  ACTION
API Pods    Camera VLAN  554    TCP        ALLOW   # RTSP
API Pods    Camera VLAN  80     TCP        ALLOW   # HTTP snapshot
Camera VLAN API Pods     any    TCP        DENY    # Cameras do not initiate connections
```

---

## Software Configuration

### Environment Variables

Add to `config/.env`:

```dotenv
# Camera registry (JSON mapping camera_id → RTSP URL)
CAMERA_REGISTRY={"CAM-B1":"rtsp://admin:pass@10.20.30.11:554/stream1","CAM-B2":"rtsp://admin:pass@10.20.30.12:554/stream1"}

# Frame capture settings
CAMERA_CAPTURE_FPS=2          # Frames per second captured from each camera
CAMERA_JPEG_QUALITY=92        # JPEG quality for stored evidence frames (1-100)
CAMERA_FRAME_BUFFER_SECONDS=5 # Buffer held in memory before writing to S3

# Detection trigger
CAMERA_MIN_CONFIDENCE=0.45    # Minimum YOLO confidence to report detection
CAMERA_VLLM_THRESHOLD=0.72    # Confidence below which VLLM review is triggered
```

### Frame Ingestion

The API endpoint `POST /api/v1/inspections/frame` accepts base64-encoded JPEG
frames.  For continuous ingest from live cameras, use the provided ingestion
worker (not included in this release; use any RTSP→HTTP bridge such as
FFmpeg or rtsp-simple-server):

```bash
# Example: capture one frame per second from CAM-B1 and POST to API
ffmpeg -i "rtsp://admin:pass@10.20.30.11:554/stream1" \
  -vf fps=1 -f image2pipe -vcodec mjpeg - \
  | while read -n 65536 frame; do
      echo $frame | base64 | \
      curl -s -X POST http://localhost:8000/api/v1/inspections/frame \
        -H "Content-Type: application/json" \
        -d "{\"camera_id\":\"CAM-B1\",\"zone\":\"B1\",\"frame_b64\":\"$(echo $frame | base64)\"}"
    done
```

### Edge Deployment (Optional)

For < 50 ms latency, deploy YOLO on a co-located NVIDIA Jetson:

1. Export model: `python scripts/deploy_edge.py --model yolo --weights models/weights/yolo_damage.pt --format tensorrt --fp16`
2. Copy `models/weights/yolo_damage.engine` to Jetson.
3. Run the edge inference server (not included; use Triton Inference Server or custom FastAPI worker).
4. POST detection results to the central API's `/inspections/frame` endpoint.

---

## Calibration & Testing

### Calibration Procedure

1. Place a standard CHEP pallet (1.2 m × 1.0 m) in the dock staging zone.
2. Capture 10 frames from each camera.
3. Verify the pallet occupies ≥ 60% of the frame width.
4. Adjust mount angle and zoom until this criterion is met.
5. Run `python scripts/calibrate_cameras.py` (creates bounding-box overlay).

### Acceptance Test

Place the following test items on the pallet and run a live inspection:

| Test Item | Expected Detection |
|-----------|-------------------|
| Torn stretch wrap (20 × 20 cm) | `tear` ≥ 0.70 confidence |
| Crushed carton corner | `crush` ≥ 0.70 confidence |
| Water stain on carton | `wet_damage` ≥ 0.60 confidence |
| Intact pallet, no damage | No damage class ≥ confidence threshold |

Acceptance criteria: all three damage items detected; no false positive on
intact pallet.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Black / no-image frames | RTSP URL wrong or camera offline | Ping camera IP; verify RTSP credentials |
| High false-positive rate | Insufficient lighting | Add IR or LED strip lights (≥ 500 lux at pallet level) |
| SSCC OCR failures | Label too far or tilted | Reduce capture distance; add label-level close-up camera |
| `inference_ms` > 800 ms | YOLO running on CPU | Enable GPU or deploy edge TensorRT model |
| Duplicate detections | Frame rate too high | Lower `CAMERA_CAPTURE_FPS` to 1–2 fps |
