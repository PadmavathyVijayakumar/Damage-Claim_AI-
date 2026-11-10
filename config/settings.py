"""
Application settings — loaded from environment / .env file via pydantic-settings.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / "config" / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ──────────────────────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"

    # ── Database ─────────────────────────────────────────────────────
    database_url: str = "postgresql://mindwave:mindwave@localhost:5432/damageclaim"

    # ── Cache ─────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── InfluxDB (cold-chain) ─────────────────────────────────────────
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = ""
    influxdb_org: str = "mindwave"
    influxdb_bucket: str = "cold_chain"

    # ── Kafka / MQTT ──────────────────────────────────────────────────
    kafka_bootstrap_servers: str = "localhost:9092"
    mqtt_broker_host: str = "localhost"
    mqtt_broker_port: int = 1883

    # ── S3 evidence store ─────────────────────────────────────────────
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_evidence_bucket: str = "mindwave-evidence"

    # ── Vision models ─────────────────────────────────────────────────
    yolo_model_path: str = "models/weights/yolov9_damage.pt"
    classifier_model_path: str = "models/weights/efficientnet_damage.pt"
    openai_api_key: str = ""
    vllm_confidence_threshold: float = 0.72

    # ── Cold-chain models ─────────────────────────────────────────────
    lstm_model_path: str = "models/weights/lstm_temp.pt"
    lstm_ae_model_path: str = "models/weights/lstm_ae.pt"
    ann_risk_model_path: str = "models/weights/ann_risk.h5"
    excursion_alert_horizon_min: int = 60

    # ── WMS connectors ────────────────────────────────────────────────
    manhattan_api_base_url: str = ""
    manhattan_client_id: str = ""
    manhattan_client_secret: str = ""

    sap_ewm_host: str = ""
    sap_btp_token_url: str = ""
    sap_client_id: str = ""
    sap_client_secret: str = ""

    blue_yonder_api_base: str = ""
    blue_yonder_client_id: str = ""
    blue_yonder_client_secret: str = ""

    oracle_wms_host: str = ""
    oracle_oic_base: str = ""
    oracle_client_id: str = ""
    oracle_client_secret: str = ""

    # ── ERP connectors ────────────────────────────────────────────────
    sap_s4_host: str = ""
    oracle_erp_host: str = ""

    # ── API security ──────────────────────────────────────────────────
    api_secret_key: str = Field(default="dev-secret-change-in-prod", min_length=16)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
