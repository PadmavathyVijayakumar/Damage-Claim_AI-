"""
Mindwave AI Damage Claims Platform — FastAPI application entry point.

API surface:
  /health                    — liveness + readiness probe
  /api/v1/inspections        — dock camera pipeline results
  /api/v1/claims             — claims lifecycle (open, process, close)
  /api/v1/cold-chain         — cold-chain zone status and alerts
  /api/v1/analytics          — dashboard data and root cause reports
  /api/v1/integrations/wms   — WMS connector endpoints

Auth: JWT bearer token (HS256).
Observability: OpenTelemetry traces + Prometheus metrics + structlog JSON.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from config.settings import get_settings

# ── Structured logging setup ─────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
)
DAMAGE_EVENTS = Counter(
    "damage_events_total",
    "Total damage detection events",
    ["severity", "zone"],
)
CLAIMS_PROCESSED = Counter(
    "claims_processed_total",
    "Claims processed by outcome",
    ["action"],
)


# ── Application lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hook."""
    settings = get_settings()
    logger.info("Starting Mindwave AI Platform", env=settings.environment)
    yield
    logger.info("Shutting down Mindwave AI Platform")


# ── FastAPI app ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Mindwave AI Damage Claims Platform",
        description=(
            "AI-powered warehouse damage detection, cold-chain monitoring, "
            "and intelligent claims processing."
        ),
        version="1.0.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.environment == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Prometheus middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(duration)
        return response

    # ── Health endpoints ──────────────────────────────────────────────────
    @app.get("/health", tags=["ops"])
    async def health():
        return {"status": "ok", "version": "1.0.0"}

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ── Route registration ────────────────────────────────────────────────
    from src.api.routes import claims, cold_chain, inspections, analytics

    app.include_router(inspections.router, prefix="/api/v1/inspections", tags=["inspections"])
    app.include_router(claims.router, prefix="/api/v1/claims", tags=["claims"])
    app.include_router(cold_chain.router, prefix="/api/v1/cold-chain", tags=["cold-chain"])
    app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_config=None,  # structlog handles logging
    )
