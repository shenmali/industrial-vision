"""FastAPI app: /health, /metrics, /predict."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest

from industrial_vision.observability.metrics import MetricsRegistry

app = FastAPI(title="IndustrialVision", version="0.1.0")
_metrics = MetricsRegistry()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> bytes:
    return generate_latest(_metrics.registry)


@app.get("/predict")
def predict() -> dict[str, str]:
    return {"note": "wire pipeline here"}
