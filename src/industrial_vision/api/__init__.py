"""FastAPI HTTP service exposing health, metrics, and prediction endpoint."""
from industrial_vision.api.fastapi_app import app

__all__ = ["app"]
