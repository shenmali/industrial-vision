from fastapi.testclient import TestClient

from industrial_vision.api.fastapi_app import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metrics_endpoint() -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "iv_frame_latency_seconds" in r.text or "iv_defect_total" in r.text


def test_predict_endpoint_smoke() -> None:
    r = client.get("/predict")
    assert r.status_code == 200
    assert "note" in r.json()
