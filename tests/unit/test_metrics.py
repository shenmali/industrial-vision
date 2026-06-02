from prometheus_client import CollectorRegistry, generate_latest

from industrial_vision.observability.metrics import MetricsRegistry


def test_metrics_registry_exposes_expected_names() -> None:
    reg = MetricsRegistry()
    reg.frame_latency.labels(stage="inference").observe(0.025)
    reg.defect_total.labels(defect_type="scratch", decision="reject").inc()
    out = generate_latest(reg.registry).decode()
    assert "iv_frame_latency_seconds" in out
    assert "iv_defect_total" in out


def test_metrics_registry_uses_custom_registry() -> None:
    cr = CollectorRegistry()
    reg = MetricsRegistry(registry=cr)
    reg.frame_throughput.set(15.5)
    out = generate_latest(cr).decode()
    assert "iv_frame_throughput_fps" in out


def test_modbus_metrics_present() -> None:
    reg = MetricsRegistry()
    reg.modbus_errors.labels(op="write").inc()
    reg.modbus_roundtrip.observe(0.003)
    out = generate_latest(reg.registry).decode()
    assert "iv_modbus_errors_total" in out
    assert "iv_modbus_roundtrip_seconds" in out
