"""Prometheus metrics registry used across the application."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class MetricsRegistry:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.frame_latency = Histogram(
            "iv_frame_latency_seconds",
            "Frame processing latency",
            labelnames=("stage",),
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self.registry,
        )
        self.frame_throughput = Gauge(
            "iv_frame_throughput_fps",
            "Throughput in frames per second",
            registry=self.registry,
        )
        self.defect_total = Counter(
            "iv_defect_total",
            "Total defect events",
            labelnames=("defect_type", "decision"),
            registry=self.registry,
        )
        self.modbus_errors = Counter(
            "iv_modbus_errors_total",
            "Modbus errors",
            labelnames=("op",),
            registry=self.registry,
        )
        self.modbus_roundtrip = Histogram(
            "iv_modbus_roundtrip_seconds",
            "Modbus round-trip latency",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
            registry=self.registry,
        )
        self.model_score = Histogram(
            "iv_model_score",
            "Model score distributions",
            labelnames=("model",),
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry,
        )
