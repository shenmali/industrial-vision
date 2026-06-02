# Benchmark

TBD after first training run on MVTec AD.

## Target metrics

| Model | AUROC target | Jetson latency p50 | Jetson latency p99 |
|-------|--------------|--------------------|--------------------|
| PatchCore | ≥ 0.97 | < 30 ms | < 50 ms |
| EfficientAD | ≥ 0.95 | < 35 ms | < 50 ms |
| Ensemble (0.5/0.5) | ≥ 0.98 | < 50 ms | < 80 ms |
| EfficientNet-B0 (top-1) | ≥ 0.95 | < 10 ms | < 20 ms |

## Modbus round-trip

| Operation | Latency target |
|-----------|----------------|
| `write_reject` | < 10 ms p99 |
| `read_trigger` | < 5 ms p99 |
| `heartbeat` toggle | < 5 ms p99 |

## How to reproduce

```bash
# 1. Train anomaly ensemble
uv run python -m industrial_vision.models.anomaly.patchcore \
    --data-dir data/raw/mvtec_ad --category bottle

# 2. Train classifier
uv run python -m industrial_vision.models.classifier.efficientnet \
    --data-dir data/processed/classification

# 3. Run benchmark
uv run pytest tests/perf -v
```

After the first real run, replace this file with measured numbers.
