# IndustrialVision

End-to-end **defect detection on production lines** — unsupervised anomaly
detection, defect classification, and pixel-level localization, deployed on
Jetson and wired to a real (or simulated) PLC over **Modbus TCP**.

![demo](docs/demo.gif)

## Why

Rule-based machine vision is fragile and expensive to maintain. This project
shows how a modern CV stack — **PatchCore + EfficientAD + EfficientNet-B0 +
Grad-CAM** — can replace it, while still speaking the same industrial
protocols that PLCs expect.

## Architecture (5 layers)

1. **PLC Integration** — Modbus TCP, real or simulated (`pyModbusTCP`).
2. **Data Plane** — MVTec AD + VisA + custom webcam, DVC tracked.
3. **Model Layer** — 3-stage pipeline (anomaly → classify → localize).
4. **Inference Engine** — async pipeline, PC (PyTorch) / Jetson (TensorRT).
5. **Observability** — Prometheus metrics + Grafana dashboard + JSON logs.

See [`docs/architecture.md`](docs/architecture.md) for details.

## Frame lifecycle

```
camera → capture → preprocess → [anomaly?] → [classify + heatmap] → decision → PLC write → metrics
```

A frame is REJECTed iff `anomaly_score >= 0.5` AND `classifier_confidence >= 0.7`.
Thresholds live in `configs/policy.yaml`.

## Quick start

```bash
git clone https://github.com/<you>/CompVi
cd CompVi
docker compose up
```

Then open:

- API: <http://localhost:8000/docs>
- Grafana: <http://localhost:3000> (admin / admin)
- Prometheus: <http://localhost:9090>
- PLC sim is reachable on `plc-sim:5020` from the app container.

## Benchmark (MVTec AD)

Final numbers are reported in `docs/benchmark.md` after the first training run.
Targets:

| Model | AUROC target | Latency (Jetson) |
|-------|--------------|------------------|
| PatchCore | ≥ 0.97 | < 30 ms |
| EfficientAD | ≥ 0.95 | < 35 ms |
| Ensemble (0.5/0.5) | ≥ 0.98 | < 50 ms |

## Repo layout

```
src/industrial_vision/    # application code
configs/                  # Hydra YAML
deployment/               # PLC sim, Jetson, Prometheus, Grafana
tests/                    # unit, integration
docs/                     # architecture, blog, LinkedIn drafts
```

## Notebooks

`notebooks/` contains EDA, training, and visualization notebooks (Jupyter).

## License

MIT
