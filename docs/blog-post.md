# From Rule-Based Vision to Self-Supervised Defect Detection on the Factory Floor

## Outline

1. **Hook** — The hidden cost of rule-based vision in modern factories.
2. **Why anomaly detection** — Unsupervised models learn "good" and flag anything else.
3. **Architecture walkthrough** — 5 layers, why each exists.
4. **Model choices** — PatchCore + EfficientAD ensemble, EfficientNet-B0, Grad-CAM.
5. **Edge deployment** — Why Jetson, why TensorRT FP16, what the latency budget buys us.
6. **PLC integration** — Modbus TCP, register map, why Modbus over OPC UA.
7. **Observability** — Prometheus, Grafana, what to alert on.
8. **Reproducibility** — DVC, Docker Compose, single command to demo.
9. **What I'd do next** — Active learning, multi-camera, cloud training.

---

## Draft

(To be filled in. Until the project is benchmarked on real data, the post is
intentionally a skeleton — numbers will be inserted after the first training
run. See `docs/benchmark.md` for target metrics.)
