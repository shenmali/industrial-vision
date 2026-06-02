# Architecture

This is the executive summary. The full spec lives at
`docs/superpowers/specs/2026-06-02-industrial-vision-design.md`.

## 5 layers

1. **PLC integration** — Modbus TCP, abstracted behind a `PLCClient` protocol.
   Other drivers (OPC UA, Siemens S7) are stubbed and can be added without
   touching the rest of the system.
2. **Data plane** — MVTec AD, VisA, and user-supplied webcam captures, versioned
   with DVC. Splits are stratified and leakage-checked.
3. **Model layer** — 3-stage pipeline:
   - **Anomaly ensemble** (PatchCore + EfficientAD) flags defects.
   - **EfficientNet-B0** classifies defect type.
   - **Grad-CAM** localizes it on the image.
4. **Inference engine** — async pipeline with PyTorch (PC) or TensorRT
   (Jetson) backend.
5. **Observability** — Prometheus, Grafana, JSON logs.

## Frame lifecycle

```
camera -> capture -> preprocess -> [anomaly?] -> [classify+heatmap] -> decision -> PLC write -> metrics
```

A frame is REJECTed iff `anomaly_score >= 0.5` AND `classifier_confidence >= 0.7`.
Thresholds live in `configs/policy.yaml`.

## Modbus register map

| Address | Type | Name | Meaning |
|---------|------|------|---------|
| 0 | Coil | TRIGGER | Inference complete (toggled by `write_reject`) |
| 1 | Coil | HEARTBEAT | Toggles 1 Hz (by `heartbeat`) |
| 1 | Holding | REJECT | 0=ok, 1=reject |
| 2 | Holding | CONFIDENCE | ×10000 (uint16) |
| 10 | Holding | DEFECT_CODE | uint16 enum |

## File map

- `src/industrial_vision/data/` — registry, splitter, augment, datasets.
- `src/industrial_vision/models/` — anomaly (PatchCore, EfficientAD, ensemble),
  classifier (EfficientNet-B0), heatmap (Grad-CAM).
- `src/industrial_vision/inference/` — capture (file, V4L2, GStreamer),
  backends (PyTorch, TensorRT), decision, pipeline.
- `src/industrial_vision/plc/` — base protocol, pymodbus client, OPC UA / S7 stubs, factory.
- `src/industrial_vision/observability/` — Prometheus metrics, JSON logs.
- `src/industrial_vision/api/` — FastAPI app exposing `/health` and `/metrics`.

## Why these choices

- **Modbus over OPC UA**: Modbus is the de-facto standard in factory-floor
  vision, lower friction than OPC UA for this use case.
- **PatchCore + EfficientAD ensemble**: complementary — PatchCore uses a
  memory bank, EfficientAD uses teacher-student distillation. Averaging
  gives more robust anomaly scores.
- **EfficientNet-B0 over ResNet50**: better accuracy/MAdds ratio, well-suited
  for embedded deployment.
- **Grad-CAM over segmentation**: zero extra annotation cost (uses classifier's
  last conv layer), good-enough localization for the vitrin.
