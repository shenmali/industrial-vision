# IndustrialVision

**Production-line defect detection that reasons like a modern computer-vision system and speaks the language of the factory floor (Modbus TCP).**

[![CI](https://github.com/shenmali/industrial-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/shenmali/industrial-vision/actions/workflows/ci.yml)
[![Full Tests](https://github.com/shenmali/industrial-vision/actions/workflows/full-tests.yml/badge.svg)](https://github.com/shenmali/industrial-vision/actions/workflows/full-tests.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Lint: Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

IndustrialVision inspects parts coming off a production line, decides whether each
one is defective, classifies the defect, shows *where* it is on the image, and
writes the verdict back to a PLC — the same controller that already drives the
reject actuator. It runs on a PC during development and on an NVIDIA Jetson at the
edge, and it ships with a simulated PLC and a full observability stack so you can
watch the whole loop end to end on a laptop.

---

## Why

Classic machine vision on a production line is **rule-based**: hand-tuned
thresholds, blob detectors, and templates. It works until the lighting changes, a
new defect appears, or a supplier swaps a material — then someone has to re-tune
it. That maintenance burden is the real cost.

This project takes a different position:

- **Learn "normal," flag the rest.** Unsupervised anomaly detection means you only
  need images of *good* parts to get started — no exhaustive defect labelling.
- **Explain the decision.** A reject is only useful on the floor if an operator can
  see why. Every reject comes with a defect class *and* a pixel-level heatmap.
- **Fit into what already exists.** No new SCADA layer. The system talks **Modbus
  TCP**, so it drops in next to the PLC that's already there.
- **Run at the edge, observably.** Targeted at Jetson with TensorRT, with
  Prometheus/Grafana metrics so you can prove latency and reject rates, not guess.

## What it does

- 🔍 **Anomaly detection** — a PatchCore + EfficientAD **ensemble** scores how
  unusual each frame is, trained only on good samples.
- 🏷️ **Defect classification** — an EfficientNet-B0 head names the defect type.
- 🌡️ **Localization** — Grad-CAM produces a heatmap over the suspect region, with
  zero extra annotation cost.
- 🔌 **PLC integration** — writes reject / confidence / defect-code over Modbus TCP
  (real or simulated), with a heartbeat coil.
- 📊 **Observability** — Prometheus metrics + structured JSON logs, surfaced through
  a FastAPI service and a ready-made Grafana dashboard.

**Frame lifecycle:**

```
camera → capture → preprocess → anomaly → classify + heatmap → decision → PLC write → metrics
```

---

## How it works — the technical approach

The system is built as **five layers**, each replaceable behind a small interface.

| Layer | What it solves | How |
|-------|----------------|-----|
| **PLC** (`plc/`) | Talk to factory hardware | `PLCClient` ABC + `pymodbus` driver; OPC UA / Siemens S7 stubbed behind a `factory` selector |
| **Data** (`data/`) | Get clean, leak-free training data | `DatasetRegistry` (MVTec AD, VisA), stratified leakage-checked splits, albumentations augments, DVC-tracked |
| **Models** (`models/`) | Detect → classify → localize | 3-stage pipeline (below) |
| **Inference** (`inference/`) | Run fast on PC and Jetson | `capture/` (file, V4L2, GStreamer) → `PyTorchBackend` / `tensorrt_backend` → `decision` → `pipeline` |
| **Observability** (`observability/`) | Prove it works in production | Prometheus metrics + JSON logging via FastAPI |

### The 3-stage model pipeline

1. **Anomaly ensemble** (`models/anomaly/`) — two complementary detectors, averaged
   `0.5 / 0.5`:
   - **PatchCore** — mid-level ResNet-18 features form a *memory bank* of "normal"
     patches; the anomaly score is the nearest-neighbour distance to that bank.
   - **EfficientAD** — a frozen EfficientNet-B0 *teacher* and a *student* trained to
     mimic it on good images; the score is the teacher↔student feature MSE.

   They fail differently (memory bank vs. teacher–student distillation), so
   averaging is more robust than either alone.

2. **Classifier** (`models/classifier/`) — **EfficientNet-B0** with an ImageNet
   backbone and a fresh head, returning `(defect_code, confidence)`.

3. **Localization** (`models/heatmap/`) — **Grad-CAM** on the classifier's last
   conv block (`backbone.features.7`) turns the prediction into a heatmap.

### The decision rule

The business logic is one deliberately simple, auditable rule
(`inference/decision.py`), driven by config — never code:

> A frame is **REJECT**ed iff `anomaly_score ≥ anomaly_threshold` **AND**
> `classifier_confidence ≥ confidence_threshold` (defaults `0.5` / `0.7` in
> [`configs/policy.yaml`](configs/policy.yaml)).

Requiring *both* a high anomaly score and a confident classification suppresses
false rejects — the metric that actually matters on a line.

### Why these choices

- **Modbus over OPC UA** — the de-facto standard for factory-floor vision; far less
  integration friction.
- **PatchCore + EfficientAD** — complementary anomaly methods; the ensemble is
  steadier than either.
- **EfficientNet-B0 over ResNet-50** — better accuracy-per-MAdd, which is what
  matters on embedded hardware.
- **Grad-CAM over segmentation** — reuses the classifier's own gradients, so
  localization costs zero extra labels.

### Modbus register map

| Address | Type | Name | Meaning |
|---------|------|------|---------|
| 0 | Coil | `TRIGGER` | Inference complete |
| 1 | Coil | `HEARTBEAT` | Toggles ~1 Hz (liveness) |
| 1 | Holding | `REJECT` | `0` = ok, `1` = reject |
| 2 | Holding | `CONFIDENCE` | confidence × 10000 (uint16) |
| 10 | Holding | `DEFECT_CODE` | defect-type enum |

---

## How it's built — engineering methodology

The *system* solves a vision problem; the *repo* is built to keep solving it
reliably. The conventions below are deliberate.

- **Partitioned dependencies.** Core deps install **without** PyTorch; `torch` /
  `torchvision` live in a separate `pytorch` group. Lint, config, decision, PLC, and
  data-plane logic are testable in seconds without pulling GPU-scale wheels — torch
  is only added where it's genuinely needed.
- **Two-tier testing.** A torch-free subset runs on **every push/PR** for fast
  feedback ([`ci.yml`](.github/workflows/ci.yml)); the **full suite** (with torch,
  downloading pretrained weights) runs **nightly**
  ([`full-tests.yml`](.github/workflows/full-tests.yml)). Coverage gate: **70 %
  branch**. Tests import the package via a `conftest.py` path shim — no editable
  install required.
- **Reproducible by construction.** Managed with `uv`; a committed `uv.lock` plus
  `--frozen` installs in Docker and CI mean the same dependency graph everywhere.
- **Data discipline.** Datasets are **DVC-tracked** (and git-ignored); splits are
  **stratified and leakage-checked** with a fixed seed.
- **Observability from day one.** Metrics and structured logs aren't an
  afterthought — they're wired through the FastAPI service and a provisioned Grafana
  dashboard.
- **Enforced quality gates.** `ruff` (lint + format), `mypy` (typed `src/`), and
  `pre-commit` run locally and in CI.

### Systematic debugging — a worked example

The methodology shows up in how problems get fixed. The nightly **Full Tests**
workflow was failing every run. Rather than guess, the fix followed the loop:

1. **Read the evidence** — `gh run` logs showed every failure was the *same*
   workflow, dying in ~10 s at dependency install — not a test failure.
2. **Find the root cause** — `uv sync --frozen` with **no committed `uv.lock`**.
3. **Reproduce locally, step by step** — which surfaced two *more* latent issues:
   an undeclared `httpx` test dependency, and an `@torch.no_grad()` decorator that
   broke `EfficientAD.fit`'s backward pass.
4. **Fix the causes, then verify** — all 70 tests green locally, then green in CI.

→ See [#1](https://github.com/shenmali/industrial-vision/pull/1) for the full
diagnosis and fix.

---

## Quickstart

```bash
git clone https://github.com/shenmali/industrial-vision
cd industrial-vision
docker compose up
```

Then open:

| Service | URL | Notes |
|---------|-----|-------|
| API (FastAPI) | <http://localhost:8000/docs> | `/health`, `/metrics`, `/predict` |
| Grafana | <http://localhost:3000> | `admin` / `admin` |
| Prometheus | <http://localhost:9090> | |
| PLC simulator | `plc-sim:5020` | Modbus TCP, reachable from the app container |

## Local development

Dependencies are split — install core first, add PyTorch only when you need the
models/backends:

```bash
uv sync                  # core deps (no torch)
uv sync --group pytorch  # add torch/torchvision
```

Run the pieces:

```bash
uv run industrial-vision serve                             # FastAPI on :8000
uv run industrial-vision plc-sim --host 0.0.0.0 --port 5020
uv run industrial-vision run --config configs/inference.yaml
```

Quality + tests:

```bash
ruff check . && ruff format --check . && mypy src   # lint, format, types
pre-commit run --all-files

uv run pytest -v                       # full suite (requires the pytorch group)
uv run pytest -k decision              # a single pattern
```

## Deployment

- **Jetson / TensorRT** — export with
  [`deployment/jetson/tensorrt_export.py`](deployment/jetson/tensorrt_export.py)
  and run under the provided `systemd` unit; the inference engine swaps
  `PyTorchBackend` for the TensorRT backend.
- **Full stack** — `docker compose up` brings up the app, the PLC simulator,
  Prometheus, and Grafana together.

## Benchmarks

**Targets** on MVTec AD (measured numbers will be published in
[`docs/benchmark.md`](docs/benchmark.md) after the first training run):

| Model | AUROC target | Latency (Jetson) |
|-------|--------------|------------------|
| PatchCore | ≥ 0.97 | < 30 ms |
| EfficientAD | ≥ 0.95 | < 35 ms |
| Ensemble (0.5 / 0.5) | ≥ 0.98 | < 50 ms |

## Project status

Honest snapshot — this is an actively built showcase, not a finished product:

- ✅ **Implemented & unit-tested:** PLC client (Modbus) + simulator, dataset
  registry / splitter / augments, the full 3-stage model stack, PyTorch backend,
  capture sources, the decision policy, metrics, and the FastAPI `/health` +
  `/metrics` endpoints.
- 🚧 **Not yet wired end-to-end:** the `run` command currently loads config only,
  `/predict` is a stub, and the `capture → pipeline → PLC → metrics` loop isn't
  assembled yet. Benchmarks are targets, not measured. OPC UA / Siemens S7 drivers
  are stubs.

See [`CLAUDE.md`](CLAUDE.md) and [`docs/architecture.md`](docs/architecture.md) for
the deeper map.

## Repository layout

```
src/industrial_vision/   # application code (plc, data, models, inference, observability, api)
configs/                 # YAML config (policy, model, data, plc, inference)
deployment/              # PLC sim, Jetson, Prometheus, Grafana
tests/                   # unit + integration
docs/                    # architecture, benchmarks, design specs
```

## License

[MIT](LICENSE)
