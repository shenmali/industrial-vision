# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

IndustrialVision — defect detection for production lines: unsupervised anomaly
detection → defect classification → pixel-level localization, deployed on Jetson
and wired to a PLC over Modbus TCP. Python 3.11, `src/` layout, managed with `uv`.

## Environment & commands

Dependencies are split: core deps install without PyTorch; `torch`/`torchvision`
live in the `pytorch` dependency-group and are pulled in separately. This split is
load-bearing — see Testing below.

```bash
uv sync                      # core deps only (no torch)
uv sync --group pytorch      # add torch/torchvision (needed for models/backends)
```

Lint / format / types (run before committing; enforced in CI and pre-commit):

```bash
ruff check .                 # ruff check . --fix to autofix
ruff format .                # ruff format --check . in CI
mypy src                     # only src/industrial_vision is type-checked
pre-commit run --all-files   # ruff + ruff-format + mypy
```

Run:

```bash
docker compose up                                          # full stack (app, plc-sim, prometheus, grafana)
uv run industrial-vision serve                             # FastAPI on :8000 (/docs, /health, /metrics)
uv run industrial-vision plc-sim --host 0.0.0.0 --port 5020
uv run industrial-vision run --config configs/inference.yaml
```

## Testing

Most model/backend code needs PyTorch, which is **not** in the default install. The
main `CI` workflow (`.github/workflows/ci.yml`) therefore runs only an explicit
subset of torch-free tests, while the full suite runs nightly via `uv run pytest`
in `full-tests.yml` after installing the `pytorch` group. The recent git history is
almost entirely fixes to this test partitioning — when adding a test, decide which
side of the split it belongs on and keep the `ci.yml` file list in sync.

```bash
# Full suite (requires: uv sync --group pytorch)
uv run pytest -v

# Torch-free subset — ci.yml is the source of truth for the exact file list
PYTHONPATH=src python -m pytest tests/unit/test_decision.py tests/unit/test_config.py ...

# Single test / pattern
uv run pytest tests/unit/test_decision.py
uv run pytest tests/unit/test_decision.py::test_reject_when_both_thresholds_met
uv run pytest -k decision
```

Tests import the package via `tests/conftest.py` (inserts `src/` on `sys.path`) — no
editable install needed. Coverage gate is 70% branch coverage (`pyproject.toml`).
Markers: `integration`, `perf`, `slow`; `asyncio_mode = auto`.

## Architecture

Five layers (full design in `docs/architecture.md` and `docs/superpowers/specs/`):

1. **PLC** (`plc/`) — `PLCClient` ABC (`base.py`) with a `pymodbus` implementation;
   OPC UA and Siemens S7 are stubs. `factory.build_plc_client(cfg)` selects the
   driver by `cfg["driver"]`. Modbus register map is documented in
   `docs/architecture.md` (coils TRIGGER/HEARTBEAT, holding REJECT/CONFIDENCE/DEFECT_CODE).
2. **Data** (`data/`) — `DatasetRegistry` downloads/inspects MVTec AD & VisA;
   `splitter` (stratified, leakage-checked), `augment` (albumentations), `datasets`.
   Data is DVC-tracked and gitignored (`data/raw|processed|splits`).
3. **Models** (`models/`) — 3-stage: anomaly **ensemble** (PatchCore + EfficientAD,
   0.5/0.5 weighted in `anomaly/ensemble.py`) → **EfficientNet-B0** classifier →
   **Grad-CAM** localization. Note: `GradCAM` hooks the layer named
   `backbone.features.7`, coupling it to `DefectClassifier`'s internal structure.
4. **Inference** (`inference/`) — `capture/` (file, V4L2, GStreamer),
   `backends/` (`PyTorchBackend` for PC, `tensorrt_backend` for Jetson),
   `decision.py`, `pipeline.py`. `PyTorchBackend.predict` runs the full 3-stage
   chain; `Pipeline.run_frame` applies the decision policy to its output.
5. **Observability** (`observability/`) — Prometheus metrics + JSON logging;
   exposed via FastAPI `api/fastapi_app.py`.

**Frame lifecycle:** `camera → capture → preprocess → anomaly → classify+heatmap →
decision → PLC write → metrics`.

**Core decision rule** (`inference/decision.py`): a frame is REJECTed iff
`anomaly_score >= anomaly_threshold` AND `classifier_conf >= confidence_threshold`
(defaults 0.5 / 0.7, set in `configs/policy.yaml`). This is the central business
logic — change thresholds in config, not code.

## State of wiring (important)

Components are individually implemented and unit-tested, but the end-to-end loop is
**not yet assembled**. Specifically: `cli.cmd_run` only loads and prints the config;
the FastAPI `/predict` endpoint is a stub; nothing currently drives
capture → pipeline → PLC write → metrics together. When asked to "make it run
end-to-end," expect to write this glue rather than fix existing wiring.

## Gotchas

- **No `uv.lock` is checked in**, but the `Dockerfile` and `full-tests.yml` use
  `uv sync --frozen`, which fails without a lockfile. Run `uv lock` before relying
  on those, or drop `--frozen`.
- Config is plain YAML loaded via OmegaConf (`config.load_config`) — Hydra/pydantic
  are dependencies but config loading is currently just `OmegaConf.load`.
- Ruff `line-length` is 100 but `E501` is ignored, so long lines don't fail lint.
- Pretrained EfficientNet weights download on first model use (EfficientAD teacher),
  so model tests need network access — another reason they're in the nightly suite.
