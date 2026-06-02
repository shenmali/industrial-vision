# Jetson Deployment

This directory contains deployment artifacts for NVIDIA Jetson Orin.

## Steps

1. Flash JetPack 5.1+ on the device.
2. Install Python 3.11 and uv on-device.
3. `uv sync --group pytorch` in the project root.
4. Install Jetson-specific extras (torch-tensorrt, pycuda) via `pip install`.
5. Export classifier to TensorRT:
   `python deployment/jetson/tensorrt_export.py --checkpoint checkpoints/classifier.pt --output checkpoints/classifier.ts --num-classes 5`
5. Run: `uv run industrial-vision run --config configs/inference.yaml`
6. Install systemd unit:
   `sudo cp deployment/jetson/systemd/industrial_vision.service /etc/systemd/system/`
   `sudo systemctl enable --now industrial_vision`
7. Tail logs: `journalctl -u industrial_vision -f`
