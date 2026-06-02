"""Inference backends: PyTorch (PC) and TensorRT (Jetson)."""
from industrial_vision.inference.backends.pytorch_backend import PyTorchBackend
from industrial_vision.inference.backends.tensorrt_backend import TensorRTBackend

__all__ = ["PyTorchBackend", "TensorRTBackend"]
