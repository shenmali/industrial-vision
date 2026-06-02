"""Frame capture backends: file replay, V4L2 webcam, GStreamer pipelines."""
from industrial_vision.inference.capture.file import FileCapture
from industrial_vision.inference.capture.gstreamer import GStreamerCapture
from industrial_vision.inference.capture.v4l2 import V4L2Capture

__all__ = ["FileCapture", "V4L2Capture", "GStreamerCapture"]
