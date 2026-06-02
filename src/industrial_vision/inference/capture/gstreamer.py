"""GStreamer pipeline capture (CSI cameras on Jetson or arbitrary GStreamer sources)."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


class GStreamerCapture(Iterator[np.ndarray]):
    def __init__(self, pipeline: str) -> None:
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open GStreamer pipeline: {pipeline}")

    def __next__(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise StopIteration
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __del__(self) -> None:
        if hasattr(self, "cap"):
            self.cap.release()
