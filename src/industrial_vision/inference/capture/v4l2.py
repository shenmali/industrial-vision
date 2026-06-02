"""V4L2 USB webcam capture backend."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


class V4L2Capture(Iterator[np.ndarray]):
    def __init__(self, device: str = "/dev/video0", width: int = 640, height: int = 480) -> None:
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open {device}")

    def __next__(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise StopIteration
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __del__(self) -> None:
        if hasattr(self, "cap"):
            self.cap.release()
