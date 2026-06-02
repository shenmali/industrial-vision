"""Iterate over image files in a directory. Useful for replay, CI, and demos."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class FileCapture(Iterator[np.ndarray]):
    def __init__(self, directory: str | Path, loop: bool = False) -> None:
        self.directory = Path(directory)
        self.loop = loop
        self.files = sorted(
            p for p in self.directory.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if not self.files:
            raise FileNotFoundError(f"No images found in {self.directory}")
        self._idx = 0

    def __next__(self) -> np.ndarray:
        if self._idx >= len(self.files):
            if self.loop:
                self._idx = 0
            else:
                raise StopIteration
        path = self.files[self._idx]
        self._idx += 1
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to read {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
