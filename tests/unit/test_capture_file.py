from pathlib import Path

import numpy as np
import pytest

from industrial_vision.inference.capture.file import FileCapture


def _write_valid_png(path: Path, h: int = 10, w: int = 10) -> None:
    import cv2

    arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    cv2.imwrite(str(path), arr)


def test_file_capture_yields_frames(tmp_path: Path) -> None:
    for i in range(3):
        _write_valid_png(tmp_path / f"img_{i}.png")
    cap = FileCapture(tmp_path, loop=False)
    frames = list(cap)
    assert len(frames) == 3
    assert all(f.shape == (10, 10, 3) for f in frames)


def test_file_capture_stops_without_loop(tmp_path: Path) -> None:
    _write_valid_png(tmp_path / "a.png")
    cap = FileCapture(tmp_path, loop=False)
    next(cap)
    with pytest.raises(StopIteration):
        next(cap)


def test_file_capture_loops(tmp_path: Path) -> None:
    _write_valid_png(tmp_path / "a.png")
    cap = FileCapture(tmp_path, loop=True)
    a = next(cap)
    b = next(cap)
    assert a.shape == b.shape


def test_file_capture_raises_on_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FileCapture(tmp_path)
