import numpy as np
from pathlib import Path

from industrial_vision.inference.capture.file import FileCapture


def test_file_capture_yields_frames(tmp_path: Path) -> None:
    for i in range(3):
        (tmp_path / f"img_{i}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + np.zeros((10, 10, 3), dtype=np.uint8).tobytes()
        )
    cap = FileCapture(tmp_path, loop=False)
    frames = list(cap)
    assert len(frames) == 3
    assert all(f.shape == (10, 10, 3) for f in frames)


def test_file_capture_stops_without_loop(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + np.zeros((10, 10, 3), dtype=np.uint8).tobytes()
    )
    cap = FileCapture(tmp_path, loop=False)
    next(cap)
    with pytest.raises(StopIteration):
        next(cap)


def test_file_capture_loops(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + np.zeros((10, 10, 3), dtype=np.uint8).tobytes()
    )
    cap = FileCapture(tmp_path, loop=True)
    a = next(cap)
    b = next(cap)
    assert a.shape == b.shape


def test_file_capture_raises_on_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FileCapture(tmp_path)
