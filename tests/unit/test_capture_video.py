from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from industrial_vision.inference.capture.gstreamer import GStreamerCapture
from industrial_vision.inference.capture.v4l2 import V4L2Capture


@pytest.fixture
def fake_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_v4l2_yields_frames(fake_frame: np.ndarray) -> None:
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, fake_frame)
        mock_cap_cls.return_value = mock_cap
        cap = V4L2Capture("/dev/video0")
        frame = next(iter(cap))
        assert frame.shape == (480, 640, 3)


def test_gstreamer_yields_frames(fake_frame: np.ndarray) -> None:
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, fake_frame)
        mock_cap_cls.return_value = mock_cap
        cap = GStreamerCapture("v4l2src device=/dev/video0 ! fakesink")
        frame = next(iter(cap))
        assert frame.shape == (480, 640, 3)


def test_v4l2_raises_on_open_failure() -> None:
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap
        with pytest.raises(RuntimeError):
            V4L2Capture("/dev/none")
