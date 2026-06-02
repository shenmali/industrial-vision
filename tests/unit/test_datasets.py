import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import pytest

from industrial_vision.data.datasets import AnomalyDataset, ClassificationDataset


def _write_png(path: Path, h: int = 16, w: int = 16) -> None:
    arr = (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(path), arr)


@pytest.fixture
def fake_classification_dir(tmp_path: Path) -> Path:
    base = tmp_path / "cls"
    for cls in ("good", "defect"):
        d = base / cls
        d.mkdir(parents=True)
        for i in range(3):
            _write_png(d / f"img_{i}.png")
    return base


def test_classification_dataset_length(fake_classification_dir: Path) -> None:
    ds = ClassificationDataset(fake_classification_dir)
    assert len(ds) == 6


def test_classification_dataset_item_shape(fake_classification_dir: Path) -> None:
    ds = ClassificationDataset(fake_classification_dir)
    item = ds[0]
    assert "image" in item and "label" in item
    assert item["label"] in {0, 1}
    assert item["image"].shape[0] == 3


def test_classification_dataset_train_uses_augment(fake_classification_dir: Path) -> None:
    ds = ClassificationDataset(fake_classification_dir, train=True)
    assert len(ds) == 6


@pytest.fixture
def fake_anomaly_dirs(tmp_path: Path) -> dict[str, Path]:
    good = tmp_path / "good"
    good.mkdir()
    test_good = tmp_path / "test" / "good"
    test_good.mkdir(parents=True)
    test_bad = tmp_path / "test" / "broken"
    test_bad.mkdir(parents=True)
    for d, n in [(good, 5), (test_good, 3), (test_bad, 4)]:
        for i in range(n):
            _write_png(d / f"{i}.png")
    return {"good": good, "test": tmp_path / "test"}


def test_anomaly_dataset_train_length(fake_anomaly_dirs: dict[str, Path]) -> None:
    ds = AnomalyDataset(good_dir=fake_anomaly_dirs["good"], train=True)
    assert len(ds) == 5


def test_anomaly_dataset_test_has_labels(fake_anomaly_dirs: dict[str, Path]) -> None:
    ds = AnomalyDataset(
        good_dir=fake_anomaly_dirs["good"], test_dir=fake_anomaly_dirs["test"], train=False
    )
    assert len(ds) == 7
    labels = [ds[i]["label"] for i in range(len(ds))]
    assert sum(labels) == 4
