from pathlib import Path

import numpy as np
import pytest

from industrial_vision.data.datasets import AnomalyDataset, ClassificationDataset


@pytest.fixture
def fake_classification_dir(tmp_path: Path) -> Path:
    base = tmp_path / "cls"
    for cls in ("good", "defect"):
        d = base / cls
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"img_{i}.png").write_bytes(
                b"\x89PNG\r\n\x1a\n" + np.zeros((10, 10, 3), dtype=np.uint8).tobytes()
            )
    return base


def test_classification_dataset_length(fake_classification_dir: Path) -> None:
    ds = ClassificationDataset(fake_classification_dir)
    assert len(ds) == 6
    item = ds[0]
    assert "image" in item and "label" in item
    assert item["label"] in {0, 1}
    assert item["image"].shape[0] == 3  # C, H, W


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
            (d / f"{i}.png").write_bytes(
                b"\x89PNG\r\n\x1a\n" + np.zeros((10, 10, 3), dtype=np.uint8).tobytes()
            )
    return {"good": good, "test": tmp_path / "test"}


def test_anomaly_dataset_train_length(fake_anomaly_dirs: dict[str, Path]) -> None:
    ds = AnomalyDataset(good_dir=fake_anomaly_dirs["good"], train=True)
    assert len(ds) == 5


def test_anomaly_dataset_test_has_labels(fake_anomaly_dirs: dict[str, Path]) -> None:
    ds = AnomalyDataset(good_dir=fake_anomaly_dirs["good"],
                        test_dir=fake_anomaly_dirs["test"], train=False)
    assert len(ds) == 7
    labels = [ds[i]["label"] for i in range(len(ds))]
    assert sum(labels) == 4  # 4 broken samples
