from pathlib import Path

import pytest

from industrial_vision.data.registry import SUPPORTED_DATASETS, DatasetRegistry


@pytest.fixture
def fake_data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    (root / "raw" / "mvtec_ad" / "bottle" / "train" / "good").mkdir(parents=True)
    (root / "raw" / "mvtec_ad" / "bottle" / "test" / "good").mkdir(parents=True)
    (root / "raw" / "mvtec_ad" / "bottle" / "test" / "broken_large").mkdir(parents=True)
    (root / "raw" / "mvtec_ad" / "bottle" / "ground_truth" / "broken_large").mkdir(parents=True)
    (root / "raw" / "mvtec_ad" / "bottle" / "train" / "good" / "000.png").touch()
    (root / "raw" / "mvtec_ad" / "bottle" / "test" / "good" / "000.png").touch()
    (root / "raw" / "mvtec_ad" / "bottle" / "test" / "broken_large" / "000.png").touch()
    return root


def test_list_categories(fake_data_root: Path) -> None:
    reg = DatasetRegistry(fake_data_root)
    cats = reg.list_categories("mvtec_ad")
    assert cats == ["bottle"]


def test_count_samples_per_split(fake_data_root: Path) -> None:
    reg = DatasetRegistry(fake_data_root)
    counts = reg.count_samples("mvtec_ad", "bottle")
    assert counts["train/good"] == 1
    assert counts["test/good"] == 1
    assert counts["test/broken_large"] == 1


def test_unknown_dataset_raises(fake_data_root: Path) -> None:
    reg = DatasetRegistry(fake_data_root)
    with pytest.raises(ValueError):
        reg.list_categories("nope")


def test_supported_datasets_has_mvtec_and_visa() -> None:
    assert "mvtec_ad" in SUPPORTED_DATASETS
    assert "visa" in SUPPORTED_DATASETS
