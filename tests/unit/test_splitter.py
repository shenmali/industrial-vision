from pathlib import Path

import pytest

from industrial_vision.data.splitter import SplitConfig, Splitter, check_leakage


@pytest.fixture
def fake_classification_dataset(tmp_path: Path) -> Path:
    base = tmp_path / "data" / "processed" / "classification"
    for cls in ("good", "defect_a", "defect_b"):
        d = base / cls
        d.mkdir(parents=True)
        for i in range(20):
            (d / f"{i:03d}.png").touch()
    return base


def test_splitter_split_ratios(fake_classification_dataset: Path) -> None:
    cfg = SplitConfig(train=0.7, val=0.15, test=0.15, seed=42)
    sp = Splitter(fake_classification_dataset)
    manifest = sp.split(cfg)
    assert sum(manifest.split_counts.values()) == 60  # 3 classes * 20


def test_splitter_no_leakage(fake_classification_dataset: Path) -> None:
    cfg = SplitConfig(train=0.7, val=0.15, test=0.15, seed=42)
    sp = Splitter(fake_classification_dataset)
    manifest = sp.split(cfg)
    overlaps = check_leakage(manifest)
    assert overlaps == 0


def test_split_config_validates_sum() -> None:
    with pytest.raises(ValueError):
        SplitConfig(train=0.5, val=0.2, test=0.2)
