from pathlib import Path

import pytest

from industrial_vision.config import load_config


def test_load_config_returns_dict(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("foo: 1\nbar: baz\n")
    cfg = load_config(cfg_path)
    assert cfg.foo == 1
    assert cfg.bar == "baz"


def test_load_config_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")
