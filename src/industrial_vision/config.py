"""YAML config loader backed by OmegaConf."""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def load_config(path: str | Path) -> object:
    """Load a YAML config file and return an OmegaConf object.

    Raises:
        FileNotFoundError: if the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return OmegaConf.load(p)
