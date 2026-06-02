"""Stratified dataset splitter with leakage detection."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SplitConfig:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    seed: int = 42

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {total}")


@dataclass
class SplitManifest:
    dataset_root: str
    items: list[dict[str, str]] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))


class Splitter:
    """Stratified per-(split, class) splitter; reproducible via seed."""

    def __init__(self, dataset_root: str | Path) -> None:
        self.root = Path(dataset_root)

    def split(self, cfg: SplitConfig) -> SplitManifest:
        random.seed(cfg.seed)
        manifest = SplitManifest(dataset_root=str(self.root))
        for cls_dir in self.root.iterdir():
            if not cls_dir.is_dir():
                continue
            files = sorted(p.name for p in cls_dir.iterdir() if p.is_file())
            random.shuffle(files)
            n = len(files)
            n_train = int(n * cfg.train)
            n_val = int(n * cfg.val)
            train_files = files[:n_train]
            val_files = files[n_train : n_train + n_val]
            test_files = files[n_train + n_val :]
            for f in train_files:
                manifest.items.append({"class": cls_dir.name, "file": f, "subset": "train"})
            for f in val_files:
                manifest.items.append({"class": cls_dir.name, "file": f, "subset": "val"})
            for f in test_files:
                manifest.items.append({"class": cls_dir.name, "file": f, "subset": "test"})
        manifest.split_counts = dict(Counter(item["subset"] for item in manifest.items))
        return manifest


def check_leakage(manifest: SplitManifest) -> int:
    """Return the number of (class, file) keys that appear in more than one subset."""
    seen: dict[tuple[str, str], set[str]] = {}
    for item in manifest.items:
        key = (item["class"], item["file"])
        subset = item["subset"]
        seen.setdefault(key, set()).add(subset)
    return sum(1 for subsets in seen.values() if len(subsets) > 1)
