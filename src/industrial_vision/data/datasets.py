"""Dataset classes: classification and unsupervised anomaly detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset

from industrial_vision.data.augment import build_eval_transform, build_train_transform


def _read_image(path: Path) -> np.ndarray:
    """Read an image from disk as RGB. Raises ValueError on decode failure."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class ClassificationDataset(Dataset):
    """ImageFolder-style classification dataset with per-index transform."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        class_names: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.transform = build_train_transform() if train else build_eval_transform()
        self.class_names = class_names or sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.samples: list[tuple[Path, int]] = []
        for cls_name in self.class_names:
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.samples.append((img_path, self.class_names.index(cls_name)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path, label = self.samples[idx]
        img = _read_image(path)
        out = self.transform(image=img)
        return {"image": out["image"], "label": label, "path": str(path)}


class AnomalyDataset(Dataset):
    """Unsupervised anomaly detection dataset. `good` is the only training class."""

    def __init__(
        self,
        good_dir: str | Path,
        test_dir: str | Path | None = None,
        train: bool = True,
    ) -> None:
        self.good_dir = Path(good_dir)
        self.test_dir = Path(test_dir) if test_dir else None
        self.train = train
        self.transform = build_train_transform() if train else build_eval_transform()
        if train:
            self.samples = [p for p in sorted(self.good_dir.iterdir()) if p.is_file()]
        else:
            assert self.test_dir is not None, "test_dir required when train=False"
            self.samples = [p for p in sorted(self.test_dir.iterdir()) if p.is_file()]
            self.labels: list[int] = [0 if p.parent.name == "good" else 1 for p in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.samples[idx]
        img = _read_image(path)
        out = self.transform(image=img)
        item: dict[str, Any] = {"image": out["image"], "path": str(path)}
        if not self.train:
            item["label"] = self.labels[idx]
        return item
