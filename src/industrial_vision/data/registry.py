"""Dataset registry: download and inspect MVTec AD, VisA, and custom webcam data."""

from __future__ import annotations

import tarfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    checksum: str
    categories: tuple[str, ...]


SUPPORTED_DATASETS: dict[str, DatasetSpec] = {
    "mvtec_ad": DatasetSpec(
        name="mvtec_ad",
        url=(
            "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/"
            "download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
        ),
        checksum="",
        categories=(
            "bottle",
            "cable",
            "capsule",
            "carpet",
            "grid",
            "hazelnut",
            "leather",
            "metal_nut",
            "pill",
            "screw",
            "tile",
            "toothbrush",
            "transistor",
            "wood",
            "zipper",
        ),
    ),
    "visa": DatasetSpec(
        name="visa",
        url="https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
        checksum="",
        categories=(
            "candle",
            "capsules",
            "cashew",
            "chewinggum",
            "fryum",
            "macaroni1",
            "macaroni2",
            "pcb1",
            "pcb2",
            "pcb3",
            "pcb4",
            "pipe_fryum",
        ),
    ),
}


class DatasetRegistry:
    """Discover, count, and download industrial defect datasets."""

    def __init__(self, data_root: str | Path) -> None:
        self.root = Path(data_root)
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.splits_dir = self.root / "splits"

    def list_categories(self, dataset: str) -> list[str]:
        """Return sorted category names available locally for the given dataset."""
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        base = self.raw_dir / dataset
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())

    def count_samples(self, dataset: str, category: str) -> dict[str, int]:
        """Count files in `<dataset>/<category>/<split>/<class>/` paths."""
        base = self.raw_dir / dataset / category
        counts: Counter[str] = Counter()
        if not base.exists():
            return {}
        for split in base.iterdir():
            if not split.is_dir():
                continue
            for kind in split.iterdir():
                if kind.is_dir():
                    key = f"{split.name}/{kind.name}"
                    counts[key] = sum(1 for f in kind.iterdir() if f.is_file())
        return dict(counts)

    def download(self, dataset: str, force: bool = False) -> Path:
        """Download and extract a dataset if it is not already present locally."""
        spec = SUPPORTED_DATASETS[dataset]
        target = self.raw_dir / dataset
        if target.exists() and not force:
            return target
        target.mkdir(parents=True, exist_ok=True)
        archive = target / f"{dataset}.tar"
        self._stream_download(spec.url, archive)
        self._extract(archive, target)
        archive.unlink()
        return target

    def _stream_download(self, url: str, dest: Path) -> None:
        with urllib.request.urlopen(url) as response, dest.open("wb") as fh:
            total = int(response.headers.get("Content-Length", 0))
            chunk = 1024 * 1024
            with tqdm(total=total, unit="B", unit_scale=True) as bar:
                while True:
                    data = response.read(chunk)
                    if not data:
                        break
                    fh.write(data)
                    bar.update(len(data))

    def _extract(self, archive: Path, dest: Path) -> None:
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
