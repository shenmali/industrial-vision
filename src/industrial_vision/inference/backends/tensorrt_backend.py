"""TensorRT inference backend for Jetson. Classifier is JIT/TensorRT-compiled; anomaly stays PyTorch."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from industrial_vision.data.augment import build_eval_transform
from industrial_vision.models.anomaly.efficientad import EfficientAD
from industrial_vision.models.anomaly.ensemble import AnomalyEnsemble
from industrial_vision.models.anomaly.patchcore import PatchCore


class TensorRTBackend:
    def __init__(self, engine_path: str | Path, num_classes: int) -> None:
        self.engine_path = Path(engine_path)
        self.engine = torch.jit.load(str(self.engine_path))
        self.engine.eval()
        self.patchcore = PatchCore()
        self.efficientad = EfficientAD()
        self.anomaly_ensemble = AnomalyEnsemble(self.patchcore, self.efficientad)
        self.num_classes = num_classes

    def warmup(self, sample_batch: torch.Tensor) -> None:
        self.patchcore.fit(sample_batch)
        self.efficientad.fit(sample_batch)

    def predict(self, frame: np.ndarray) -> dict[str, object]:
        tensor = build_eval_transform()(image=frame)["image"].unsqueeze(0)
        with torch.no_grad():
            logits = self.engine(tensor)
            probs = torch.softmax(logits, dim=-1)
            conf, idx = probs.max(dim=-1)
        anomaly_score = self.anomaly_ensemble.predict(tensor)
        return {
            "anomaly_score": float(anomaly_score),
            "defect_code": int(idx.item()),
            "confidence": float(conf.item()),
            "heatmap": None,
        }
