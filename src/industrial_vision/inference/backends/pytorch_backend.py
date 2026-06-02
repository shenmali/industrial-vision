"""PyTorch inference backend combining anomaly ensemble, classifier, and Grad-CAM."""

from __future__ import annotations

import numpy as np
import torch

from industrial_vision.data.augment import build_eval_transform
from industrial_vision.models.anomaly.efficientad import EfficientAD
from industrial_vision.models.anomaly.ensemble import AnomalyEnsemble
from industrial_vision.models.anomaly.patchcore import PatchCore
from industrial_vision.models.classifier.efficientnet import DefectClassifier
from industrial_vision.models.heatmap.gradcam import GradCAM


class PyTorchBackend:
    def __init__(self, num_classes: int, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.classifier = DefectClassifier(num_classes=num_classes).to(self.device)
        self.patchcore = PatchCore(device=self.device)
        self.efficientad = EfficientAD(device=self.device)
        self.anomaly_ensemble = AnomalyEnsemble(self.patchcore, self.efficientad)
        self.gradcam = GradCAM(self.classifier)

    def warmup(self, sample_batch: torch.Tensor) -> None:
        self.patchcore.fit(sample_batch)
        self.efficientad.fit(sample_batch)

    def predict(self, frame: np.ndarray) -> dict[str, float | int | torch.Tensor | None]:
        tensor = build_eval_transform()(image=frame)["image"].unsqueeze(0).to(self.device)
        anomaly_score = self.anomaly_ensemble.predict(tensor)
        label, conf = self.classifier.predict(tensor)
        heatmap = self.gradcam(tensor, class_idx=label)
        return {
            "anomaly_score": float(anomaly_score),
            "defect_code": int(label),
            "confidence": float(conf),
            "heatmap": heatmap,
        }
