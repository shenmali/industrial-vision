"""Weighted average of PatchCore and EfficientAD anomaly scores."""

from __future__ import annotations

import torch

from industrial_vision.models.anomaly.efficientad import EfficientAD
from industrial_vision.models.anomaly.patchcore import PatchCore


class AnomalyEnsemble:
    def __init__(
        self,
        patchcore: PatchCore,
        efficientad: EfficientAD,
        weights: tuple[float, float] = (0.5, 0.5),
    ) -> None:
        if len(weights) != 2:
            raise ValueError("weights must be a 2-tuple")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("weights must sum to 1.0")
        self.patchcore = patchcore
        self.efficientad = efficientad
        self.weights = weights

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> float:
        s_pc, _ = self.patchcore.predict(image)
        s_ad, _ = self.efficientad.predict(image)
        return float(self.weights[0] * s_pc + self.weights[1] * s_ad)
