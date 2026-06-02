"""Lightweight PatchCore implementation.

Uses mid-level ResNet features, builds a memory bank of patch features,
and predicts anomaly score as the distance to the nearest neighbor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class PatchCore:
    def __init__(
        self,
        backbone: str = "resnet18",
        coreset_subsample: int = 16,
        device: torch.device | None = None,
    ) -> None:
        self.backbone_name = backbone
        self.coreset_subsample = coreset_subsample
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _Backbone().to(self.device).eval()
        self.memory_bank: torch.Tensor | None = None

    @torch.no_grad()
    def _features(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.model(batch.to(self.device))
        b, c, h, w = feats.shape
        return feats.permute(0, 2, 3, 1).reshape(b * h * w, c)

    @torch.no_grad()
    def fit(self, images: torch.Tensor) -> None:
        feats = self._features(images)
        if feats.shape[0] > self.coreset_subsample:
            idx = torch.randperm(feats.shape[0])[: self.coreset_subsample]
            feats = feats[idx]
        self.memory_bank = F.normalize(feats, dim=1)

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> tuple[float, list[float]]:
        assert self.memory_bank is not None, "Call fit() before predict()"
        feats = self._features(image)
        feats = F.normalize(feats, dim=1)
        dists = torch.cdist(feats, self.memory_bank).min(dim=1).values
        top_k = max(1, len(dists) // 100)
        score = float(dists.topk(top_k).values.mean())
        return min(score * 10.0, 1.0), [float(d) for d in dists.cpu()]
