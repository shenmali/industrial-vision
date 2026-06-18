"""Simplified EfficientAD: teacher is frozen, student distills from teacher on good images.

Anomaly score = per-pixel MSE between teacher and student, scaled and clamped.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class _Teacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = m.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class _Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        m = efficientnet_b0(weights=None)
        self.features = m.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class EfficientAD:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = _Teacher().to(self.device).eval()
        self.student = _Student().to(self.device)

    def fit(self, images: torch.Tensor, epochs: int = 1, lr: float = 1e-4) -> None:
        opt = torch.optim.Adam(self.student.parameters(), lr=lr)
        self.student.train()
        for _ in range(epochs):
            opt.zero_grad()
            batch = images.to(self.device)
            with torch.no_grad():
                t = self.teacher(batch)
            s = self.student(batch)
            loss = F.mse_loss(s, t)
            loss.backward()
            opt.step()
        self.student.eval()

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> tuple[float, torch.Tensor]:
        t = self.teacher(image.to(self.device))
        s = self.student(image.to(self.device))
        diff = (t - s) ** 2
        map_ = diff.mean(dim=1, keepdim=True)
        map_ = F.interpolate(map_, size=image.shape[-2:], mode="bilinear", align_corners=False)
        map_ = map_.squeeze().cpu()
        score = float(map_.max())
        return min(score * 5.0, 1.0), map_
