"""EfficientNet-B0 based defect type classifier with ImageNet pretrained backbone."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class DefectClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[int, float]:
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        return int(idx.item()), float(conf.item())
