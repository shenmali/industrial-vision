"""Grad-CAM: gradient-weighted class activation map from a named conv layer."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: str = "backbone.features.7") -> None:
        self.model = model.eval()
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        layer = dict(self.model.named_modules())[target_layer]
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: object, _in: object, out: torch.Tensor) -> None:
        self._activations = out.detach()

    def _save_gradient(self, _module: object, _in: object, _grad_out: torch.Tensor) -> None:
        self._gradients = _grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> torch.Tensor:
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=-1).item())
        score = logits[0, class_idx]
        score.backward()
        assert self._activations is not None and self._gradients is not None
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu()
        cam_min = float(cam.min())
        cam_max = float(cam.max())
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam
