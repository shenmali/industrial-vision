import numpy as np
import pytest
import torch

from industrial_vision.inference.backends.pytorch_backend import PyTorchBackend
from industrial_vision.inference.pipeline import Pipeline


def test_pipeline_returns_decision_for_synthetic_frame() -> None:
    backend = PyTorchBackend(num_classes=3)
    backend.classifier = backend.classifier  # ensure no pretrained weights in test
    # Replace the classifier head with random weights for deterministic test
    for p in backend.classifier.parameters():
        p.requires_grad_(False)
    backend.patchcore.fit(torch.randn(4, 3, 224, 224))
    backend.efficientad.fit(torch.randn(4, 3, 224, 224), epochs=1)
    pipe = Pipeline(backend, policy_anomaly=0.3, policy_conf=0.5)
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    out = pipe.run_frame(frame)
    assert hasattr(out, "reject")
    assert hasattr(out, "anomaly_score")
    assert hasattr(out, "confidence")
    assert hasattr(out, "defect_code")
    assert 0.0 <= out.anomaly_score <= 1.0
    assert 0.0 <= out.confidence <= 1.0
