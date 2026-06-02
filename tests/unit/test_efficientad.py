import torch

from industrial_vision.models.anomaly.efficientad import EfficientAD


def test_efficientad_predicts_in_range() -> None:
    model = EfficientAD()
    model.fit(torch.randn(4, 3, 224, 224), epochs=1)
    score, map_ = model.predict(torch.randn(1, 3, 224, 224))
    assert 0.0 <= score <= 1.0
    assert map_.shape == (224, 224)
    assert map_.min() >= 0.0
