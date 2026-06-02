import pytest
import torch

from industrial_vision.models.anomaly.efficientad import EfficientAD
from industrial_vision.models.anomaly.ensemble import AnomalyEnsemble
from industrial_vision.models.anomaly.patchcore import PatchCore


def test_ensemble_combines_scores() -> None:
    pc = PatchCore(coreset_subsample=4)
    pc.fit(torch.randn(4, 3, 224, 224))
    ad = EfficientAD()
    ad.fit(torch.randn(4, 3, 224, 224), epochs=1)
    ens = AnomalyEnsemble(pc, ad, weights=(0.5, 0.5))
    score = ens.predict(torch.randn(1, 3, 224, 224))
    assert 0.0 <= score <= 1.0


def test_ensemble_validates_weights() -> None:
    pc = PatchCore()
    ad = EfficientAD()
    with pytest.raises(ValueError):
        AnomalyEnsemble(pc, ad, weights=(0.3, 0.3))
    with pytest.raises(ValueError):
        AnomalyEnsemble(pc, ad, weights=(0.3,))  # type: ignore[arg-type]
