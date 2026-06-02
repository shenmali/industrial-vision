import torch

from industrial_vision.models.anomaly.patchcore import PatchCore


def test_patchcore_predicts_anomaly_score_in_range() -> None:
    model = PatchCore(coreset_subsample=4)
    model.fit(torch.randn(8, 3, 224, 224))
    score, dists = model.predict(torch.randn(1, 3, 224, 224))
    assert 0.0 <= score <= 1.0
    assert len(dists) > 0


def test_patchcore_requires_fit() -> None:
    model = PatchCore()
    try:
        model.predict(torch.randn(1, 3, 224, 224))
    except AssertionError:
        return
    raise AssertionError("predict() should require fit() to be called first")
