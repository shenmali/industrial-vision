import numpy as np

from industrial_vision.data.augment import build_eval_transform, build_train_transform


def test_train_transform_changes_image() -> None:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    out = build_train_transform()(image=img)["image"]
    assert out.shape == (3, 224, 224)


def test_eval_transform_deterministic() -> None:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    t = build_eval_transform()
    a = t(image=img)["image"]
    b = t(image=img)["image"]
    assert np.array_equal(a, b)


def test_train_transform_produces_normalized_tensor() -> None:
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    out = build_train_transform()(image=img)["image"]
    assert out.min() < 0
    assert out.max() > 0
