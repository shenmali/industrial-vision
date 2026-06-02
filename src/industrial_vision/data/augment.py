"""Albumentations pipelines: training-time augmentations and deterministic eval transform."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_SIZE = 224


def build_train_transform(size: int = DEFAULT_SIZE) -> A.Compose:
    """Training-time augmentation: flips, brightness, blur, ImageNet normalization."""
    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_eval_transform(size: int = DEFAULT_SIZE) -> A.Compose:
    """Deterministic eval transform: resize + normalize, no random ops."""
    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
