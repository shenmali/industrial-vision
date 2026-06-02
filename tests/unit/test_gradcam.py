import torch

from industrial_vision.models.classifier.efficientnet import DefectClassifier
from industrial_vision.models.heatmap.gradcam import GradCAM


def test_gradcam_heatmap_shape() -> None:
    model = DefectClassifier(num_classes=3, pretrained=False)
    model.eval()
    cam = GradCAM(model)
    heatmap = cam(torch.randn(1, 3, 224, 224))
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_gradcam_uses_specified_class_idx() -> None:
    model = DefectClassifier(num_classes=3, pretrained=False)
    model.eval()
    cam = GradCAM(model)
    h1 = cam(torch.randn(1, 3, 224, 224), class_idx=0)
    h2 = cam(torch.randn(1, 3, 224, 224), class_idx=1)
    assert h1.shape == h2.shape
