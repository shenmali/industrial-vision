import torch

from industrial_vision.models.classifier.efficientnet import DefectClassifier


def test_classifier_forward_shape() -> None:
    model = DefectClassifier(num_classes=5, pretrained=False)
    model.eval()
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 5)


def test_classifier_predict_returns_label_and_confidence() -> None:
    model = DefectClassifier(num_classes=3, pretrained=False)
    label, conf = model.predict(torch.randn(1, 3, 224, 224))
    assert 0 <= label < 3
    assert 0.0 <= conf <= 1.0
