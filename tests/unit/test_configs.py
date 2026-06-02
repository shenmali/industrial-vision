from pathlib import Path


def test_load_data_config() -> None:
    from industrial_vision.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "data.yaml")
    assert cfg.data_root == "data"
    assert cfg.splits.train == 0.7


def test_load_model_config() -> None:
    from industrial_vision.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "model.yaml")
    assert cfg.classifier.num_classes == 5
    assert cfg.anomaly.patchcore.backbone == "resnet18"


def test_load_policy_config() -> None:
    from industrial_vision.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "policy.yaml")
    assert cfg.anomaly_threshold == 0.5
    assert cfg.confidence_threshold == 0.7


def test_load_plc_config() -> None:
    from industrial_vision.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "plc.yaml")
    assert cfg.driver == "pymodbus"
    assert cfg.port == 5020


def test_load_inference_config() -> None:
    from industrial_vision.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "inference.yaml")
    assert cfg.backend == "pytorch"
