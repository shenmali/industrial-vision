import pytest

from industrial_vision.inference.decision import Decision, DecisionPolicy


def test_decision_policy_rejects_when_both_above_threshold() -> None:
    p = DecisionPolicy(anomaly_threshold=0.5, confidence_threshold=0.7)
    d = p.decide(anomaly_score=0.8, classifier_conf=0.9, defect_code=1)
    assert isinstance(d, Decision)
    assert d.reject is True
    assert d.defect_code == 1


def test_decision_policy_passes_when_low_confidence() -> None:
    p = DecisionPolicy(anomaly_threshold=0.5, confidence_threshold=0.7)
    d = p.decide(anomaly_score=0.8, classifier_conf=0.6, defect_code=1)
    assert d.reject is False
    assert d.defect_code == 0


def test_decision_policy_passes_when_low_anomaly() -> None:
    p = DecisionPolicy(anomaly_threshold=0.5, confidence_threshold=0.7)
    d = p.decide(anomaly_score=0.3, classifier_conf=0.95, defect_code=2)
    assert d.reject is False


def test_decision_policy_rejects_at_thresholds() -> None:
    p = DecisionPolicy(anomaly_threshold=0.5, confidence_threshold=0.7)
    d = p.decide(anomaly_score=0.5, classifier_conf=0.7, defect_code=3)
    assert d.reject is True


def test_decision_policy_validates_thresholds() -> None:
    with pytest.raises(ValueError):
        DecisionPolicy(anomaly_threshold=1.5, confidence_threshold=0.7)
    with pytest.raises(ValueError):
        DecisionPolicy(anomaly_threshold=0.5, confidence_threshold=-0.1)


def test_severity_is_anomaly_times_confidence() -> None:
    p = DecisionPolicy()
    d = p.decide(0.4, 0.8, 1)
    assert abs(d.severity - 0.32) < 1e-6
