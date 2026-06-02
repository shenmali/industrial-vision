"""Decision policy: deterministic rule combining anomaly score and classifier confidence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    reject: bool
    defect_code: int
    anomaly_score: float
    confidence: float
    severity: float


class DecisionPolicy:
    def __init__(self, anomaly_threshold: float = 0.5, confidence_threshold: float = 0.7) -> None:
        if not 0.0 <= anomaly_threshold <= 1.0:
            raise ValueError("anomaly_threshold must be in [0,1]")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0,1]")
        self.anomaly_threshold = anomaly_threshold
        self.confidence_threshold = confidence_threshold

    def decide(
        self,
        anomaly_score: float,
        classifier_conf: float,
        defect_code: int,
    ) -> Decision:
        reject = (anomaly_score >= self.anomaly_threshold) and (
            classifier_conf >= self.confidence_threshold
        )
        severity = float(anomaly_score * classifier_conf)
        return Decision(
            reject=reject,
            defect_code=defect_code if reject else 0,
            anomaly_score=anomaly_score,
            confidence=classifier_conf,
            severity=severity,
        )
