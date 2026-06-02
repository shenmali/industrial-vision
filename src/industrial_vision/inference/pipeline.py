"""Pipeline orchestrator: backend + decision policy, runs one frame at a time."""
from __future__ import annotations

import numpy as np

from industrial_vision.inference.backends.pytorch_backend import PyTorchBackend
from industrial_vision.inference.decision import Decision, DecisionPolicy


class Pipeline:
    def __init__(self, backend: PyTorchBackend, policy_anomaly: float, policy_conf: float) -> None:
        self.backend = backend
        self.policy = DecisionPolicy(policy_anomaly, policy_conf)

    def run_frame(self, frame: np.ndarray) -> Decision:
        out = self.backend.predict(frame)
        return self.policy.decide(
            anomaly_score=out["anomaly_score"],
            classifier_conf=out["confidence"],
            defect_code=int(out["defect_code"]),
        )
