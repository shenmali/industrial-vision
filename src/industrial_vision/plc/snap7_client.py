"""Stub Siemens S7 client. Implement with `python-snap7` when integrating a real PLC."""
from __future__ import annotations

from industrial_vision.plc.base import PLCClient, PLCConnectionError


class SiemensS7Client(PLCClient):
    def connect(self) -> None:
        raise PLCConnectionError("S7 client not yet implemented")

    def close(self) -> None:
        pass

    def write_reject(self, reject: bool, defect_code: int, confidence: float) -> None:
        raise NotImplementedError

    def read_trigger(self) -> bool:
        raise NotImplementedError

    def heartbeat(self) -> None:
        raise NotImplementedError
