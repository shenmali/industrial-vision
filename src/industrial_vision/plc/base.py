"""Abstract base for PLC clients. All drivers must implement this interface."""
from __future__ import annotations

from abc import ABC, abstractmethod


class PLCConnectionError(RuntimeError):
    pass


class PLCClient(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def write_reject(self, reject: bool, defect_code: int, confidence: float) -> None: ...

    @abstractmethod
    def read_trigger(self) -> bool: ...

    @abstractmethod
    def heartbeat(self) -> None: ...
