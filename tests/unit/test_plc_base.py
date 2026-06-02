import pytest

from industrial_vision.plc.base import PLCClient, PLCConnectionError


def test_plc_client_is_abstract() -> None:
    with pytest.raises(TypeError):
        PLCClient()  # type: ignore[abstract]


def test_plc_client_subclass_works() -> None:
    class FakePLC(PLCClient):
        def connect(self) -> None:
            return None

        def close(self) -> None:
            return None

        def write_reject(self, reject: bool, defect_code: int, confidence: float) -> None:
            self.last = (reject, defect_code, confidence)

        def read_trigger(self) -> bool:
            return True

        def heartbeat(self) -> None:
            return None

    plc = FakePLC()
    plc.connect()
    plc.write_reject(True, 1, 0.8)
    assert plc.last == (True, 1, 0.8)
    assert plc.read_trigger() is True
