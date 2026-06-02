import pytest

from industrial_vision.plc.base import PLCConnectionError
from industrial_vision.plc.opcua_client import OPCUAClient
from industrial_vision.plc.snap7_client import SiemensS7Client


def test_opcua_connect_raises_not_implemented() -> None:
    client = OPCUAClient()
    with pytest.raises(PLCConnectionError):
        client.connect()


def test_opcua_methods_raise() -> None:
    client = OPCUAClient()
    with pytest.raises(NotImplementedError):
        client.write_reject(True, 1, 0.5)
    with pytest.raises(NotImplementedError):
        client.read_trigger()
    with pytest.raises(NotImplementedError):
        client.heartbeat()


def test_s7_connect_raises_not_implemented() -> None:
    client = SiemensS7Client()
    with pytest.raises(PLCConnectionError):
        client.connect()


def test_s7_methods_raise() -> None:
    client = SiemensS7Client()
    with pytest.raises(NotImplementedError):
        client.write_reject(True, 1, 0.5)
    with pytest.raises(NotImplementedError):
        client.read_trigger()
    with pytest.raises(NotImplementedError):
        client.heartbeat()
