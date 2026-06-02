from industrial_vision.plc.factory import build_plc_client
from industrial_vision.plc.opcua_client import OPCUAClient
from industrial_vision.plc.pymodbus_client import PyModbusClient
from industrial_vision.plc.snap7_client import SiemensS7Client


def test_factory_builds_pymodbus() -> None:
    client = build_plc_client({"driver": "pymodbus", "host": "127.0.0.1", "port": 5020})
    assert isinstance(client, PyModbusClient)


def test_factory_builds_opcua() -> None:
    client = build_plc_client({"driver": "opcua"})
    assert isinstance(client, OPCUAClient)


def test_factory_builds_snap7() -> None:
    client = build_plc_client({"driver": "snap7"})
    assert isinstance(client, SiemensS7Client)


def test_factory_unknown_driver_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        build_plc_client({"driver": "modbus_proprietary"})
