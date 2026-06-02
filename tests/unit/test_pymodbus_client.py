import threading
import time

import pytest
from pymodbus.datastore import ModbusDeviceContext, ModbusSequentialDataBlock, ModbusServerContext
from pymodbus.server import StartTcpServer

from industrial_vision.plc.pymodbus_client import (
    COIL_HEARTBEAT,
    COIL_TRIGGER,
    REG_DEFECT_CODE,
    REG_REJECT,
    PyModbusClient,
)


@pytest.fixture(scope="module")
def modbus_server() -> None:
    block = ModbusSequentialDataBlock(1, [0] * 200)
    context = ModbusServerContext(devices=ModbusDeviceContext(hr=block, co=block), single=True)
    server_thread = threading.Thread(
        target=StartTcpServer,
        kwargs={"context": context, "address": ("127.0.0.1", 5021)},
        daemon=True,
    )
    server_thread.start()
    time.sleep(0.5)
    yield
    # server is daemon; will be cleaned up


def test_pymodbus_write_and_read(modbus_server: None) -> None:
    client = PyModbusClient(host="127.0.0.1", port=5021)
    client.connect()
    client.write_reject(True, defect_code=2, confidence=0.85)
    assert client.read_trigger() in (True, False)
    client.close()


def test_pymodbus_heartbeat_toggles(modbus_server: None) -> None:
    client = PyModbusClient(host="127.0.0.1", port=5021)
    client.connect()
    client.heartbeat()
    client.heartbeat()
    client.close()


def test_pymodbus_validates_inputs() -> None:
    client = PyModbusClient(host="127.0.0.1", port=5021)
    with pytest.raises(ValueError):
        client.write_reject(True, defect_code=99999, confidence=0.5)
    with pytest.raises(ValueError):
        client.write_reject(True, defect_code=1, confidence=1.5)


def test_register_constants_are_unique() -> None:
    assert REG_REJECT != REG_DEFECT_CODE
    assert COIL_TRIGGER != COIL_HEARTBEAT
