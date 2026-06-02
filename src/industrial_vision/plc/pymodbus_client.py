"""PyModbus TCP client for Modbus-TCP PLCs."""
from __future__ import annotations

from pymodbus.client import ModbusTcpClient

from industrial_vision.plc.base import PLCClient, PLCConnectionError

REG_REJECT = 1
REG_CONFIDENCE = 2
REG_DEFECT_CODE = 10
REG_HEARTBEAT = 0
COIL_TRIGGER = 0


class PyModbusClient(PLCClient):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5020,
        slave_id: int = 1,
        timeout: float = 2.0,
    ) -> None:
        self.host = host
        self.port = port
        self.slave_id = slave_id
        self.client = ModbusTcpClient(host, port=port, timeout=timeout)

    def connect(self) -> None:
        if not self.client.connect():
            raise PLCConnectionError(f"Cannot connect to Modbus {self.host}:{self.port}")

    def close(self) -> None:
        self.client.close()

    def write_reject(self, reject: bool, defect_code: int, confidence: float) -> None:
        if not 0 <= defect_code <= 0xFFFF:
            raise ValueError("defect_code out of range")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        self.client.write_coil(COIL_TRIGGER, True, slave=self.slave_id)
        self.client.write_register(REG_REJECT, 1 if reject else 0, slave=self.slave_id)
        self.client.write_register(REG_CONFIDENCE, int(confidence * 10000), slave=self.slave_id)
        self.client.write_register(REG_DEFECT_CODE, int(defect_code), slave=self.slave_id)
        self.client.write_coil(COIL_TRIGGER, False, slave=self.slave_id)

    def read_trigger(self) -> bool:
        result = self.client.read_coils(COIL_TRIGGER, 1, slave=self.slave_id)
        if result.isError():
            return False
        return bool(result.bits[0])

    def heartbeat(self) -> None:
        result = self.client.read_coils(REG_HEARTBEAT, 1, slave=self.slave_id)
        if result.isError():
            return
        self.client.write_coil(REG_HEARTBEAT, not result.bits[0], slave=self.slave_id)
