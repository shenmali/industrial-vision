"""PLC client factory. Picks driver from config dict."""
from __future__ import annotations

from industrial_vision.plc.base import PLCClient
from industrial_vision.plc.opcua_client import OPCUAClient
from industrial_vision.plc.pymodbus_client import PyModbusClient
from industrial_vision.plc.snap7_client import SiemensS7Client


def build_plc_client(cfg: dict) -> PLCClient:
    driver = cfg.get("driver", "pymodbus")
    if driver == "pymodbus":
        return PyModbusClient(
            host=cfg.get("host", "127.0.0.1"),
            port=int(cfg.get("port", 5020)),
            slave_id=int(cfg.get("slave_id", 1)),
            timeout=float(cfg.get("timeout", 2.0)),
        )
    if driver == "opcua":
        return OPCUAClient()
    if driver == "snap7":
        return SiemensS7Client()
    raise ValueError(f"Unknown PLC driver: {driver}")
