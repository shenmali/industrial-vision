"""Simulated Modbus TCP PLC for development and demos."""
from __future__ import annotations

import argparse
import logging

from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusServerContext,
    ModbusSlaveContext,
)
from pymodbus.server import StartTcpServer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated Modbus TCP PLC")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5020)
    args = parser.parse_args()

    block = ModbusSequentialDataBlock(0, [0] * 200)
    context = ModbusServerContext(slaves=ModbusSlaveContext(hr=block, co=block), single=True)
    log.info("Starting Modbus TCP sim on %s:%d", args.host, args.port)
    StartTcpServer(context=context, address=(args.host, args.port))


if __name__ == "__main__":
    main()
