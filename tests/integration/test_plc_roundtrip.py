import subprocess
import sys
import time
from pathlib import Path

import pytest

from industrial_vision.plc.factory import build_plc_client


@pytest.fixture(scope="module")
def plc_server() -> None:
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent.parent.parent / "deployment/plc_sim/server.py"),
         "--host", "127.0.0.1", "--port", "5022"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2.0)
    yield
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_modbus_roundtrip(plc_server: None) -> None:
    cfg = {"driver": "pymodbus", "host": "127.0.0.1", "port": 5022}
    client = build_plc_client(cfg)
    client.connect()
    client.write_reject(True, defect_code=3, confidence=0.9)
    assert client.read_trigger() in (True, False)
    client.heartbeat()
    client.close()
