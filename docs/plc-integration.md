# PLC Integration

IndustrialVision talks to PLCs over **Modbus TCP** by default. The protocol
is abstracted behind a `PLCClient` interface so swapping to OPC UA or
Siemens S7 is a config change in `configs/plc.yaml`.

## Register map (Modbus)

| Address | Type | Name | Meaning |
|---------|------|------|---------|
| 0 | Coil | HEARTBEAT | Toggles 1 Hz |
| 0 | Coil | TRIGGER | Inference complete |
| 1 | Holding | REJECT | 0=ok, 1=reject |
| 2 | Holding | CONFIDENCE | ×10000 (uint16, 0–10000) |
| 10 | Holding | DEFECT_CODE | uint16 enum |

## Simulated PLC (development)

```bash
uv run industrial-vision plc-sim --port 5020
```

Or via Docker Compose:

```bash
docker compose up plc-sim
```

## Real PLC

### Siemens S7 (via `python-snap7`)

```bash
uv sync --extra plc-snap7
```

Edit `configs/plc.yaml`:

```yaml
driver: snap7
host: 192.168.0.10
rack: 0
slot: 1
```

(Implementation is stubbed — extend `src/industrial_vision/plc/snap7_client.py`
to read/write your specific DB layout.)

### OPC UA (via `asyncua`)

```bash
uv sync --extra plc-opcua
```

Edit `configs/plc.yaml`:

```yaml
driver: opcua
endpoint: opc.tcp://192.168.0.10:4840
```

(Implementation is stubbed — extend `src/industrial_vision/plc/opcua_client.py`
to bind to your specific node IDs.)

## Verifying connectivity

```bash
uv run python -c "
from industrial_vision.plc.factory import build_plc_client
client = build_plc_client({'driver': 'pymodbus', 'host': '127.0.0.1', 'port': 5020})
client.connect()
client.heartbeat()
print('OK')
client.close()
"
```
