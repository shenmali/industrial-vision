"""CLI entrypoint for industrial-vision."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from industrial_vision.config import load_config
from industrial_vision.observability.logging_config import configure_logging


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    configure_logging(level="INFO", log_file="logs/app.log")
    log = logging.getLogger("iv.cli")
    log.info("Starting industrial-vision", extra={"config": str(args.config)})
    print(f"Loaded config: {cfg}")
    return 0


def cmd_plc_sim(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "deployment"))
    from plc_sim.server import main as plc_main

    sys.argv = ["plc-sim", "--host", args.host, "--port", str(args.port)]
    plc_main()
    return 0


def cmd_serve(_args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("industrial_vision.api.fastapi_app:app", host="0.0.0.0", port=8000)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="industrial-vision")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--config", type=Path, required=True)
    p_run.set_defaults(func=cmd_run)

    p_plc = sub.add_parser("plc-sim")
    p_plc.add_argument("--host", default="0.0.0.0")
    p_plc.add_argument("--port", type=int, default=5020)
    p_plc.set_defaults(func=cmd_plc_sim)

    p_srv = sub.add_parser("serve")
    p_srv.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
