import json
import logging

from industrial_vision.observability.logging_config import configure_logging


def test_configure_logging_emits_json(tmp_path) -> None:
    log_file = tmp_path / "app.log"
    configure_logging(level="INFO", log_file=str(log_file))
    log = logging.getLogger("iv.test")
    log.info("hello", extra={"frame_id": 42})
    line = log_file.read_text().strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["message"] == "hello"
    assert payload["frame_id"] == 42


def test_configure_logging_respects_level(tmp_path) -> None:
    log_file = tmp_path / "app.log"
    configure_logging(level="WARNING", log_file=str(log_file))
    log = logging.getLogger("iv.test")
    log.info("info_msg")
    log.warning("warn_msg")
    lines = log_file.read_text().strip().splitlines()
    levels = [json.loads(line)["level"] for line in lines]
    assert "INFO" not in levels
    assert "WARNING" in levels
