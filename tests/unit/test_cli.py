import sys

import pytest

# import the cli as a module; the conftest adds src/ to sys.path
from industrial_vision import cli as cli_mod


def test_main_help_exits_cleanly(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as e:
        sys.argv = ["industrial-vision", "--help"]
        cli_mod.main()
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "industrial-vision" in out


def test_main_unknown_command_exits_nonzero(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as e:
        sys.argv = ["industrial-vision", "bogus"]
        cli_mod.main()
    assert e.value.code != 0
