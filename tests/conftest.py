"""Conftest: ensure src/ is on sys.path so tests can import the package without install."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
