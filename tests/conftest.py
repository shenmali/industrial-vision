"""Conftest: ensure src/ is on sys.path so tests can import the package without install."""

import os
import sys
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
