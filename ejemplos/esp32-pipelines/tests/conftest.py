"""Hace importables los módulos de pipelib/ desde los tests (imports planos)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipelib"))
