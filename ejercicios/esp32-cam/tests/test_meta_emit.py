"""Tests para el sidecar model_data.meta.json emitido junto a model_data.h."""

import json
from pathlib import Path

import pytest

from esp32cam_utils import write_model_data_meta


def test_write_model_data_meta_emits_json(tmp_path: Path):
    meta = {
        "num_classes": 10,
        "input_w": 32,
        "input_h": 32,
        "input_channels": 1,
        "mean": [124],
        "std": [58],
        "classes": [f"c{i}" for i in range(10)],
    }
    write_model_data_meta(tmp_path / "model_data.meta.json", meta)
    assert (tmp_path / "model_data.meta.json").exists()
    loaded = json.loads((tmp_path / "model_data.meta.json").read_text())
    assert loaded == meta


def test_write_model_data_meta_rejects_incomplete(tmp_path: Path):
    with pytest.raises(KeyError, match="num_classes"):
        write_model_data_meta(tmp_path / "x.json", {"input_w": 32})
