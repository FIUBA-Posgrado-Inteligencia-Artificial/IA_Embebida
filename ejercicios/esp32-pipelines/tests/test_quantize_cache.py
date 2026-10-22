"""Tests del cache por mtime de script04a-quantize."""

import importlib.util
import json
import os
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "script04a-quantize.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("script04a_quantize", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_artifacts(tmp_path: Path, target: str = "c"):
    src = tmp_path / "m.pth"
    espdl = tmp_path / "m_quantized_ptq.espdl"
    header = tmp_path / "m_quantized_ptq_model_data.h"
    meta = tmp_path / "m_quantized_ptq_model_data.meta.json"
    src.write_bytes(b"pth")
    old = time.time() - 100
    os.utime(src, (old, old))
    espdl.write_bytes(b"espdl")
    header.write_text("#define MODEL_NUM_CLASSES 10\n")
    meta.write_text(json.dumps({"target": target}))
    return src, espdl, header, meta


def test_cache_fresh_when_artifacts_newer_and_target_matches(tmp_path):
    mod = _load_module()
    src, espdl, header, meta = _make_artifacts(tmp_path)
    assert mod.cache_is_fresh(src, espdl, header, meta, "c") is True


def test_cache_stale_when_source_is_newer(tmp_path):
    mod = _load_module()
    src, espdl, header, meta = _make_artifacts(tmp_path)
    now = time.time() + 100
    os.utime(src, (now, now))
    assert mod.cache_is_fresh(src, espdl, header, meta, "c") is False


def test_cache_stale_on_target_mismatch(tmp_path):
    mod = _load_module()
    src, espdl, header, meta = _make_artifacts(tmp_path, target="esp32s3")
    assert mod.cache_is_fresh(src, espdl, header, meta, "c") is False


def test_cache_stale_when_artifact_missing(tmp_path):
    mod = _load_module()
    src, espdl, header, meta = _make_artifacts(tmp_path)
    header.unlink()
    assert mod.cache_is_fresh(src, espdl, header, meta, "c") is False


def test_cache_stale_on_corrupt_sidecar(tmp_path):
    mod = _load_module()
    src, espdl, header, meta = _make_artifacts(tmp_path)
    meta.write_text("{not json")
    assert mod.cache_is_fresh(src, espdl, header, meta, "c") is False
