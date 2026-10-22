"""Tests del parser de self-test de script06-bench."""

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "script06-bench.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("script06_bench", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SAMPLE = "  frog      -> monkey    24.0%  (pc_q: monkey    24.0% | float  23.8%)  dlogits<=0  39ms  PASS"


def test_selftest_line_regex():
    mod = _load_module()
    m = mod._SELFTEST_LINE.match(SAMPLE)
    assert m
    assert m.group(1) == "frog"
    assert m.group(2) == "monkey"
    assert m.group(4) == "0"
    assert m.group(5) == "39"
    assert m.group(6) == "PASS"


def test_selftest_line_regex_fail_case():
    mod = _load_module()
    line = "  truck     -> car       35.9%  (pc_q: truck 40.0% | float 34.0%)  dlogits<=3  111ms  FAIL"
    m = mod._SELFTEST_LINE.match(line)
    assert m and m.group(6) == "FAIL" and m.group(4) == "3" and m.group(5) == "111"


def test_selftest_line_regex_ignores_noise():
    mod = _load_module()
    assert mod._SELFTEST_LINE.match("I (3435) wifi:state: auth -> assoc (0x0)") is None


# ---------------------------------------------------------------------------
# flags --runtime y --build-dir
# ---------------------------------------------------------------------------


def test_parse_args_runtime_default_espdl(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr("sys.argv", ["script06-bench.py", "--source", "m"])
    args = mod.parse_args()
    assert args.runtime == "espdl"
    assert args.build_dir is None


def test_parse_args_runtime_and_build_dir(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr("sys.argv", [
        "script06-bench.py", "--source", "m", "--board", "esp32s3",
        "--runtime", "tflm_espnn", "--build-dir", "firmware_rt/tflm/build_espnn",
    ])
    args = mod.parse_args()
    assert args.runtime == "tflm_espnn"
    assert args.build_dir == "firmware_rt/tflm/build_espnn"


def test_parse_args_runtime_invalid_rejected(monkeypatch):
    import pytest
    mod = _load_module()
    monkeypatch.setattr("sys.argv", ["script06-bench.py", "--source", "m",
                                     "--runtime", "tvm"])
    with pytest.raises(SystemExit):
        mod.parse_args()
