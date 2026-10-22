"""Tests para model_registry."""

from pathlib import Path

import pytest

from model_registry import infer_stage_from_name, scan_models_dir


@pytest.mark.parametrize("name,expected", [
    ("depthwise_gray32", "baseline"),
    ("standard_rgb96", "baseline"),
    ("depthwise_gray32_pruned_activation_30", "pruned"),
    ("depthwise_gray32_pruned_magnitude_50", "pruned"),
    ("depthwise_gray32_quantized_ptq", "quantized"),
    ("depthwise_gray32_pruned_activation_30_quantized_qat", "quantized"),
    ("model.espdl", "espdl"),
    ("depthwise_gray32.espdl", "espdl"),
])
def test_infer_stage_from_name(name, expected):
    assert infer_stage_from_name(name) == expected


def test_infer_stage_finetuned():
    assert infer_stage_from_name("m_pruned_activation_30_ft5.pth") == "finetuned"


def test_finetuned_loses_to_quantized_and_espdl():
    assert infer_stage_from_name("m_ft5_quantized_ptq.pth") == "quantized"
    assert infer_stage_from_name("m_ft5_quantized_ptq.espdl") == "espdl"


def test_pruned_after_ft_is_pruned():
    # el último paso fue prune: _ft5_pruned_... no termina en _ftN
    assert infer_stage_from_name("m_ft5_pruned_activation_20.pth") == "pruned"


def test_scan_models_dir_empty(tmp_path: Path):
    (tmp_path / "models").mkdir()
    assert scan_models_dir(tmp_path / "models") == []


def test_scan_models_dir_finds_pth_and_espdl(tmp_path: Path):
    d = tmp_path / "models"
    d.mkdir()
    (d / "depthwise_gray32.pth").write_bytes(b"x" * 100)
    (d / "depthwise_gray32_quantized_ptq.espdl").write_bytes(b"y" * 200)
    (d / "README.md").write_text("ignore me")

    entries = scan_models_dir(d)
    entries.sort(key=lambda e: e["name"])

    assert len(entries) == 2
    assert entries[0]["name"] == "depthwise_gray32"
    assert entries[0]["path"] == str(d / "depthwise_gray32.pth")
    assert entries[0]["size_bytes"] == 100
    assert entries[0]["stage"] == "baseline"
    assert entries[0]["modified"]  # non-empty ISO string
    assert entries[1]["name"] == "depthwise_gray32_quantized_ptq.espdl"
    assert entries[1]["stage"] == "espdl"
    assert entries[1]["size_bytes"] == 200


def test_scan_models_dir_missing(tmp_path: Path):
    assert scan_models_dir(tmp_path / "nonexistent") == []


import json

from model_registry import load_registry, save_registry


def test_load_registry_missing(tmp_path: Path):
    assert load_registry(tmp_path / "models.json") == {"models": [], "updated_at": ""}


def test_save_and_load_registry_roundtrip(tmp_path: Path):
    registry_path = tmp_path / "models.json"
    data = {
        "models": [
            {"name": "depthwise_gray32", "path": "models/x.pth", "stage": "baseline",
             "size_bytes": 100, "modified": "2026-04-23T10:00:00", "notes": ""}
        ],
        "updated_at": "2026-04-23T10:05:00",
    }
    save_registry(registry_path, data)

    assert registry_path.exists()
    raw = json.loads(registry_path.read_text())
    assert raw == data
    assert load_registry(registry_path) == data


def test_save_registry_atomic_write(tmp_path: Path):
    registry_path = tmp_path / "models.json"
    save_registry(registry_path, {"models": [], "updated_at": "x"})
    # No leftover tmp file
    siblings = list(tmp_path.iterdir())
    assert siblings == [registry_path]


from model_registry import reconcile


def _entry(name, path="models/x.pth", size=100, stage="baseline", notes=""):
    return {"name": name, "path": path, "stage": stage,
            "size_bytes": size, "modified": "2026-04-23T10:00:00", "notes": notes}


def test_reconcile_adds_new_entries():
    registry = {"models": [], "updated_at": ""}
    disk = [_entry("a"), _entry("b")]
    updated, changes = reconcile(registry, disk)
    names = sorted(m["name"] for m in updated["models"])
    assert names == ["a", "b"]
    assert sorted(changes["added"]) == ["a", "b"]
    assert changes["removed"] == []
    assert changes["updated"] == []


def test_reconcile_removes_orphans():
    registry = {"models": [_entry("a"), _entry("gone")], "updated_at": ""}
    disk = [_entry("a")]
    updated, changes = reconcile(registry, disk)
    assert [m["name"] for m in updated["models"]] == ["a"]
    assert changes["removed"] == ["gone"]
    assert changes["added"] == []


def test_reconcile_preserves_notes_and_updates_size():
    registry = {
        "models": [_entry("a", size=50, notes="manual note")],
        "updated_at": "",
    }
    disk = [_entry("a", size=200)]
    updated, changes = reconcile(registry, disk)
    assert len(updated["models"]) == 1
    assert updated["models"][0]["size_bytes"] == 200
    assert updated["models"][0]["notes"] == "manual note"
    assert changes["updated"] == ["a"]


def test_reconcile_sets_updated_at():
    registry = {"models": [], "updated_at": ""}
    updated, _ = reconcile(registry, [])
    assert updated["updated_at"]  # ISO string, non-empty


# ---------------------------------------------------------------------------
# benchmarks.json
# ---------------------------------------------------------------------------

from model_registry import (
    load_benchmarks, save_benchmarks, upsert_benchmark, benchmarks_by_key,
)


def test_benchmarks_roundtrip_and_upsert(tmp_path):
    p = tmp_path / "benchmarks.json"
    data = load_benchmarks(p)
    assert data["benchmarks"] == []

    upsert_benchmark(data, "m1_quantized_ptq", "esp32cam",
                     {"inf_ms_mean": 80.2, "pass_count": 10, "total": 10})
    upsert_benchmark(data, "m1_quantized_ptq", "esp32s3cam",
                     {"inf_ms_mean": 12.1, "pass_count": 10, "total": 10})
    save_benchmarks(p, data)

    back = load_benchmarks(p)
    idx = benchmarks_by_key(back)
    assert idx[("m1_quantized_ptq", "esp32cam", "espdl")]["inf_ms_mean"] == 80.2
    assert len(back["benchmarks"]) == 2

    # upsert reemplaza, no duplica
    upsert_benchmark(back, "m1_quantized_ptq", "esp32cam", {"inf_ms_mean": 79.0})
    assert len(back["benchmarks"]) == 2
    assert benchmarks_by_key(back)[("m1_quantized_ptq", "esp32cam", "espdl")]["inf_ms_mean"] == 79.0


def test_upsert_default_runtime_is_espdl():
    data = {"benchmarks": [], "updated_at": ""}
    upsert_benchmark(data, "m", "esp32s3", {"inf_ms_mean": 5.0})
    assert data["benchmarks"][0]["runtime"] == "espdl"


def test_upsert_runtime_dimension_keys_independently():
    data = {"benchmarks": [], "updated_at": ""}
    upsert_benchmark(data, "m", "esp32s3", {"inf_ms_mean": 5.0}, runtime="espdl")
    upsert_benchmark(data, "m", "esp32s3", {"inf_ms_mean": 9.0}, runtime="tflm")
    assert len(data["benchmarks"]) == 2

    # upsert del mismo (model, board, runtime) reemplaza, no duplica
    upsert_benchmark(data, "m", "esp32s3", {"inf_ms_mean": 8.0}, runtime="tflm")
    assert len(data["benchmarks"]) == 2
    idx = benchmarks_by_key(data)
    assert idx[("m", "esp32s3", "tflm")]["inf_ms_mean"] == 8.0
    assert idx[("m", "esp32s3", "espdl")]["inf_ms_mean"] == 5.0


def test_legacy_entries_without_runtime_read_as_espdl():
    # benchmarks.json viejo: filas sin campo `runtime`
    data = {"benchmarks": [{"model": "m", "board": "esp32cam", "inf_ms_mean": 80.0}],
            "updated_at": ""}
    idx = benchmarks_by_key(data)
    assert idx[("m", "esp32cam", "espdl")]["inf_ms_mean"] == 80.0

    # un upsert espdl sobre la fila legacy la reemplaza (misma key)
    upsert_benchmark(data, "m", "esp32cam", {"inf_ms_mean": 75.0})
    assert len(data["benchmarks"]) == 1
    assert benchmarks_by_key(data)[("m", "esp32cam", "espdl")]["inf_ms_mean"] == 75.0


# ---------------------------------------------------------------------------
# tabla comparativa con dimensión runtime
# ---------------------------------------------------------------------------


def _table_results():
    return [{"name": "m", "accuracy": 0.5, "params": 1000, "macs": 2000,
             "size_kb": 10.0, "arch": "depthwise", "input_mode": "rgb",
             "resolution": 32}]


def test_table_bench_column_lists_boards_with_data(capsys):
    """La tabla no muestra tiempos (eso es `main.py bench`); solo qué placas
    tienen mediciones, en el orden canónico de _BENCH_BOARD_ORDER."""
    from esp32cam_utils import print_comparison_table
    bench = {
        ("m", "esp32s3", "espdl"): {"inf_ms_mean": 12.0},
        ("m", "esp32cam", "espdl"): {"inf_ms_mean": 80.0},
        ("m", "esp32s3", "tflm"): {"inf_ms_mean": 30.0},  # misma placa, no duplica
        ("otro", "esp32", "espdl"): {"inf_ms_mean": 1.0},  # otro modelo, no aparece
    }
    print_comparison_table(_table_results(), benchmarks=bench)
    out = capsys.readouterr().out
    assert "Bench" in out
    assert "esp32cam,esp32s3" in out
    # los tiempos no van en esta tabla
    assert "12.0" not in out and "80.0" not in out


def test_table_bench_column_dash_without_data(capsys):
    from esp32cam_utils import print_comparison_table
    print_comparison_table(_table_results(), benchmarks=None)
    out = capsys.readouterr().out
    row = [l for l in out.splitlines() if l.startswith("m ")][0]
    assert row.rstrip().endswith("-")
