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
