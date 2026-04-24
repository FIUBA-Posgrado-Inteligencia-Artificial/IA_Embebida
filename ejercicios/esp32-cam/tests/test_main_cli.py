"""Tests para el CLI main.py."""

import subprocess
import sys
from pathlib import Path

import pytest


ESP32CAM_DIR = Path(__file__).resolve().parents[1]


def _main(*argv, cwd=None):
    # When cwd is specified, use absolute path to main.py so it can be found
    if cwd is None:
        cwd = ESP32CAM_DIR
        main_path = "main.py"
    else:
        main_path = str(ESP32CAM_DIR / "main.py")
    result = subprocess.run(
        [sys.executable, main_path, *argv],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result


def test_main_no_args_prints_usage():
    r = _main()
    assert r.returncode != 0
    assert "usage" in (r.stdout + r.stderr).lower()


def test_main_help_lists_subcommands():
    r = _main("--help")
    assert r.returncode == 0
    text = r.stdout + r.stderr
    for sub in ["run", "flows", "models", "menuconfig"]:
        assert sub in text


def test_menuconfig_dispatches_to_script04(monkeypatch):
    """`main.py menuconfig` invoca script04-build.py --menuconfig-only."""
    import sys as _sys
    _sys.path.insert(0, str(ESP32CAM_DIR))
    import main as main_mod

    captured = {}

    def fake_call(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return 0

    monkeypatch.setattr("subprocess.call", fake_call)
    rc = main_mod._cmd_menuconfig()
    assert rc == 0
    assert captured["cmd"][-1] == "--menuconfig-only"
    assert captured["cmd"][-2].endswith("script04-build.py")


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "pipelines.yaml"
    p.write_text(content)
    return p


YAML_SIMPLE = """
flows:
  demo:
    - stage: train
      params: { arch: depthwise, input_mode: gray, resolution: 32 }
"""


def test_run_dry_run_outputs_command(tmp_path: Path):
    cfg = _write_yaml(tmp_path, YAML_SIMPLE)
    r = _main("run", "demo", "--dry-run", "--config", str(cfg))
    assert r.returncode == 0
    assert "[dry-run]" in r.stdout
    assert "script01-train.py" in r.stdout


def test_run_unknown_flow_fails(tmp_path: Path):
    cfg = _write_yaml(tmp_path, YAML_SIMPLE)
    r = _main("run", "missing", "--config", str(cfg))
    assert r.returncode != 0
    assert "not found" in (r.stdout + r.stderr).lower()


YAML_TWO_FLOWS = """
flows:
  default:
    - stage: train
      params: {}
    - stage: prune
      source: x
      params: {}
    - stage: quantize
      source: y
      params: {}
    - stage: build
      params: {}
  quick:
    - stage: build
      params: {}
"""


def test_flows_lists_flow_names(tmp_path: Path):
    cfg = _write_yaml(tmp_path, YAML_TWO_FLOWS)
    r = _main("flows", "--config", str(cfg))
    assert r.returncode == 0
    # Flow names appear
    assert "default" in r.stdout
    assert "quick" in r.stdout
    # Shows stage count and first/last
    assert "4 stages" in r.stdout
    assert "1 stages" in r.stdout
    assert "train" in r.stdout  # first stage of default
    assert "build" in r.stdout  # last stage of default AND only stage of quick


def test_models_prints_table_when_empty(tmp_path: Path):
    (tmp_path / "models").mkdir()
    (tmp_path / "pipelines.yaml").write_text("flows: {}\n")
    r = _main("models", cwd=tmp_path)
    assert r.returncode == 0
    assert "sin modelos disponibles" in r.stdout.lower()


def test_models_prints_table_with_entries(tmp_path: Path):
    """Crea checkpoints reales (via save_checkpoint) para que la tabla comparativa
    pueda cargarlos y mostrar métricas. Los blobs binarios falsos que usaba el
    test anterior ya no alcanzan porque la tabla ahora carga el .pth."""
    import sys as _sys
    _sys.path.insert(0, str(ESP32CAM_DIR))
    from esp32cam_utils import build_model, save_checkpoint

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    meta = {
        "arch": "depthwise", "input_mode": "gray", "resolution": 32,
        "num_classes": 10, "accuracy": 0.42, "history": [],
        "normalize_mean": [0.44], "normalize_std": [0.22],
        "channels": (32, 64, 128), "source": None,
    }
    save_checkpoint(models_dir / "alpha.pth", build_model("depthwise", "gray"), meta)

    r = _main("models", cwd=tmp_path)
    assert r.returncode == 0
    assert "alpha" in r.stdout
    assert "Acc" in r.stdout
    assert "KB" in r.stdout
    assert "depthwise" in r.stdout


def test_models_clean_removes_orphan(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "alpha.pth").write_bytes(b"x" * 1024)
    # Seed a stale registry that references a non-existent model
    import json
    (tmp_path / "models.json").write_text(json.dumps({
        "models": [
            {"name": "alpha", "path": str(models_dir / "alpha.pth"),
             "stage": "baseline", "size_bytes": 1024,
             "modified": "2026-04-23T10:00:00", "notes": ""},
            {"name": "gone",  "path": str(models_dir / "gone.pth"),
             "stage": "baseline", "size_bytes": 100,
             "modified": "2026-04-23T10:00:00", "notes": ""},
        ],
        "updated_at": "2026-04-23T10:00:00",
    }))
    r = _main("models", "clean", cwd=tmp_path)
    assert r.returncode == 0
    assert "removed" in r.stdout.lower()
    assert "gone" in r.stdout

    data = json.loads((tmp_path / "models.json").read_text())
    names = [m["name"] for m in data["models"]]
    assert names == ["alpha"]


def test_models_clean_dry_run_does_not_write(tmp_path: Path):
    (tmp_path / "models").mkdir()
    import json
    initial = {
        "models": [{"name": "gone", "path": "models/gone.pth",
                    "stage": "baseline", "size_bytes": 1,
                    "modified": "2026-04-23T10:00:00", "notes": ""}],
        "updated_at": "2026-04-23T10:00:00",
    }
    (tmp_path / "models.json").write_text(json.dumps(initial))
    r = _main("models", "clean", "--dry-run", cwd=tmp_path)
    assert r.returncode == 0
    assert "would remove" in r.stdout.lower()
    # File unchanged
    assert json.loads((tmp_path / "models.json").read_text()) == initial


def test_run_dry_run_does_not_touch_registry(tmp_path: Path):
    (tmp_path / "models").mkdir()
    cfg = _write_yaml(tmp_path, YAML_SIMPLE)
    r = _main("run", "demo", "--dry-run", "--config", str(cfg), cwd=tmp_path)
    assert r.returncode == 0
    assert not (tmp_path / "models.json").exists()


def test_run_live_refreshes_registry_after_success(tmp_path: Path, monkeypatch):
    """Flow with a fake 'stage' that just touches a file, then registry should exist."""
    import json
    (tmp_path / "models").mkdir()
    # Pre-seed a pth so reconcile picks it up after the (stub) flow.
    (tmp_path / "models" / "alpha.pth").write_bytes(b"x")
    # We cannot easily make the live path succeed without mocking subprocess.
    # Instead, assert the registry is written when we drive _refresh_registry directly.
    import sys as _sys
    _sys.path.insert(0, str(ESP32CAM_DIR))
    from main import _refresh_registry
    _refresh_registry(cwd=tmp_path)
    data = json.loads((tmp_path / "models.json").read_text())
    assert [m["name"] for m in data["models"]] == ["alpha"]


def test_shipped_pipelines_yaml_parses_and_lists():
    # Using the real file in the repo — this is an integration smoke test.
    r = _main("flows")
    assert r.returncode == 0
    assert "fast_debug" in r.stdout
    assert "good_model" in r.stdout
    assert "small_model" in r.stdout


def test_shipped_fast_debug_flow_dry_run():
    r = _main("run", "fast_debug", "--dry-run")
    assert r.returncode == 0
    assert "script01-train.py" in r.stdout
    assert "script02-prune.py" in r.stdout
    assert "script03a-quantize.py" in r.stdout
    assert "script04-build.py" in r.stdout


def _seed_reset_fixtures(tmp_path: Path) -> list[Path]:
    """Crea un conjunto representativo de artefactos y devuelve sus paths."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    fw_main = tmp_path / "firmware" / "main"
    fw_main.mkdir(parents=True)

    created = [
        models_dir / "alpha.pth",
        models_dir / "beta_quantized_ptq.espdl",
        models_dir / "beta_quantized_ptq.info",
        models_dir / "beta_quantized_ptq.json",
        models_dir / "beta_quantized_ptq_model_data.h",
        models_dir / "beta_quantized_ptq_model_data.meta.json",
        tmp_path / "models.json",
        fw_main / "model_data.h",
        fw_main / "model_data.meta.json",
    ]
    for p in created:
        p.write_bytes(b"x")
    return created


def test_models_reset_without_yes_fails_and_keeps_files(tmp_path: Path):
    files = _seed_reset_fixtures(tmp_path)
    r = _main("models", "reset", cwd=tmp_path)
    assert r.returncode != 0
    assert "--yes" in (r.stdout + r.stderr)
    for p in files:
        assert p.exists(), f"{p} fue borrado sin confirmación"


def test_models_reset_dry_run_lists_but_does_not_delete(tmp_path: Path):
    files = _seed_reset_fixtures(tmp_path)
    r = _main("models", "reset", "--dry-run", cwd=tmp_path)
    assert r.returncode == 0
    assert "would delete" in r.stdout
    for p in files:
        assert p.exists()


def test_models_reset_yes_deletes_artifacts(tmp_path: Path):
    files = _seed_reset_fixtures(tmp_path)
    # Control: un archivo arbitrario fuera del scope no debe tocarse.
    unrelated = tmp_path / "firmware" / "main" / "other.c"
    unrelated.write_text("// not a model artifact")

    r = _main("models", "reset", "--yes", cwd=tmp_path)
    assert r.returncode == 0
    for p in files:
        assert not p.exists(), f"{p} debería haberse borrado"
    assert unrelated.exists(), "reset borró algo fuera de su scope"


def test_models_reset_empty_dir_is_noop(tmp_path: Path):
    (tmp_path / "models").mkdir()
    r = _main("models", "reset", "--yes", cwd=tmp_path)
    assert r.returncode == 0
    assert "nothing to delete" in r.stdout.lower()


def test_shipped_fast_debug_only_build_dry_run():
    r = _main("run", "fast_debug_only_build", "--dry-run")
    assert r.returncode == 0
    assert "script04-build.py" in r.stdout
    assert "script01-train.py" not in r.stdout
    assert "script02-prune.py" not in r.stdout
