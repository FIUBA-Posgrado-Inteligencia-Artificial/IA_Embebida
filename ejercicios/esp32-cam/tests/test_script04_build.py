"""Tests para script04-build.py (preflight + build + flash + monitor)."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ESP32CAM_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ESP32CAM_DIR / "script04-build.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("script04_build", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_script04_help_lists_flags():
    r = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    for flag in ["--source", "--port", "--monitor", "--preflight", "--build-only"]:
        assert flag in r.stdout


def test_preflight_env_missing_idf_path(monkeypatch, tmp_path):
    monkeypatch.delenv("IDF_PATH", raising=False)
    mod = _load_module()
    monkeypatch.setattr(mod, "_try_load_idf_env", lambda: False)
    with pytest.raises(mod.PreflightError, match="IDF_PATH"):
        mod.check_idf_env()


def test_preflight_env_missing_idf_py(monkeypatch):
    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: None)
    mod = _load_module()
    with pytest.raises(mod.PreflightError, match="idf.py"):
        mod.check_idf_env()


def test_preflight_env_ok(monkeypatch):
    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/esp-idf/tools/idf.py" if name == "idf.py" else None)
    mod = _load_module()
    mod.check_idf_env()  # should not raise


def test_preflight_model_data_missing(tmp_path):
    mod = _load_module()
    with pytest.raises(mod.PreflightError, match="model_data.h"):
        mod.check_model_data_present(tmp_path / "firmware" / "main")


def test_preflight_model_data_present(tmp_path):
    firmware_main = tmp_path / "firmware" / "main"
    firmware_main.mkdir(parents=True)
    (firmware_main / "model_data.h").write_text("// dummy")
    mod = _load_module()
    mod.check_model_data_present(firmware_main)  # should not raise


MODEL_DATA_H_TEMPLATE = """\
#pragma once
#define MODEL_NUM_CLASSES {num_classes}
#define MODEL_INPUT_W {input_w}
#define MODEL_INPUT_H {input_h}
#define MODEL_INPUT_CHANNELS {input_channels}
#define MODEL_INPUT_MEAN_0 {mean0}
#define MODEL_INPUT_STD_0 {std0}
"""


def _write_pair(firmware_main: Path, *, num_classes=10, input_w=32, input_h=32,
                input_channels=1, mean0=124, std0=58, meta_override=None):
    firmware_main.mkdir(parents=True, exist_ok=True)
    (firmware_main / "model_data.h").write_text(
        MODEL_DATA_H_TEMPLATE.format(
            num_classes=num_classes, input_w=input_w, input_h=input_h,
            input_channels=input_channels, mean0=mean0, std0=std0,
        )
    )
    import json as _json
    meta = {
        "num_classes": num_classes, "input_w": input_w, "input_h": input_h,
        "input_channels": input_channels, "mean": [mean0], "std": [std0],
        "classes": [f"c{i}" for i in range(num_classes)],
    }
    if meta_override:
        meta.update(meta_override)
    (firmware_main / "model_data.meta.json").write_text(_json.dumps(meta))


def test_metadata_coherent(tmp_path):
    firmware_main = tmp_path / "firmware" / "main"
    _write_pair(firmware_main)
    mod = _load_module()
    mod.check_metadata_coherence(firmware_main)  # should not raise


def test_metadata_drift_detected(tmp_path):
    firmware_main = tmp_path / "firmware" / "main"
    _write_pair(firmware_main, meta_override={"num_classes": 7})
    mod = _load_module()
    with pytest.raises(mod.PreflightError, match="num_classes"):
        mod.check_metadata_coherence(firmware_main)


def test_metadata_rgb_channel_drift_detected(tmp_path):
    firmware_main = tmp_path / "firmware" / "main"
    firmware_main.mkdir(parents=True)
    (firmware_main / "model_data.h").write_text(
        "#pragma once\n"
        "#define MODEL_NUM_CLASSES 10\n"
        "#define MODEL_INPUT_W 32\n"
        "#define MODEL_INPUT_H 32\n"
        "#define MODEL_INPUT_CHANNELS 3\n"
        "#define MODEL_INPUT_MEAN_0 124\n"
        "#define MODEL_INPUT_MEAN_1 116\n"
        "#define MODEL_INPUT_MEAN_2 104\n"
        "#define MODEL_INPUT_STD_0 58\n"
        "#define MODEL_INPUT_STD_1 57\n"
        "#define MODEL_INPUT_STD_2 57\n"
    )
    import json as _json
    (firmware_main / "model_data.meta.json").write_text(_json.dumps({
        "num_classes": 10, "input_w": 32, "input_h": 32, "input_channels": 3,
        "mean": [124, 999, 104], "std": [58, 57, 57],
        "classes": [f"c{i}" for i in range(10)],
    }))
    mod = _load_module()
    with pytest.raises(mod.PreflightError, match=r"mean\[1\]"):
        mod.check_metadata_coherence(firmware_main)


def test_metadata_rgb_all_channels_coherent(tmp_path):
    firmware_main = tmp_path / "firmware" / "main"
    firmware_main.mkdir(parents=True)
    (firmware_main / "model_data.h").write_text(
        "#pragma once\n"
        "#define MODEL_NUM_CLASSES 10\n"
        "#define MODEL_INPUT_W 32\n"
        "#define MODEL_INPUT_H 32\n"
        "#define MODEL_INPUT_CHANNELS 3\n"
        "#define MODEL_INPUT_MEAN_0 124\n"
        "#define MODEL_INPUT_MEAN_1 116\n"
        "#define MODEL_INPUT_MEAN_2 104\n"
        "#define MODEL_INPUT_STD_0 58\n"
        "#define MODEL_INPUT_STD_1 57\n"
        "#define MODEL_INPUT_STD_2 57\n"
    )
    import json as _json
    (firmware_main / "model_data.meta.json").write_text(_json.dumps({
        "num_classes": 10, "input_w": 32, "input_h": 32, "input_channels": 3,
        "mean": [124, 116, 104], "std": [58, 57, 57],
        "classes": [f"c{i}" for i in range(10)],
    }))
    mod = _load_module()
    mod.check_metadata_coherence(firmware_main)  # should not raise


def test_metadata_missing_sidecar_skipped_with_warning(tmp_path, capsys):
    firmware_main = tmp_path / "firmware" / "main"
    firmware_main.mkdir(parents=True)
    (firmware_main / "model_data.h").write_text("// dummy")
    mod = _load_module()
    mod.check_metadata_coherence(firmware_main)  # should not raise
    out = capsys.readouterr().out
    assert "warning" in out.lower()
    assert "model_data.meta.json" in out


def test_psram_ok(tmp_path):
    fw = tmp_path / "firmware"
    fw.mkdir()
    (fw / "sdkconfig.defaults").write_text("CONFIG_ESP32_SPIRAM_SUPPORT=y\n")
    mod = _load_module()
    mod.check_psram_enabled(fw)  # should not raise


def test_psram_missing(tmp_path):
    fw = tmp_path / "firmware"
    fw.mkdir()
    (fw / "sdkconfig.defaults").write_text("# no spiram\n")
    mod = _load_module()
    with pytest.raises(mod.PreflightError, match="PSRAM"):
        mod.check_psram_enabled(fw)


def test_psram_neither_file(tmp_path):
    fw = tmp_path / "firmware"
    fw.mkdir()
    mod = _load_module()
    with pytest.raises(mod.PreflightError, match="sdkconfig"):
        mod.check_psram_enabled(fw)


def test_port_explicit_returned_as_is():
    mod = _load_module()
    assert mod.resolve_port("/dev/ttyUSB0", candidates=[]) == "/dev/ttyUSB0"


def test_port_auto_single_candidate():
    mod = _load_module()
    assert mod.resolve_port("auto", candidates=["/dev/ttyUSB0"]) == "/dev/ttyUSB0"


def test_port_auto_no_candidates():
    mod = _load_module()
    with pytest.raises(mod.PreflightError, match="no serial ports"):
        mod.resolve_port("auto", candidates=[])


def test_port_auto_multiple_candidates():
    mod = _load_module()
    with pytest.raises(mod.PreflightError, match="multiple"):
        mod.resolve_port("auto", candidates=["/dev/ttyUSB0", "/dev/ttyUSB1"])


class _FakePort:
    def __init__(self, device, vid, pid, description=""):
        self.device = device
        self.vid = vid
        self.pid = pid
        self.description = description


def test_detect_serial_ports_filters_by_esp_vid_pid(monkeypatch):
    """Matchea chips ESP y excluye VID/PID ajenos (Arduino, Digilent FT2232H)."""
    mod = _load_module()
    fake_ports = [
        _FakePort("/dev/ttyUSB0", 0x1A86, 0x7523),  # CH340 — ESP32-CAM-MB ✓
        _FakePort("/dev/ttyUSB1", 0x2341, 0x0043),  # Arduino Uno ✗
        _FakePort("/dev/ttyUSB2", 0x0403, 0x6010),  # Digilent FT2232H ✗
        _FakePort("/dev/ttyUSB3", 0x10C4, 0xEA60),  # CP210x — ESP DevKitC ✓
    ]
    import serial.tools.list_ports as lp
    monkeypatch.setattr(lp, "comports", lambda: fake_ports)
    assert mod._detect_serial_ports() == ["/dev/ttyUSB0", "/dev/ttyUSB3"]


def test_detect_serial_ports_falls_back_when_no_esp_match(monkeypatch):
    """Sin puertos ESP en pyserial, cae al glob."""
    mod = _load_module()
    import serial.tools.list_ports as lp
    monkeypatch.setattr(lp, "comports", lambda: [])
    monkeypatch.setattr(mod.glob, "glob",
                        lambda pat: ["/dev/ttyUSB9"] if "ttyUSB" in pat else [])
    assert mod._detect_serial_ports() == ["/dev/ttyUSB9"]


def test_resolve_port_multiple_error_points_to_pipelines_yaml(monkeypatch):
    mod = _load_module()
    with pytest.raises(mod.PreflightError, match="pipelines.yaml"):
        mod.resolve_port("auto", candidates=["/dev/ttyUSB0", "/dev/ttyUSB1"])


def test_main_build_only_invokes_build_and_skips_flash(tmp_path, monkeypatch):
    # Prepare a minimal firmware layout under tmp_path so preflight passes.
    fw = tmp_path / "firmware"
    main_dir = fw / "main"
    main_dir.mkdir(parents=True)
    (main_dir / "model_data.h").write_text(
        "#define MODEL_NUM_CLASSES 10\n#define MODEL_INPUT_W 32\n"
        "#define MODEL_INPUT_H 32\n#define MODEL_INPUT_CHANNELS 1\n"
        "#define MODEL_INPUT_MEAN_0 124\n#define MODEL_INPUT_STD_0 58\n"
    )
    import json as _json
    (main_dir / "model_data.meta.json").write_text(_json.dumps({
        "num_classes": 10, "input_w": 32, "input_h": 32, "input_channels": 1,
        "mean": [124], "std": [58], "classes": [f"c{i}" for i in range(10)],
    }))
    (fw / "sdkconfig.defaults").write_text("CONFIG_ESP32_SPIRAM_SUPPORT=y\n")
    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)
    monkeypatch.chdir(tmp_path)

    mod = _load_module()
    with patch.object(mod, "_run_idf", return_value=0) as m:
        rc = mod.main(["--build-only", "--port", "/dev/ttyUSB0"])
    assert rc == 0
    calls = [args[0] for args, _ in m.call_args_list]
    assert any(cmd[0:2] == ["idf.py", "build"] for cmd in calls)
    assert not any("flash" in " ".join(cmd) for cmd in calls)


def test_main_flash_then_monitor(tmp_path, monkeypatch):
    fw = tmp_path / "firmware"
    main_dir = fw / "main"
    main_dir.mkdir(parents=True)
    (main_dir / "model_data.h").write_text(
        "#define MODEL_NUM_CLASSES 10\n#define MODEL_INPUT_W 32\n"
        "#define MODEL_INPUT_H 32\n#define MODEL_INPUT_CHANNELS 1\n"
        "#define MODEL_INPUT_MEAN_0 124\n#define MODEL_INPUT_STD_0 58\n"
    )
    import json as _json
    (main_dir / "model_data.meta.json").write_text(_json.dumps({
        "num_classes": 10, "input_w": 32, "input_h": 32, "input_channels": 1,
        "mean": [124], "std": [58], "classes": [f"c{i}" for i in range(10)],
    }))
    (fw / "sdkconfig.defaults").write_text("CONFIG_ESP32_SPIRAM_SUPPORT=y\n")
    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)
    monkeypatch.chdir(tmp_path)

    mod = _load_module()
    with patch.object(mod, "_run_idf", return_value=0) as mock_idf, \
         patch.object(mod, "_run_flash", return_value=0) as mock_flash:
        rc = mod.main(["--port", "/dev/ttyUSB0", "--monitor"])
    assert rc == 0
    idf_cmds = [args[0] for args, _ in mock_idf.call_args_list]
    assert any(cmd[:2] == ["idf.py", "build"] for cmd in idf_cmds)
    assert mock_flash.call_count == 1
    assert any("monitor" in " ".join(cmd) for cmd in idf_cmds)


def test_main_fails_on_preflight_error(tmp_path, monkeypatch):
    monkeypatch.delenv("IDF_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    mod = _load_module()
    monkeypatch.setattr(mod, "_try_load_idf_env", lambda: False)
    rc = mod.main(["--build-only"])
    assert rc != 0


def test_main_monitor_only_skips_build_and_flash(tmp_path, monkeypatch):
    """--monitor-only saltea preflight de PSRAM y model_data, sólo requiere
    IDF env y un puerto, y ejecuta únicamente idf.py monitor."""
    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)
    monkeypatch.chdir(tmp_path)

    mod = _load_module()
    with patch.object(mod, "_run_idf", return_value=0) as mock_idf, \
         patch.object(mod, "_run_flash", return_value=0) as mock_flash:
        rc = mod.main(["--monitor-only", "--port", "/dev/ttyUSB0"])
    assert rc == 0
    assert mock_flash.call_count == 0
    cmds = [args[0] for args, _ in mock_idf.call_args_list]
    # Única llamada debe ser el monitor, no build.
    assert len(cmds) == 1
    assert cmds[0][:2] == ["idf.py", "-p"]
    assert "monitor" in cmds[0]


def test_main_monitor_only_conflicts_with_build_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = _load_module()
    rc = mod.main(["--monitor-only", "--build-only"])
    assert rc == 2


def test_main_menuconfig_only_runs_only_menuconfig(tmp_path, monkeypatch):
    """--menuconfig-only invoca `idf.py menuconfig` y nada más."""
    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)
    monkeypatch.chdir(tmp_path)

    mod = _load_module()
    with patch.object(mod, "_run_idf", return_value=0) as mock_idf, \
         patch.object(mod, "_run_flash", return_value=0) as mock_flash:
        rc = mod.main(["--menuconfig-only"])
    assert rc == 0
    assert mock_flash.call_count == 0
    cmds = [args[0] for args, _ in mock_idf.call_args_list]
    assert len(cmds) == 1
    assert cmds[0] == ["idf.py", "menuconfig"]


def test_main_menuconfig_only_conflicts_with_monitor_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = _load_module()
    rc = mod.main(["--menuconfig-only", "--monitor-only"])
    assert rc == 2


def test_main_monitor_only_works_without_firmware_main(tmp_path, monkeypatch):
    """Sin model_data.h ni sdkconfig — el monitor no los necesita."""
    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)
    monkeypatch.chdir(tmp_path)

    mod = _load_module()
    with patch.object(mod, "_run_idf", return_value=0):
        rc = mod.main(["--monitor-only", "--port", "/dev/ttyUSB0"])
    assert rc == 0


def test_install_model_data_copies_header_and_meta(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    fw_main = tmp_path / "firmware" / "main"
    fw_main.mkdir(parents=True)

    (models / "mymodel_quantized_ptq_model_data.h").write_text("// header")
    import json as _json
    (models / "mymodel_quantized_ptq_model_data.meta.json").write_text(
        _json.dumps({"num_classes": 10})
    )

    mod = _load_module()
    mod.install_model_data("mymodel_quantized_ptq", models, fw_main)

    assert (fw_main / "model_data.h").read_text() == "// header"
    assert _json.loads((fw_main / "model_data.meta.json").read_text()) == {"num_classes": 10}


def test_install_model_data_missing_header_raises(tmp_path):
    mod = _load_module()
    models = tmp_path / "models"
    models.mkdir()
    with pytest.raises(mod.PreflightError, match="Model header not found"):
        mod.install_model_data("nonexistent", models, tmp_path / "firmware" / "main")


def test_install_model_data_missing_meta_warns(tmp_path, capsys):
    models = tmp_path / "models"
    models.mkdir()
    fw_main = tmp_path / "firmware" / "main"
    fw_main.mkdir(parents=True)
    (models / "foo_model_data.h").write_text("// header")

    mod = _load_module()
    mod.install_model_data("foo", models, fw_main)

    assert (fw_main / "model_data.h").exists()
    out = capsys.readouterr().out
    assert "warning" in out.lower()


def test_main_with_source_copies_before_build(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    fw = tmp_path / "firmware"
    fw_main = fw / "main"
    fw_main.mkdir(parents=True)
    (fw / "sdkconfig.defaults").write_text("CONFIG_ESP32_SPIRAM_SUPPORT=y\n")

    import json as _json
    (models / "mymodel_model_data.h").write_text(
        "#define MODEL_NUM_CLASSES 10\n#define MODEL_INPUT_W 32\n"
        "#define MODEL_INPUT_H 32\n#define MODEL_INPUT_CHANNELS 1\n"
        "#define MODEL_INPUT_MEAN_0 124\n#define MODEL_INPUT_STD_0 58\n"
    )
    (models / "mymodel_model_data.meta.json").write_text(_json.dumps({
        "num_classes": 10, "input_w": 32, "input_h": 32, "input_channels": 1,
        "mean": [124], "std": [58], "classes": [f"c{i}" for i in range(10)],
    }))

    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)
    monkeypatch.chdir(tmp_path)

    mod = _load_module()
    with patch.object(mod, "_run_idf", return_value=0):
        rc = mod.main(["--source", "mymodel", "--build-only", "--port", "/dev/ttyUSB0"])
    assert rc == 0
    assert (fw_main / "model_data.h").exists()
