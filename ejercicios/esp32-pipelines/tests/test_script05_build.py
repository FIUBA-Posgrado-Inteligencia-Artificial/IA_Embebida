"""Tests para script05-build.py (preflight + build + flash + monitor)."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ESP32CAM_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ESP32CAM_DIR / "scripts" / "script05-build.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("script05_build", SCRIPT_PATH)
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
    for flag in ["--source", "--port", "--action", "--preflight"]:
        assert flag in r.stdout


def test_parser_has_action_with_default_build():
    mod = _load_module()
    args = mod.build_parser().parse_args([])
    assert args.action == "build"


def test_parser_rejects_unknown_action():
    mod = _load_module()
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--action", "deploy"])


def test_old_flags_removed():
    mod = _load_module()
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--build-only"])
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--monitor-only"])


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


def _write_psram_board_headers(fw, board="esp32cam", board_has_psram=1, feature=-1):
    main_dir = fw / "main"
    main_dir.mkdir(parents=True, exist_ok=True)
    (main_dir / "firmware_config.h").write_text(
        f"#define FEATURE_PSRAM {feature}\n"
    )
    boards_dir = main_dir / "boards"
    boards_dir.mkdir(exist_ok=True)
    (boards_dir / f"board_{board}.h").write_text(
        f"#define BOARD_HAS_PSRAM {board_has_psram}\n"
    )


def test_psram_ok(tmp_path):
    fw = tmp_path / "firmware"
    fw.mkdir()
    _write_psram_board_headers(fw)
    (fw / "sdkconfig.defaults").write_text("CONFIG_ESP32_SPIRAM_SUPPORT=y\n")
    mod = _load_module()
    mod.check_psram_enabled(fw, "esp32cam")  # should not raise


def test_psram_missing(tmp_path):
    fw = tmp_path / "firmware"
    fw.mkdir()
    _write_psram_board_headers(fw)
    (fw / "sdkconfig.defaults").write_text("# no spiram\n")
    mod = _load_module()
    with pytest.raises(mod.PreflightError, match="PSRAM"):
        mod.check_psram_enabled(fw, "esp32cam")


def test_psram_neither_file(tmp_path):
    fw = tmp_path / "firmware"
    fw.mkdir()
    _write_psram_board_headers(fw)
    mod = _load_module()
    with pytest.raises(mod.PreflightError, match="sdkconfig"):
        mod.check_psram_enabled(fw, "esp32cam")


def test_psram_skipped_when_board_has_no_psram(tmp_path):
    """BOARD_HAS_PSRAM=0 (con FEATURE_PSRAM=-1 auto) => no exige sdkconfig."""
    fw = tmp_path / "firmware"
    fw.mkdir()
    _write_psram_board_headers(fw, board="esp32s3", board_has_psram=0)
    mod = _load_module()
    mod.check_psram_enabled(fw, "esp32s3")  # should not raise, sin sdkconfig


def test_psram_feature_flag_forces_check_even_if_board_lacks_psram(tmp_path):
    """FEATURE_PSRAM=1 fuerza el chequeo aunque BOARD_HAS_PSRAM=0."""
    fw = tmp_path / "firmware"
    fw.mkdir()
    _write_psram_board_headers(fw, board="esp32s3", board_has_psram=0, feature=1)
    (fw / "sdkconfig.defaults").write_text("# no spiram\n")
    mod = _load_module()
    with pytest.raises(mod.PreflightError, match="PSRAM"):
        mod.check_psram_enabled(fw, "esp32s3")


def test_parser_has_board_and_no_target():
    mod = _load_module()
    args = mod.build_parser().parse_args([])
    assert args.board == "esp32cam"
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--target", "c"])
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--board", "banana"])


def test_board_table_derivations():
    mod = _load_module()
    assert mod.BOARDS["esp32s3cam"] == {"chip": "esp32s3", "espdl_target": "esp32s3"}
    assert mod.BOARDS["esp32"]["espdl_target"] == "c"
    assert mod.board_build_dir("esp32s3") == "build_esp32s3"


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


def test_main_build_action_invokes_build_and_skips_flash(tmp_path, monkeypatch):
    """--action build (default) sólo compila; no flashea ni monitorea."""
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
        "target": "c",
    }))
    (fw / "sdkconfig.defaults").write_text("CONFIG_ESP32_SPIRAM_SUPPORT=y\n")
    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)
    monkeypatch.chdir(tmp_path)

    mod = _load_module()
    with patch.object(mod, "_run_idf", return_value=0) as m:
        rc = mod.main(["--action", "build"])
    assert rc == 0
    calls = [args[0] for args, _ in m.call_args_list]
    assert any(
        cmd[-1] == "build" and "-DBOARD=esp32cam" in cmd
        and "-DSDKCONFIG=build_esp32cam/sdkconfig" in cmd
        for cmd in calls
    )
    assert not any("flash" in " ".join(cmd) for cmd in calls)


def test_main_build_does_not_need_port(tmp_path, monkeypatch):
    """build no resuelve puerto: debe andar sin ningún puerto serie presente."""
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
        "target": "c",
    }))
    (fw / "sdkconfig.defaults").write_text("CONFIG_ESP32_SPIRAM_SUPPORT=y\n")
    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)
    monkeypatch.chdir(tmp_path)

    mod = _load_module()
    with patch.object(mod, "resolve_port") as mock_resolve_port, \
         patch.object(mod, "_run_idf", return_value=0):
        rc = mod.main([])
    assert rc == 0
    mock_resolve_port.assert_not_called()


def test_main_flash_action_resolves_port_and_flashes(tmp_path, monkeypatch):
    fw = tmp_path / "firmware"
    main_dir = fw / "main"
    main_dir.mkdir(parents=True)
    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)
    monkeypatch.chdir(tmp_path)

    mod = _load_module()
    with patch.object(mod, "_run_flash", return_value=0) as mock_flash:
        rc = mod.main(["--action", "flash", "--port", "/dev/ttyUSB0"])
    assert rc == 0
    mock_flash.assert_called_once()
    assert mock_flash.call_args[0][1] == "/dev/ttyUSB0"


def test_main_monitor_action_resolves_port_and_opens_monitor(tmp_path, monkeypatch):
    """--action monitor sólo requiere IDF env y un puerto, y ejecuta
    únicamente idf.py monitor (sin build/flash ni preflight de PSRAM/model_data)."""
    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)
    monkeypatch.chdir(tmp_path)

    mod = _load_module()
    with patch.object(mod, "_run_idf", return_value=0) as mock_idf, \
         patch.object(mod, "_run_flash", return_value=0) as mock_flash:
        rc = mod.main(["--action", "monitor", "--port", "/dev/ttyUSB0"])
    assert rc == 0
    assert mock_flash.call_count == 0
    cmds = [args[0] for args, _ in mock_idf.call_args_list]
    # Única llamada debe ser el monitor, no build.
    assert len(cmds) == 1
    assert cmds[0][:2] == ["idf.py", "-B"]
    assert "monitor" in cmds[0]
    assert "-p" in cmds[0]


def test_main_fails_on_preflight_error(tmp_path, monkeypatch):
    monkeypatch.delenv("IDF_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    mod = _load_module()
    monkeypatch.setattr(mod, "_try_load_idf_env", lambda: False)
    rc = mod.main(["--action", "build"])
    assert rc != 0


def test_main_monitor_action_works_without_firmware_main(tmp_path, monkeypatch):
    """Sin model_data.h ni sdkconfig — el monitor no los necesita."""
    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)
    monkeypatch.chdir(tmp_path)

    mod = _load_module()
    with patch.object(mod, "_run_idf", return_value=0):
        rc = mod.main(["--action", "monitor", "--port", "/dev/ttyUSB0"])
    assert rc == 0


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


def test_install_model_data_copies_header_and_meta(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    fw_main = tmp_path / "firmware" / "main"
    fw_main.mkdir(parents=True)

    (models / "mymodel_quantized_ptq_model_data.h").write_text("// header")
    import json as _json
    (models / "mymodel_quantized_ptq_model_data.meta.json").write_text(
        _json.dumps({"num_classes": 10, "target": "c"})
    )

    mod = _load_module()
    mod.install_model_data("mymodel_quantized_ptq", models, fw_main)

    assert (fw_main / "model_data.h").read_text() == "// header"
    assert _json.loads((fw_main / "model_data.meta.json").read_text()) == {
        "num_classes": 10, "target": "c"}


def test_install_model_data_prefers_header_staged_in_firmware_main(tmp_path):
    """El header staged en firmware/main gana sobre el de models/.

    Es el caso de `--target esp32s3 --firmware-dir ../esp32-s3-cam/firmware/main`:
    models/ conserva el header del ESP32 clásico con el mismo nombre, y si ese
    ganara se instalaría un .espdl de otro target sin ningún error visible.
    """
    import json as _json
    models = tmp_path / "models"
    models.mkdir()
    fw_main = tmp_path / "firmware" / "main"
    fw_main.mkdir(parents=True)

    (models / "m_model_data.h").write_text("// target c")
    (models / "m_model_data.meta.json").write_text(
        _json.dumps({"num_classes": 1, "target": "c"}))
    (fw_main / "m_model_data.h").write_text("// target esp32s3")
    (fw_main / "m_model_data.meta.json").write_text(
        _json.dumps({"num_classes": 10, "target": "esp32s3"}))

    mod = _load_module()
    mod.install_model_data("m", models, fw_main, "esp32s3")

    assert (fw_main / "model_data.h").read_text() == "// target esp32s3"
    assert _json.loads((fw_main / "model_data.meta.json").read_text()) == {
        "num_classes": 10, "target": "esp32s3"}


def test_install_model_data_missing_header_raises(tmp_path):
    mod = _load_module()
    models = tmp_path / "models"
    models.mkdir()
    with pytest.raises(mod.PreflightError, match="Model header not found"):
        mod.install_model_data("nonexistent", models, tmp_path / "firmware" / "main")


def test_install_model_data_missing_meta_raises(tmp_path):
    """Sin sidecar no se puede verificar el target, así que se rechaza.

    Antes esto era sólo un warning. Se endureció porque un header sin sidecar
    es indistinguible de uno compilado para el otro chip, y montar el target
    equivocado no falla: da predicciones basura en silencio.
    """
    models = tmp_path / "models"
    models.mkdir()
    fw_main = tmp_path / "firmware" / "main"
    fw_main.mkdir(parents=True)
    (models / "foo_model_data.h").write_text("// header")

    mod = _load_module()
    with pytest.raises(mod.PreflightError, match="Sidecar no encontrado"):
        mod.install_model_data("foo", models, fw_main)

    assert not (fw_main / "model_data.h").exists()


def test_install_model_data_rejects_target_mismatch(tmp_path):
    """El chequeo que realmente cierra el agujero: sidecar target != esperado.

    Cubre la variante que abre la prioridad de firmware_main: un
    `<source>_model_data.h` viejo staged ahí por un `--firmware-dir` manual le
    gana a todo quantize posterior. Con el chequeo de target, ese header
    pegajoso hace fallar el build en vez de embeberse en silencio.
    """
    import json as _json
    models = tmp_path / "models"
    models.mkdir()
    fw_main = tmp_path / "firmware" / "main"
    fw_main.mkdir(parents=True)

    (fw_main / "m_model_data.h").write_text("// compilado para el ESP32 clasico")
    (fw_main / "m_model_data.meta.json").write_text(
        _json.dumps({"num_classes": 10, "target": "c"}))

    mod = _load_module()
    with pytest.raises(mod.PreflightError, match="Target mismatch"):
        mod.install_model_data("m", models, fw_main, "esp32s3")

    assert not (fw_main / "model_data.h").exists()


def test_install_model_data_rejects_sidecar_without_target(tmp_path):
    """Artefacto anterior al chequeo: no declara target, no se puede confiar."""
    import json as _json
    models = tmp_path / "models"
    models.mkdir()
    fw_main = tmp_path / "firmware" / "main"
    fw_main.mkdir(parents=True)
    (models / "m_model_data.h").write_text("// header viejo")
    (models / "m_model_data.meta.json").write_text(_json.dumps({"num_classes": 10}))

    mod = _load_module()
    with pytest.raises(mod.PreflightError, match="no declara 'target'"):
        mod.install_model_data("m", models, fw_main)

    assert not (fw_main / "model_data.h").exists()


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
        "target": "c",
    }))

    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)
    monkeypatch.chdir(tmp_path)

    mod = _load_module()
    with patch.object(mod, "_run_idf", return_value=0):
        rc = mod.main(["--source", "mymodel", "--action", "build"])
    assert rc == 0
    assert (fw_main / "model_data.h").exists()


def test_main_rejects_wrong_target_end_to_end(tmp_path, monkeypatch):
    """El build entero falla si el modelo es de otro chip, sin --preflight.

    Es el caso del flow fast_debug_esp32s3, que corre con --action build y sin
    preflight: el chequeo de target tiene que actuar igual.
    """
    models = tmp_path / "models"
    models.mkdir()
    fw = tmp_path / "firmware"
    fw_main = fw / "main"
    fw_main.mkdir(parents=True)
    (fw / "sdkconfig.defaults").write_text("CONFIG_ESP32_SPIRAM_SUPPORT=y\n")

    import json as _json
    (models / "mymodel_model_data.h").write_text("#define MODEL_NUM_CLASSES 10\n")
    (models / "mymodel_model_data.meta.json").write_text(_json.dumps({
        "num_classes": 10, "target": "c",
    }))

    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)
    monkeypatch.chdir(tmp_path)

    mod = _load_module()
    with patch.object(mod, "_run_idf", return_value=0) as run_idf:
        rc = mod.main(["--source", "mymodel", "--action", "build",
                       "--board", "esp32s3"])
    assert rc == 1
    run_idf.assert_not_called()
    assert not (fw_main / "model_data.h").exists()


def test_script04_help_lists_firmware_dir():
    r = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "--firmware-dir" in r.stdout


def test_firmware_dir_defaults_to_cwd_firmware(monkeypatch, tmp_path):
    """Sin la flag, el proyecto es ./firmware relativo al cwd (comportamiento previo)."""
    mod = _load_module()
    monkeypatch.chdir(tmp_path)
    args = mod.build_parser().parse_args([])
    assert mod.resolve_firmware_dir(args.firmware_dir) == tmp_path / "firmware"


def test_firmware_dir_relative_resolves_against_cwd(monkeypatch, tmp_path):
    mod = _load_module()
    monkeypatch.chdir(tmp_path)
    args = mod.build_parser().parse_args(["--firmware-dir", "../esp32-s3-cam/firmware"])
    expected = (tmp_path / "../esp32-s3-cam/firmware").resolve()
    assert mod.resolve_firmware_dir(args.firmware_dir) == expected


def test_firmware_dir_absolute_is_kept(tmp_path):
    mod = _load_module()
    target = tmp_path / "otro" / "firmware"
    args = mod.build_parser().parse_args(["--firmware-dir", str(target)])
    assert mod.resolve_firmware_dir(args.firmware_dir) == target.resolve()


def test_check_psram_accepts_s3_config_key(tmp_path):
    """El S3 usa CONFIG_SPIRAM=y, no CONFIG_ESP32_SPIRAM_SUPPORT=y."""
    mod = _load_module()
    fw = tmp_path / "firmware"
    fw.mkdir()
    _write_psram_board_headers(fw, board="esp32s3cam")
    (fw / "sdkconfig.defaults").write_text(
        'CONFIG_IDF_TARGET="esp32s3"\nCONFIG_SPIRAM=y\nCONFIG_SPIRAM_MODE_OCT=y\n',
        encoding="utf-8",
    )
    mod.check_psram_enabled(fw, "esp32s3cam")  # no debe lanzar


def test_check_psram_rejects_when_absent(tmp_path):
    mod = _load_module()
    fw = tmp_path / "firmware"
    fw.mkdir()
    _write_psram_board_headers(fw, board="esp32s3cam")
    (fw / "sdkconfig.defaults").write_text('CONFIG_IDF_TARGET="esp32s3"\n', encoding="utf-8")
    with pytest.raises(mod.PreflightError, match="PSRAM"):
        mod.check_psram_enabled(fw, "esp32s3cam")


# --- Important 1: validación de target del header YA instalado ---------------


def _write_installed_model(fw_main, target):
    import json as _json
    (fw_main / "model_data.h").write_text("#define MODEL_NUM_CLASSES 10\n")
    meta = {"num_classes": 10}
    if target is not None:
        meta["target"] = target
    (fw_main / "model_data.meta.json").write_text(_json.dumps(meta))


def test_check_installed_target_accepts_match(tmp_path):
    mod = _load_module()
    fw_main = tmp_path / "main"
    fw_main.mkdir()
    _write_installed_model(fw_main, "esp32s3")
    mod.check_installed_target(fw_main, "esp32s3")  # no debe lanzar


def test_check_installed_target_rejects_mismatch(tmp_path):
    mod = _load_module()
    fw_main = tmp_path / "main"
    fw_main.mkdir()
    _write_installed_model(fw_main, "c")
    with pytest.raises(mod.PreflightError, match="ya instalado"):
        mod.check_installed_target(fw_main, "esp32s3")


def test_check_installed_target_rejects_sidecar_without_target(tmp_path):
    mod = _load_module()
    fw_main = tmp_path / "main"
    fw_main.mkdir()
    _write_installed_model(fw_main, None)
    with pytest.raises(mod.PreflightError, match="sin 'target'"):
        mod.check_installed_target(fw_main, "esp32s3")


def test_check_installed_target_warns_without_sidecar(tmp_path, capsys):
    """Sin sidecar no se puede verificar, pero tampoco se rompe el build."""
    mod = _load_module()
    fw_main = tmp_path / "main"
    fw_main.mkdir()
    (fw_main / "model_data.h").write_text("#define MODEL_NUM_CLASSES 10\n")
    mod.check_installed_target(fw_main, "esp32s3")
    assert "warning" in capsys.readouterr().out


def test_main_build_without_source_rejects_wrong_installed_target(tmp_path, monkeypatch):
    """Un stage `build` sin `source` (re-build de lo ya instalado) tiene que
    fallar si el model_data.h que quedó es de otro chip."""
    fw = tmp_path / "firmware"
    fw_main = fw / "main"
    fw_main.mkdir(parents=True)
    (fw / "sdkconfig.defaults").write_text("CONFIG_SPIRAM=y\n")
    _write_installed_model(fw_main, "c")

    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)
    monkeypatch.chdir(tmp_path)

    mod = _load_module()
    with patch.object(mod, "_run_idf", return_value=0) as run_idf:
        rc = mod.main(["--action", "build", "--board", "esp32s3"])
    assert rc == 1
    run_idf.assert_not_called()


def test_main_build_without_source_accepts_matching_installed_target(tmp_path, monkeypatch):
    fw = tmp_path / "firmware"
    fw_main = fw / "main"
    fw_main.mkdir(parents=True)
    (fw / "sdkconfig.defaults").write_text("CONFIG_SPIRAM=y\n")
    _write_installed_model(fw_main, "esp32s3")

    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)
    monkeypatch.chdir(tmp_path)

    mod = _load_module()
    with patch.object(mod, "_run_idf", return_value=0) as run_idf:
        rc = mod.main(["--action", "build", "--board", "esp32s3"])
    assert rc == 0
    calls = [args[0] for args, _ in run_idf.call_args_list]
    assert any(cmd[-1] == "build" for cmd in calls)


def test_main_build_no_preflight_skips_installed_target_check(tmp_path, monkeypatch):
    """--no-preflight sigue siendo una salida legítima: documentado como tal."""
    fw = tmp_path / "firmware"
    fw_main = fw / "main"
    fw_main.mkdir(parents=True)
    _write_installed_model(fw_main, "c")

    monkeypatch.setenv("IDF_PATH", "/tmp/esp-idf")
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)
    monkeypatch.chdir(tmp_path)

    mod = _load_module()
    with patch.object(mod, "_run_idf", return_value=0) as run_idf:
        rc = mod.main(["--action", "build", "--board", "esp32s3", "--no-preflight"])
    assert rc == 0
    calls = [args[0] for args, _ in run_idf.call_args_list]
    assert any(cmd[-1] == "build" for cmd in calls)


# --- Minor 2: VID:PID del CH343 de la placa S3 -------------------------------


def test_esp_usb_ids_includes_ch343(tmp_path):
    mod = _load_module()
    assert (0x1A86, 0x55D3) in mod._ESP_USB_IDS


# --- Minor 8: mensaje de bootloader según la placa ---------------------------


def _flash_and_capture(mod, tmp_path, port, capsys, board="esp32cam"):
    import json as _json
    fw = tmp_path / "firmware"
    build = fw / mod.board_build_dir(board)
    build.mkdir(parents=True)
    (build / "flasher_args.json").write_text(_json.dumps(
        {"write_flash_args": ["--flash_mode", "dio"], "flash_files": {"0x1000": "bl.bin"}}))
    with patch("subprocess.call", return_value=0):
        mod._run_flash(fw, port, board)
    return capsys.readouterr().out


def test_flash_note_s3_mentions_auto_reset(tmp_path, capsys):
    mod = _load_module()
    out = _flash_and_capture(mod, tmp_path, "/dev/ttyACM0", capsys)
    assert "auto-reset" in out
    assert "[BOOT]" not in out


def test_flash_note_classic_mentions_boot_button(tmp_path, capsys):
    mod = _load_module()
    out = _flash_and_capture(mod, tmp_path, "/dev/ttyUSB0", capsys)
    assert "[BOOT]" in out


def test_flash_note_unknown_port_mentions_both(tmp_path, capsys):
    mod = _load_module()
    out = _flash_and_capture(mod, tmp_path, "/dev/foo0", capsys)
    assert "[BOOT]" in out and "auto-reset" in out


# ---------------------------------------------------------------------------
# MODEL_INPUT_EXPONENT: coherencia header <-> sidecar (con paréntesis en el
# header, formato `#define MODEL_INPUT_EXPONENT   (-6)`)
# ---------------------------------------------------------------------------

def _write_pair_with_exponent(firmware_main, exponent_header, exponent_meta):
    import json as _json
    firmware_main.mkdir(parents=True, exist_ok=True)
    (firmware_main / "model_data.h").write_text(
        MODEL_DATA_H_TEMPLATE.format(
            num_classes=10, input_w=32, input_h=32,
            input_channels=1, mean0=124, std0=58,
        ) + f"#define MODEL_INPUT_EXPONENT   ({exponent_header})\n"
    )
    (firmware_main / "model_data.meta.json").write_text(_json.dumps({
        "num_classes": 10, "input_w": 32, "input_h": 32,
        "input_channels": 1, "mean": [124], "std": [58],
        "classes": [f"c{i}" for i in range(10)],
        "input_exponent": exponent_meta,
    }))


def test_input_exponent_coherent(tmp_path):
    firmware_main = tmp_path / "firmware" / "main"
    _write_pair_with_exponent(firmware_main, -6, -6)
    mod = _load_module()
    mod.check_metadata_coherence(firmware_main)  # should not raise


def test_input_exponent_drift_detected(tmp_path):
    firmware_main = tmp_path / "firmware" / "main"
    _write_pair_with_exponent(firmware_main, -7, -6)
    mod = _load_module()
    with pytest.raises(mod.PreflightError, match="input_exponent"):
        mod.check_metadata_coherence(firmware_main)


def test_legacy_pair_without_exponent_still_coherent(tmp_path):
    """Artefactos viejos sin la clave en ninguno de los dos lados: OK."""
    firmware_main = tmp_path / "firmware" / "main"
    _write_pair(firmware_main)
    mod = _load_module()
    mod.check_metadata_coherence(firmware_main)  # should not raise
