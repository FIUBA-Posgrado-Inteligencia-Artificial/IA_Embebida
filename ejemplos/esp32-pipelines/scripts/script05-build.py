#!/usr/bin/env python3
"""Stage 'build' del pipeline ESP32-CAM.

`--action build` (default): pre-flight (opcional, on por default, sin
resolver puerto) → idf.py build. `--action flash`: chequeo de entorno IDF +
resolución de puerto + flasheo de lo ya compilado. `--action monitor`:
chequeo de entorno IDF + resolución de puerto + idf.py monitor.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path


class PreflightError(RuntimeError):
    """Falló un chequeo pre-build."""


BOARDS = {
    "esp32":      {"chip": "esp32",   "espdl_target": "c"},
    "esp32cam":   {"chip": "esp32",   "espdl_target": "c"},
    "esp32s3":    {"chip": "esp32s3", "espdl_target": "esp32s3"},
    "esp32s3cam": {"chip": "esp32s3", "espdl_target": "esp32s3"},
}


def board_build_dir(board: str) -> str:
    return f"build_{board}"


_IDF_EXPORT = Path.home() / "esp" / "esp-idf" / "export.sh"


def _try_load_idf_env() -> bool:
    """Intenta cargar el entorno IDF sourciando export.sh en bash y copiando las vars."""
    if not _IDF_EXPORT.exists():
        return False
    import subprocess
    # env -i: entorno limpio para que export.sh no falle por VIRTUAL_ENV de uv.
    system_path = "/usr/local/bin:/usr/bin:/bin"
    result = subprocess.run(
        ["env", "-i", f"HOME={Path.home()}", f"PATH={system_path}", "bash", "-c",
         f". {_IDF_EXPORT} > /dev/null 2>&1 && env"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False
    new_vars: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            new_vars[k] = v
    # PATH: prefijar con las rutas IDF que no estén ya presentes.
    if "PATH" in new_vars:
        existing = set(os.environ.get("PATH", "").split(":"))
        extras = [p for p in new_vars["PATH"].split(":") if p and p not in existing]
        if extras:
            os.environ["PATH"] = ":".join(extras) + ":" + os.environ.get("PATH", "")
    for k, v in new_vars.items():
        if k != "PATH":
            os.environ.setdefault(k, v)
    return True


def check_idf_env() -> None:
    if not os.environ.get("IDF_PATH"):
        if _try_load_idf_env():
            print(f"info: IDF_PATH loaded automatically from {_IDF_EXPORT}")
        else:
            raise PreflightError(
                f"IDF_PATH not set and {_IDF_EXPORT} not found. "
                "Source ~/esp/esp-idf/export.sh and re-run."
            )
    if shutil.which("idf.py") is None:
        raise PreflightError(
            "idf.py not found on PATH. Source ~/esp/esp-idf/export.sh and re-run."
        )


def check_model_data_present(firmware_main: Path) -> None:
    if not (firmware_main / "model_data.h").exists():
        raise PreflightError(
            f"{firmware_main / 'model_data.h'} not found. Run the 'quantize' stage first."
        )


_HEADER_INT_KEYS = {
    "MODEL_NUM_CLASSES": "num_classes",
    "MODEL_INPUT_W": "input_w",
    "MODEL_INPUT_H": "input_h",
    "MODEL_INPUT_CHANNELS": "input_channels",
    "MODEL_INPUT_EXPONENT": "input_exponent",
    "MODEL_INPUT_MEAN_0": ("mean", 0),
    "MODEL_INPUT_MEAN_1": ("mean", 1),
    "MODEL_INPUT_MEAN_2": ("mean", 2),
    "MODEL_INPUT_STD_0": ("std", 0),
    "MODEL_INPUT_STD_1": ("std", 1),
    "MODEL_INPUT_STD_2": ("std", 2),
}


def _parse_model_data_header(header_path: Path) -> dict:
    """Extrae constantes MODEL_* del header. Sólo las integer scalars."""
    text = header_path.read_text(encoding="utf-8", errors="replace")
    parsed: dict = {}
    for define, target in _HEADER_INT_KEYS.items():
        m = re.search(rf"#define\s+{define}\s+\(?\s*(-?\d+)\s*\)?", text)
        if not m:
            continue
        val = int(m.group(1))
        if isinstance(target, tuple):
            key, idx = target
            parsed.setdefault(key, {})[idx] = val
        else:
            parsed[target] = val
    return parsed


def check_metadata_coherence(firmware_main: Path) -> None:
    """Compara model_data.h con model_data.meta.json. Warn si falta sidecar."""
    header = firmware_main / "model_data.h"
    meta_path = firmware_main / "model_data.meta.json"
    if not meta_path.exists():
        print(f"warning: {meta_path} not found — skipping coherence check. "
              "Re-run the quantize stage to emit it.")
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    parsed = _parse_model_data_header(header)

    mismatches = []
    for scalar in ("num_classes", "input_w", "input_h", "input_channels",
                   "input_exponent"):
        if parsed.get(scalar) != meta.get(scalar):
            mismatches.append(f"{scalar}: header={parsed.get(scalar)} meta={meta.get(scalar)}")

    input_channels = meta.get("input_channels") or 1
    for key in ("mean", "std"):
        header_map = parsed.get(key, {})
        meta_list = meta.get(key) or []
        for idx in range(input_channels):
            header_val = header_map.get(idx)
            meta_val = meta_list[idx] if idx < len(meta_list) else None
            if header_val != meta_val:
                mismatches.append(f"{key}[{idx}]: header={header_val} meta={meta_val}")

    if mismatches:
        raise PreflightError(
            "model_data.h out of sync with model_data.meta.json:\n  - "
            + "\n  - ".join(mismatches)
            + "\nRe-run the quantize stage, then rebuild."
        )


def check_installed_target(firmware_main: Path, expected_target: str) -> None:
    """Valida el sidecar del model_data.h **ya instalado** contra `expected_target`.

    Complementa a `_check_target_matches`, que sólo cubre el camino en que el
    build instala el header (`--source`). Un stage `build` sin `source`
    —patrón que ya existe en `pipelines.yaml` para re-buildear sin
    re-cuantizar— compila lo que haya quedado en `firmware/main/`, que puede
    ser el modelo de otro chip. Acá el header ya está instalado, así que no
    hay nada que rechazar antes de copiar: lo que se rechaza es el build.

    Es parte del preflight, o sea que `--no-preflight` lo desactiva.
    """
    meta_path = firmware_main / "model_data.meta.json"
    regen = (f"Reinstalá el modelo del target correcto "
             f"(stage build con `source:` y `target: {expected_target}`, o "
             f"regenerá el modelo con el quantize de ese target).")
    if not meta_path.exists():
        print(f"warning: {meta_path} not found — no se puede verificar el "
              f"target del model_data.h ya instalado. Re-corré el stage "
              f"quantize para emitirlo.")
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise PreflightError(f"No se pudo leer el sidecar {meta_path}: {e}")

    actual = meta.get("target")
    if actual is None:
        raise PreflightError(
            f"El model_data.h ya instalado en {firmware_main} tiene un sidecar "
            f"sin 'target' — es un artefacto anterior al chequeo de target y no "
            f"hay forma de saber para qué chip está compilado. {regen}"
        )
    if actual != expected_target:
        raise PreflightError(
            f"Target mismatch: el model_data.h ya instalado en {firmware_main} "
            f"está compilado para '{actual}' pero se está buildeando para "
            f"'{expected_target}'. El .espdl NO es intercambiable entre targets "
            f"y montarlo en la placa equivocada no falla: la inferencia corre y "
            f"devuelve basura. {regen}"
        )


def _read_effective_psram_flag(firmware_dir: Path, board: str) -> int:
    """Resuelve FEATURE_PSRAM (firmware_config.h) considerando BOARD_HAS_PSRAM
    (boards/board_<board>.h) cuando FEATURE_PSRAM == -1 (auto)."""
    feature_path = firmware_dir / "main" / "firmware_config.h"
    board_path = firmware_dir / "main" / "boards" / f"board_{board}.h"

    feature = -1
    if feature_path.exists():
        text = feature_path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"#define\s+FEATURE_PSRAM\s+\(?\s*(-?\d+)\s*\)?", text)
        if m:
            feature = int(m.group(1))
    else:
        print(f"warning: {feature_path} not found; assuming FEATURE_PSRAM auto (-1)")

    if feature != -1:
        return feature

    board_has_psram = 0
    if board_path.exists():
        text = board_path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"#define\s+BOARD_HAS_PSRAM\s+\(?\s*(-?\d+)\s*\)?", text)
        if m:
            board_has_psram = int(m.group(1))
    else:
        print(f"warning: {board_path} not found; assuming BOARD_HAS_PSRAM 0")
    return board_has_psram


def check_psram_enabled(firmware_dir: Path, board: str) -> None:
    """Verifica que el **soporte** de PSRAM esté habilitado en sdkconfig(.defaults),
    pero sólo cuando la placa efectivamente lo necesita.

    El efectivo se resuelve como `FEATURE_PSRAM` (firmware_config.h) si no es
    `-1` (auto), y si no como `BOARD_HAS_PSRAM` del header de la placa
    (`boards/board_<board>.h`). Si el efectivo es 0, no hay nada que chequear
    acá — la placa no usa PSRAM.

    Cuando el efectivo es 1, acepta cualquiera de las dos llaves según la
    versión de IDF/target: `CONFIG_ESP32_SPIRAM_SUPPORT=y` (ESP32 clásico,
    IDF v4) o `CONFIG_SPIRAM=y` (IDF v5, ESP32-S3 y sucesores).

    Ojo: esto valida la **configuración**, no el hardware. Que el soporte esté
    compilado no implica que la placa tenga PSRAM funcional — el firmware del
    ESP32-S3 de este repo, por ejemplo, activa `CONFIG_SPIRAM=y` (lo necesita
    para que exista `esp_psram_is_initialized()`) sobre una placa cuya PSRAM no
    responde, y decide en runtime. La verificación real es el log de boot.
    """
    if _read_effective_psram_flag(firmware_dir, board) == 0:
        return

    candidates = [
        firmware_dir / board_build_dir(board) / "sdkconfig",
        firmware_dir / "sdkconfig",
        firmware_dir / "sdkconfig.defaults",
    ] + sorted(firmware_dir.glob("sdkconfig.defaults.*"))
    present = [p for p in candidates if p.exists()]
    if not present:
        raise PreflightError(
            f"No sdkconfig or sdkconfig.defaults found under {firmware_dir}. "
            "Run `idf.py menuconfig` first."
        )
    for cfg in present:
        text = cfg.read_text(encoding="utf-8", errors="replace")
        # ESP-IDF v4/v5 use different config key names for PSRAM support.
        if "CONFIG_ESP32_SPIRAM_SUPPORT=y" in text or "CONFIG_SPIRAM=y" in text:
            return
    raise PreflightError(
        "PSRAM not enabled. Run `idf.py menuconfig → Component config → ESP PSRAM` "
        "and set `Support for external, SPI-connected RAM`."
    )


# Chips USB-serie usados por placas ESP32. Filtramos por (VID, PID) para
# excluir falsos positivos como el FT2232H de los debuggers Digilent (0403:6010).
_ESP_USB_IDS = {
    (0x10C4, 0xEA60),  # CP210x — ESP-WROOM, DevKitC
    (0x1A86, 0x7523),  # CH340 — ESP32-CAM-MB, placas genéricas
    (0x1A86, 0x55D3),  # CH343 — ESP32-S3-CAM (USB-C rotulado UART, /dev/ttyACM*)
    (0x1A86, 0x55D4),  # CH9102 — ESP32-S3 y otras placas nuevas
    (0x0403, 0x6001),  # FT232R — algunas placas dev ESP
    (0x0403, 0x6015),  # FT231X — placas ESP nuevas con FTDI
}


def _detect_serial_ports() -> list[str]:
    """Detecta puertos serie USB. Con pyserial disponible, prefiere los que
    tienen (VID, PID) conocido de placas ESP; si no hay match o pyserial
    falta, cae al glob genérico sobre /dev/ttyUSB*/ttyACM*."""
    try:
        from serial.tools import list_ports
        esp_ports = [
            p.device for p in list_ports.comports()
            if (p.vid, p.pid) in _ESP_USB_IDS
            and p.device.startswith(("/dev/ttyUSB", "/dev/ttyACM"))
        ]
        if esp_ports:
            return sorted(esp_ports)
    except ImportError:
        pass
    return sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))


def resolve_port(port: str, candidates: list[str] | None = None) -> str:
    """Si port == 'auto', autodetecta por (VID, PID) de chip ESP; error si 0
    o >1 candidatos. Si se pasa un path explícito, se usa tal cual."""
    if port != "auto":
        return port
    if candidates is None:
        candidates = _detect_serial_ports()
    if not candidates:
        raise PreflightError(
            "--port auto: no serial ports found under /dev/ttyUSB* or /dev/ttyACM*. "
            "Connect the ESP32-CAM or pass --port explicitly."
        )
    if len(candidates) > 1:
        raise PreflightError(
            "--port auto: multiple serial ports detected:\n"
            + "\n".join(f"  {c}  ({_port_description(c)})" for c in candidates)
            + "\nHardcodealo en pipelines.yaml (`port: /dev/ttyUSBN` top-level) "
              "o pasá --port explícitamente."
        )
    return candidates[0]


def _port_description(device: str) -> str:
    """Descripción legible del puerto (VID:PID + descripción pyserial). Fallback: ''."""
    try:
        from serial.tools import list_ports
        for p in list_ports.comports():
            if p.device == device:
                vid = f"{p.vid:04X}" if p.vid else "????"
                pid = f"{p.pid:04X}" if p.pid else "????"
                return f"{vid}:{pid} {p.description or ''}".strip()
    except ImportError:
        pass
    return ""


def install_model_data(source: str, models_dir: Path, firmware_main: Path,
                       expected_target: str = "c") -> None:
    """Copies {source}_model_data.h (and sidecar) to firmware/main/model_data.h.

    Orden de búsqueda del header staged:

    1. `firmware_main` — es donde lo deja `script03*-quantize.py` cuando se le
       pasa `--firmware-dir <proyecto>/main` (caso del port a ESP32-S3).
    2. `models_dir` — destino por defecto de `script03*` (caso ESP32 clásico).

    El orden importa y no es cosmético: el nombre del header **no codifica el
    target**, así que `models/<source>_model_data.h` puede ser un `.espdl`
    compilado para otro chip. Si `models_dir` ganara, un flow con
    `--target esp32s3` instalaría silenciosamente el binario del ESP32 clásico
    y la inferencia daría basura sin ningún error. Priorizando el header que
    quedó staged dentro del propio proyecto firmware, cada placa consume el
    suyo.

    Pero la prioridad sola no alcanza, y de hecho abre su propia variante del
    mismo problema: `--firmware-dir` es una flag pública, así que basta con
    correr una vez `script03* --firmware-dir firmware/main` sobre el proyecto
    clásico para que ese `<source>_model_data.h` quede ahí y le gane para
    siempre a todo `quantize` posterior que escriba en `models/`. Sería el
    mismo modo de falla — sin error, modelo equivocado — y encima el archivo
    pegajoso está gitignoreado, o sea invisible en `git status`.

    Por eso el chequeo real no es el orden sino `expected_target`: el sidecar
    ahora lleva la llave `target` y acá se verifica contra la placa para la
    que se está buildeando. Se valida **siempre**, no sólo con `--preflight`,
    porque el flow del S3 corre con `--build-only` y sin preflight — es decir,
    justo el caso donde más importa. Un header sin `target` en el sidecar (o
    sin sidecar) se rechaza: es un artefacto anterior a este chequeo y no hay
    forma de saber para qué chip está compilado. Regenerarlo cuesta segundos;
    embeber el modelo equivocado cuesta una sesión de debugging.

    Uses shutil.copy (not copy2) deliberately: copy2 preserves source mtime,
    which can leave the destination header OLDER than the existing
    inference.cpp.obj — make/ninja then skip recompilation and the old
    model stays embedded in the final binary. copy updates mtime to 'now',
    forcing a rebuild every time a different model is installed.
    """
    for candidate in (firmware_main, models_dir):
        header_src = candidate / f"{source}_model_data.h"
        if header_src.exists():
            meta_src = candidate / f"{source}_model_data.meta.json"
            break
    else:
        raise PreflightError(
            f"Model header not found: {models_dir / f'{source}_model_data.h'} "
            f"(tampoco en {firmware_main}). "
            "Run the 'quantize' stage first."
        )

    _check_target_matches(header_src, meta_src, expected_target)

    shutil.copy(header_src, firmware_main / "model_data.h")
    print(f"info: installed {header_src.name} → {firmware_main / 'model_data.h'} "
          f"(target={expected_target})")
    shutil.copy(meta_src, firmware_main / "model_data.meta.json")
    print(f"info: installed {meta_src.name} → {firmware_main / 'model_data.meta.json'}")

    # Golden images del self-test (opcionales): viajan con el modelo para que
    # la referencia pc_q corresponda siempre al .espdl instalado.
    test_src = header_src.parent / f"{source}_test_images.h"
    test_dst = firmware_main / "test_images.h"
    if test_src.exists():
        shutil.copy(test_src, test_dst)
        print(f"info: installed {test_src.name} → {test_dst}")
    elif test_dst.exists():
        # No dejar golden images de OTRO modelo instaladas: referencia stale.
        test_dst.unlink()
        print("warning: no test_images.h for this model — removed stale one "
              "(self-test disabled until re-quantize)")


def _check_target_matches(header_src: Path, meta_src: Path,
                          expected_target: str) -> None:
    """Rechaza un header cuyo sidecar no declare el target esperado."""
    regen = (f"Regenerá el modelo con el quantize del target correcto "
             f"(--target {expected_target}).")
    if not meta_src.exists():
        raise PreflightError(
            f"Sidecar no encontrado: {meta_src}. Sin él no se puede verificar "
            f"para qué chip está compilado {header_src.name}, y montar el "
            f"target equivocado no da error: da predicciones basura. {regen}"
        )
    try:
        meta = json.loads(meta_src.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise PreflightError(f"No se pudo leer el sidecar {meta_src}: {e}")

    actual = meta.get("target")
    if actual is None:
        raise PreflightError(
            f"El sidecar {meta_src.name} no declara 'target' — es un artefacto "
            f"anterior al chequeo de target. No hay forma de saber para qué "
            f"chip está compilado. {regen}"
        )
    if actual != expected_target:
        raise PreflightError(
            f"Target mismatch: {header_src} está compilado para "
            f"'{actual}' pero se está buildeando para '{expected_target}'. "
            f"El .espdl NO es intercambiable entre targets y montarlo en la "
            f"placa equivocada no falla: la inferencia corre y devuelve "
            f"basura. {regen}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="script05-build.py", description=__doc__)
    p.add_argument("--source", default=None,
                   help="Model name (without extension) to install before building. "
                        "Copies models/{source}_model_data.h → firmware/main/model_data.h.")
    p.add_argument("--port", default="auto",
                   help="Serial port for flashing. 'auto' → autodetect. Default: auto")
    p.add_argument("--action", default="build", choices=["build", "flash", "monitor"],
                   help="Qué hacer: build (compila), flash (flashea lo ya compilado), "
                        "monitor (abre monitor serie). Default: build")
    p.add_argument("--preflight", action=argparse.BooleanOptionalAction, default=True,
                   help="Ejecuta chequeos pre-build (default: on)")
    p.add_argument("--menuconfig-only", action="store_true",
                   help="Sólo abre idf.py menuconfig; no compila ni flashea")
    p.add_argument("--yes", action="store_true",
                   help="Auto-confirma prompts de preflight (útil en CI)")
    p.add_argument("--firmware-dir", type=Path, default=Path("firmware"),
                   help="Proyecto ESP-IDF a buildear. Relativo al cwd o absoluto. "
                        "(default: firmware)")
    p.add_argument("--board", default="esp32cam", choices=sorted(BOARDS),
                   help="Placa para la que se buildea/flashea/monitorea. "
                        "Determina chip, build dir (build_<board>) y el "
                        "target esp-dl esperado del modelo instalado: se "
                        "rechaza instalar/buildear un header compilado para "
                        "otra placa. (default: esp32cam)")
    return p


def _build_subprocess_env() -> dict:
    """Subprocess env with uv markers stripped and IDF python venv at front of PATH."""
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    idf_py_env = env.get("IDF_PYTHON_ENV_PATH", "")
    if idf_py_env:
        idf_bin = str(Path(idf_py_env) / "bin")
        env["PATH"] = idf_bin + ":" + env.get("PATH", "")
    return env


def _run_idf(cmd: list[str], cwd: Path) -> int:
    import subprocess
    env = _build_subprocess_env()
    print(f"$ {' '.join(cmd)}  (cwd={cwd})", flush=True)
    return subprocess.call(cmd, cwd=cwd, env=env)


def _run_flash(firmware_dir: Path, port: str, board: str = "esp32cam") -> int:
    """Flash via esptool with --before no_reset (AI-Thinker MB has no auto-reset circuit)."""
    import json, subprocess
    build_dir = firmware_dir / board_build_dir(board)
    flasher_json = build_dir / "flasher_args.json"
    if not flasher_json.exists():
        print(f"error: {build_dir.name}/flasher_args.json not found — run build first.",
              file=sys.stderr)
        return 1

    fa = json.loads(flasher_json.read_text())
    write_flash_args = fa.get("write_flash_args", [])
    flash_files = fa.get("flash_files", {})

    cmd = [
        "esptool.py", "--chip", BOARDS[board]["chip"],
        "-p", port, "-b", "115200",
        "--before", "no_reset",
        "--after", "hard_reset",
        "write_flash", *write_flash_args,
    ]
    for addr, fname in flash_files.items():
        cmd.extend([addr, str(build_dir / fname)])

    print()
    # El único dato de placa disponible acá es el puerto. La ESP32-CAM clásica
    # va sobre el adaptador ESP32-CAM-MB (CH341 → /dev/ttyUSB*) y no tiene
    # auto-reset; la ESP32-S3-CAM expone el CH343 como /dev/ttyACM* y sí lo
    # tiene. Se usa esa heurística en vez de una flag nueva para no cambiar la
    # CLI, y ante un puerto que no matchea ninguno de los dos se imprimen los
    # dos casos en vez de asumir.
    if port.startswith("/dev/ttyACM"):
        print("NOTE: this board (CH343 on /dev/ttyACM*) has auto-reset — "
              "no button press needed.")
    elif port.startswith("/dev/ttyUSB"):
        print("NOTE: ESP32-CAM-MB requires manual bootloader mode before flashing.")
        print("  1. Unplug USB  2. Hold [BOOT]  3. Plug USB  4. Release [BOOT]")
    else:
        print("NOTE: bootloader mode depends on the board.")
        print("  - ESP32-CAM-MB (CH341, /dev/ttyUSB*): no auto-reset — "
              "unplug USB, hold [BOOT], plug USB, release [BOOT].")
        print("  - ESP32-S3-CAM (CH343, /dev/ttyACM*): auto-reset — "
              "no button press needed.")
    print()
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, env=_build_subprocess_env())


def _run_preflight(firmware_dir: Path, board: str) -> None:
    """Chequeos pre-build. `check_installed_target` va acá (y no sólo en
    `install_model_data`) para que el camino sin `--source` —el stage `build`
    que re-buildea lo ya instalado— también quede cubierto."""
    expected_target = BOARDS[board]["espdl_target"]
    check_idf_env()
    check_model_data_present(firmware_dir / "main")
    check_installed_target(firmware_dir / "main", expected_target)
    check_metadata_coherence(firmware_dir / "main")
    check_psram_enabled(firmware_dir, board)


def _ensure_set_target(firmware_dir: Path, board: str) -> int:
    """Corre `idf.py set-target` si el build dir del board no existe o su
    sdkconfig no declara el chip esperado. Devuelve el returncode de
    set-target (0 si no hizo falta correrlo)."""
    chip = BOARDS[board]["chip"]
    bdir = board_build_dir(board)
    sdkconfig = firmware_dir / bdir / "sdkconfig"
    needs_set_target = True
    if sdkconfig.exists():
        text = sdkconfig.read_text(encoding="utf-8", errors="replace")
        if f'CONFIG_IDF_TARGET="{chip}"' in text:
            needs_set_target = False
    if not needs_set_target:
        return 0
    return _run_idf(
        ["idf.py", "-B", bdir, f"-DBOARD={board}",
         f"-DSDKCONFIG={bdir}/sdkconfig", "set-target", chip],
        cwd=firmware_dir,
    )


def resolve_firmware_dir(raw: Path) -> Path:
    """Resuelve el path del proyecto ESP-IDF (relativo al cwd o absoluto)."""
    return (Path.cwd() / raw).resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    firmware_dir = resolve_firmware_dir(args.firmware_dir)

    if args.menuconfig_only:
        try:
            check_idf_env()
        except PreflightError as e:
            print(f"preflight error: {e}", file=sys.stderr)
            return 1
        return _run_idf(["idf.py", "menuconfig"], cwd=firmware_dir)

    try:
        if args.action == "build":
            if args.source:
                install_model_data(args.source, Path.cwd() / "models",
                                   firmware_dir / "main",
                                   BOARDS[args.board]["espdl_target"])
            if args.preflight:
                _run_preflight(firmware_dir, args.board)
            else:
                check_idf_env()
            rc = _ensure_set_target(firmware_dir, args.board)
            if rc != 0:
                return rc
            bdir = board_build_dir(args.board)
            return _run_idf(
                ["idf.py", "-B", bdir, f"-DBOARD={args.board}",
                 f"-DSDKCONFIG={bdir}/sdkconfig", "build"],
                cwd=firmware_dir,
            )

        if args.action == "flash":
            check_idf_env()
            port = resolve_port(args.port)
            return _run_flash(firmware_dir, port, args.board)

        # monitor
        check_idf_env()
        port = resolve_port(args.port)
        bdir = board_build_dir(args.board)
        return _run_idf(["idf.py", "-B", bdir, "-p", port, "monitor"], cwd=firmware_dir)
    except PreflightError as e:
        print(f"preflight error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
