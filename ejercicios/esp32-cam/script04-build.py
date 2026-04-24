#!/usr/bin/env python3
"""Stage 'build' del pipeline ESP32-CAM.

Pre-flight (opcional, on por default) → idf.py build → idf.py flash →
opcionalmente idf.py monitor.

La lógica de los chequeos individuales aterriza en Tasks 18-20 y la
secuencia build/flash/monitor en Task 21.
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
        m = re.search(rf"#define\s+{define}\s+(-?\d+)", text)
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
    for scalar in ("num_classes", "input_w", "input_h", "input_channels"):
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


def check_psram_enabled(firmware_dir: Path) -> None:
    """Verifica que CONFIG_ESP32_SPIRAM_SUPPORT=y esté en sdkconfig o sdkconfig.defaults."""
    candidates = [firmware_dir / "sdkconfig", firmware_dir / "sdkconfig.defaults"]
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


def install_model_data(source: str, models_dir: Path, firmware_main: Path) -> None:
    """Copies {source}_model_data.h (and sidecar) from models/ to firmware/main/.

    Uses shutil.copy (not copy2) deliberately: copy2 preserves source mtime,
    which can leave the destination header OLDER than the existing
    inference.cpp.obj — make/ninja then skip recompilation and the old
    model stays embedded in the final binary. copy updates mtime to 'now',
    forcing a rebuild every time a different model is installed.
    """
    header_src = models_dir / f"{source}_model_data.h"
    meta_src = models_dir / f"{source}_model_data.meta.json"
    if not header_src.exists():
        raise PreflightError(
            f"Model header not found: {header_src}. "
            "Run the 'quantize' stage first."
        )
    shutil.copy(header_src, firmware_main / "model_data.h")
    print(f"info: installed {header_src.name} → {firmware_main / 'model_data.h'}")
    if meta_src.exists():
        shutil.copy(meta_src, firmware_main / "model_data.meta.json")
        print(f"info: installed {meta_src.name} → {firmware_main / 'model_data.meta.json'}")
    else:
        print(f"warning: sidecar {meta_src.name} not found — skipping meta copy")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="script04-build.py", description=__doc__)
    p.add_argument("--source", default=None,
                   help="Model name (without extension) to install before building. "
                        "Copies models/{source}_model_data.h → firmware/main/model_data.h.")
    p.add_argument("--port", default="auto",
                   help="Serial port for flashing. 'auto' → autodetect. Default: auto")
    p.add_argument("--monitor", action="store_true",
                   help="Abre idf.py monitor al final del flash")
    p.add_argument("--preflight", action=argparse.BooleanOptionalAction, default=True,
                   help="Ejecuta chequeos pre-build (default: on)")
    p.add_argument("--build-only", action="store_true",
                   help="Sólo compila; no flashea ni monitorea")
    p.add_argument("--monitor-only", action="store_true",
                   help="Sólo abre idf.py monitor; no compila ni flashea")
    p.add_argument("--menuconfig-only", action="store_true",
                   help="Sólo abre idf.py menuconfig; no compila ni flashea")
    p.add_argument("--yes", action="store_true",
                   help="Auto-confirma prompts de preflight (útil en CI)")
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


def _run_flash(firmware_dir: Path, port: str) -> int:
    """Flash via esptool with --before no_reset (AI-Thinker MB has no auto-reset circuit)."""
    import json, subprocess
    build_dir = firmware_dir / "build"
    flasher_json = build_dir / "flasher_args.json"
    if not flasher_json.exists():
        print("error: build/flasher_args.json not found — run build first.", file=sys.stderr)
        return 1

    fa = json.loads(flasher_json.read_text())
    write_flash_args = fa.get("write_flash_args", [])
    flash_files = fa.get("flash_files", {})

    cmd = [
        "esptool.py", "--chip", "esp32",
        "-p", port, "-b", "115200",
        "--before", "no_reset",
        "--after", "hard_reset",
        "write_flash", *write_flash_args,
    ]
    for addr, fname in flash_files.items():
        cmd.extend([addr, str(build_dir / fname)])

    print()
    print("NOTE: ESP32-CAM-MB requires manual bootloader mode before flashing.")
    print("  1. Unplug USB  2. Hold [BOOT]  3. Plug USB  4. Release [BOOT]")
    print()
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, env=_build_subprocess_env())


def _run_preflight(firmware_dir: Path, port: str, yes: bool) -> str:
    """Ejecuta todos los chequeos. Devuelve el port resuelto."""
    check_idf_env()
    check_model_data_present(firmware_dir / "main")
    check_metadata_coherence(firmware_dir / "main")
    check_psram_enabled(firmware_dir)
    return resolve_port(port)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    firmware_dir = Path.cwd() / "firmware"

    exclusive = [args.build_only, args.monitor_only, args.menuconfig_only]
    if sum(bool(f) for f in exclusive) > 1:
        print("error: --build-only, --monitor-only y --menuconfig-only "
              "son mutuamente excluyentes", file=sys.stderr)
        return 2

    if args.menuconfig_only:
        try:
            check_idf_env()
        except PreflightError as e:
            print(f"preflight error: {e}", file=sys.stderr)
            return 1
        return _run_idf(["idf.py", "menuconfig"], cwd=firmware_dir)

    if args.monitor_only:
        try:
            check_idf_env()
            port = resolve_port(args.port)
        except PreflightError as e:
            print(f"preflight error: {e}", file=sys.stderr)
            return 1
        return _run_idf(["idf.py", "-p", port, "monitor"], cwd=firmware_dir)

    if args.source:
        try:
            install_model_data(args.source, Path.cwd() / "models", firmware_dir / "main")
        except PreflightError as e:
            print(f"preflight error: {e}", file=sys.stderr)
            return 1

    try:
        if args.preflight:
            port = _run_preflight(firmware_dir, args.port, args.yes)
        else:
            port = resolve_port(args.port)
    except PreflightError as e:
        print(f"preflight error: {e}", file=sys.stderr)
        return 1

    rc = _run_idf(["idf.py", "build"], cwd=firmware_dir)
    if rc != 0:
        print("build failed.", file=sys.stderr)
        return rc

    if args.build_only:
        return 0

    rc = _run_flash(firmware_dir, port)
    if rc != 0:
        print("flash failed.", file=sys.stderr)
        return rc

    if args.monitor:
        rc = _run_idf(["idf.py", "-p", port, "monitor"], cwd=firmware_dir)
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
