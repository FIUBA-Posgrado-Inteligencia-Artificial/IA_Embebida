#!/usr/bin/env python3
"""Stage 'build' variante OTA del pipeline ESP32-CAM.

Compila firmware_ota/ y sube el binario por HTTP a http://<ip>/ota del device
ya corriendo una firmware OTA-capable. El primer flash de cada placa sigue
requiriendo USB (la MB de AI-Thinker no tiene circuito auto-reset) — usá
--initial para esa primera vez; todas las siguientes son inalámbricas.

Modos:
  default              build + HTTP POST /ota (requiere --ip)
  --initial            build + USB flash (script04-build.py-style, BOOT button)
  --build-only         sólo compila
  --info-only          GET /ota/info al device (no compila)
  --menuconfig-only    idf.py menuconfig sobre firmware_ota/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


FIRMWARE_DIR_NAME = "firmware_ota"
DEFAULT_OTA_TIMEOUT = 180


def _import_script04():
    """script04-build.py has a dash in its filename so we can't `import` it."""
    spec = importlib.util.spec_from_file_location(
        "_script04_build", Path(__file__).parent / "script04-build.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_s04 = _import_script04()
PreflightError = _s04.PreflightError


def _upload_ota(ip: str, bin_path: Path, timeout: int) -> dict:
    url = f"http://{ip}/ota"
    size = bin_path.stat().st_size
    print(f"info: POST {url}  ({bin_path.name}, {size:,} bytes)")
    t0 = time.time()
    data = bin_path.read_bytes()
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(data)),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body.strip()}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"connection failed: {e.reason}")
    dt = time.time() - t0
    rate = (size / dt / 1024) if dt > 0 else 0
    print(f"info: upload OK in {dt:.1f}s ({rate:.1f} KiB/s)")
    print(f"info: device → {body.strip()}")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


def _query_info(ip: str, timeout: int = 10) -> None:
    url = f"http://{ip}/ota/info"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    print(body.strip())


def _find_app_binary(build_dir: Path) -> Path:
    """Locate the app .bin via flasher_args.json."""
    flasher = build_dir / "flasher_args.json"
    if not flasher.exists():
        raise PreflightError(
            f"{flasher} not found — run `idf.py build` first "
            "(or drop --build-only-skip)."
        )
    fa = json.loads(flasher.read_text())
    fname = fa.get("app", {}).get("file")
    if not fname:
        raise PreflightError(
            f"{flasher} has no 'app.file' key — unexpected IDF layout."
        )
    path = build_dir / fname
    if not path.exists():
        raise PreflightError(f"app binary not found: {path}")
    return path


def _run_initial_flash(firmware_dir: Path, port: str) -> int:
    """First-time USB flash — delegates to script04's no_reset sequence."""
    return _s04._run_flash(firmware_dir, port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="script04b-build_ota.py", description=__doc__)
    p.add_argument(
        "--source",
        default=None,
        help="Model name (without extension) to install before building. "
        "Copies models/{source}_model_data.h → firmware_ota/main/model_data.h.",
    )
    p.add_argument(
        "--ip",
        default=None,
        help="Device IP for OTA upload. Required in default mode. "
        "Not needed for --initial/--build-only/--menuconfig-only.",
    )
    p.add_argument(
        "--port",
        default="auto",
        help="Serial port for --initial flash. 'auto' → autodetect.",
    )
    p.add_argument(
        "--preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ejecuta chequeos pre-build (default: on)",
    )
    p.add_argument("--build-only", action="store_true", help="Sólo compila")
    p.add_argument(
        "--initial",
        action="store_true",
        help="USB flash (primera vez). Requiere BOOT button en placas "
        "sin auto-reset.",
    )
    p.add_argument(
        "--info-only",
        action="store_true",
        help="GET /ota/info al device y salir (no compila).",
    )
    p.add_argument("--menuconfig-only", action="store_true")
    p.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_OTA_TIMEOUT,
        help=f"OTA upload timeout en segundos (default: {DEFAULT_OTA_TIMEOUT})",
    )
    p.add_argument("--yes", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    firmware_dir = Path.cwd() / FIRMWARE_DIR_NAME

    if not firmware_dir.exists():
        print(
            f"error: {firmware_dir} not found. "
            "Run this script from examples/esp32-cam/.",
            file=sys.stderr,
        )
        return 1

    exclusive = [
        args.build_only,
        args.initial,
        args.info_only,
        args.menuconfig_only,
    ]
    if sum(bool(f) for f in exclusive) > 1:
        print(
            "error: --build-only, --initial, --info-only y --menuconfig-only "
            "son mutuamente excluyentes",
            file=sys.stderr,
        )
        return 2

    # --info-only: no compila ni preflight pesado, sólo consulta.
    if args.info_only:
        if not args.ip:
            print("error: --info-only requires --ip", file=sys.stderr)
            return 2
        try:
            _query_info(args.ip)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    if args.menuconfig_only:
        try:
            _s04.check_idf_env()
        except PreflightError as e:
            print(f"preflight error: {e}", file=sys.stderr)
            return 1
        return _s04._run_idf(["idf.py", "menuconfig"], cwd=firmware_dir)

    # Install model_data.h into firmware_ota/main/ if asked.
    if args.source:
        try:
            _s04.install_model_data(
                args.source, Path.cwd() / "models", firmware_dir / "main"
            )
        except PreflightError as e:
            print(f"preflight error: {e}", file=sys.stderr)
            return 1

    try:
        _s04.check_idf_env()
        if args.preflight:
            _s04.check_model_data_present(firmware_dir / "main")
            _s04.check_metadata_coherence(firmware_dir / "main")
            _s04.check_psram_enabled(firmware_dir)
    except PreflightError as e:
        print(f"preflight error: {e}", file=sys.stderr)
        return 1

    rc = _s04._run_idf(["idf.py", "build"], cwd=firmware_dir)
    if rc != 0:
        print("build failed.", file=sys.stderr)
        return rc

    if args.build_only:
        return 0

    if args.initial:
        try:
            port = _s04.resolve_port(args.port)
        except PreflightError as e:
            print(f"preflight error: {e}", file=sys.stderr)
            return 1
        return _run_initial_flash(firmware_dir, port)

    if not args.ip:
        print(
            "error: --ip <device_ip> es requerido para OTA "
            "(o usá --initial/--build-only)",
            file=sys.stderr,
        )
        return 2

    try:
        bin_path = _find_app_binary(firmware_dir / "build")
    except PreflightError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        _upload_ota(args.ip, bin_path, timeout=args.timeout)
    except Exception as e:
        print(f"OTA upload failed: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
