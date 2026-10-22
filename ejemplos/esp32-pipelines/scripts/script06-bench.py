"""
ESP32 — script06-bench: mide tiempos de inferencia en placa via self-test
==========================================================================
Flashea el modelo indicado en la placa indicada, dispara el self-test por
serie (tecla 't') y registra en `benchmarks.json` la media/min/max de
`t_inference` de las 10 golden images, junto con PASS n/10 y dlogits máximo.
`main.py bench` muestra las mediciones; `main.py models` solo marca en la columna `Bench` qué placas tienen datos.

Requisitos: el modelo ya cuantizado para el target de la placa (el build lo
valida contra el sidecar) y las golden images emitidas (script04a las genera).

Uso:
    uv run python scripts/script06-bench.py --source depthwise_rgb32_pruned_activation_50_ft10_quantized_ptq \
        --board esp32s3cam --port /dev/ttyACM0
    uv run python scripts/script06-bench.py --source ... --board esp32cam --port /dev/ttyUSB0
        # esp32cam: la MB no tiene auto-reset — el script espera a que hagas
        # BOOT+RST durante la ventana de conexión de esptool, y RST al final.

Flags:
    --source       [requerido] modelo cuantizado (sin extensión)
    --board        placa (tabla BOARDS de script05-build.py; default esp32cam)
    --port         puerto serie (default /dev/ttyUSB0)
    --runtime      runtime medido: espdl|tflm|tflm_espnn|onnx2c|executorch (default espdl)
    --build-dir    override del build dir a flashear (default firmware/build_<board>/)
    --skip-build   no rebuildear (usa build_<board>/ tal cual)
    --skip-flash   no flashear (medir lo que ya corre en la placa)
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "pipelib"))

# Placas cuyo puente serie NO tiene auto-reset: esptool espera con ventana
# larga a que el usuario haga BOOT+RST, y el boot final requiere RST manual.
_MANUAL_RESET_BOARDS = {"esp32cam", "esp32"}

_SELFTEST_LINE = re.compile(
    r"^\s*(\w+)\s+->\s+(\w+)\s+([\d.]+)%.*dlogits<=(\d+)\s+(\d+)ms\s+(PASS|FAIL)"
)


def parse_args():
    from importlib import util as _u
    spec = _u.spec_from_file_location("script05_build", HERE / "script05-build.py")
    s5 = _u.module_from_spec(spec)
    spec.loader.exec_module(s5)

    p = argparse.ArgumentParser(description="Benchmark de inferencia en placa (self-test).")
    p.add_argument("--source", required=True,
                   help="[requerido] Modelo cuantizado (sin extensión)")
    p.add_argument("--board", default="esp32cam", choices=sorted(s5.BOARDS),
                   help="Placa destino (default: esp32cam)")
    p.add_argument("--port", default="/dev/ttyUSB0",
                   help="Puerto serie (default: /dev/ttyUSB0)")
    p.add_argument("--runtime", default="espdl",
                   choices=["espdl", "tflm", "tflm_espnn", "onnx2c", "executorch"],
                   help="Runtime de inferencia medido (default: espdl)")
    p.add_argument("--build-dir", default=None,
                   help="Override del build dir a flashear (default: "
                        "firmware/build_<board>/; útil para firmware_rt/<rt>/)")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-flash", action="store_true")
    args = p.parse_args()
    args._s5 = s5
    return args


def _run(cmd: list[str], **kw) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, **kw)


def build(args) -> None:
    rc = _run(["uv", "run", "python", str(HERE / "script05-build.py"),
               "--action", "build", "--board", args.board,
               "--source", args.source])
    if rc != 0:
        sys.exit(f"build falló (rc={rc})")


def flash(args) -> None:
    chip = args._s5.BOARDS[args.board]["chip"]
    if args.build_dir:
        bdir = Path(args.build_dir)
        if not bdir.is_absolute():
            bdir = ROOT / bdir
    else:
        bdir = ROOT / "firmware" / args._s5.board_build_dir(args.board)
    manual = args.board in _MANUAL_RESET_BOARDS
    # esptool.py del entorno IDF (el venv de uv no lo tiene): mismo env
    # saneado que usa script05 para sus subprocesos idf.py/esptool.
    args._s5.check_idf_env()
    env = args._s5._build_subprocess_env()
    cmd = ["esptool.py", "--chip", chip, "-p", args.port,
           "-b", "115200" if manual else "460800",
           "--before", "no_reset" if manual else "default_reset",
           "--after", "hard_reset"]
    if manual:
        cmd += ["--connect-attempts", "240"]
        print("\n*** Esta placa no tiene auto-reset: mantené BOOT, tocá RST, "
              "soltá BOOT (ventana ~8 min) ***\n", flush=True)
    cmd += ["write_flash", "@flash_args"]
    rc = _run(cmd, cwd=bdir, env=env)
    if rc != 0:
        sys.exit(f"flash falló (rc={rc})")


def measure(args) -> dict:
    import serial

    s = serial.Serial()
    s.port = args.port
    s.baudrate = 115200
    s.timeout = 1
    s.rts = False
    s.dtr = False
    s.open()

    manual = args.board in _MANUAL_RESET_BOARDS
    if manual:
        print("*** Apretá RST para bootear la placa ***", flush=True)
    print("esperando boot...", flush=True)
    buf = b""
    t0 = time.time()
    while time.time() - t0 < 180:
        buf += s.read(4096)
        if b"ready" in buf:
            break
    if b"ready" not in buf:
        s.close()
        sys.exit("la placa no reportó 'ready' en 180s")

    time.sleep(1)
    s.write(b"t")
    out = b""
    t0 = time.time()
    while time.time() - t0 < 180:
        out += s.read(4096)
        if b"PASS (vs PC cuantizado)" in out:
            break
    s.close()

    text = out.decode("utf-8", "replace")
    rows = [m for line in text.splitlines()
            if (m := _SELFTEST_LINE.match(line))]
    if not rows:
        sys.exit("no pude parsear el self-test (¿FEATURE_HW_VALIDATION=0 o "
                 "sin test_images.h?)\n--- output ---\n" + text[-1500:])

    times = [int(m.group(5)) for m in rows]
    passes = sum(1 for m in rows if m.group(6) == "PASS")
    dlogits = max(int(m.group(4)) for m in rows)
    for m in rows:
        print(f"  {m.group(1):<9} -> {m.group(2):<9} {m.group(5):>5} ms  "
              f"dlogits<={m.group(4)}  {m.group(6)}")

    return {
        "inf_ms_mean": round(statistics.mean(times), 1),
        "inf_ms_min": min(times),
        "inf_ms_max": max(times),
        "pass_count": passes,
        "total": len(rows),
        "dlogits_max": dlogits,
        "measured_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    args = parse_args()

    espdl = ROOT / "models" / f"{args.source}.espdl"
    if not args.skip_build:
        build(args)
    if not args.skip_flash:
        flash(args)

    entry = measure(args)
    if espdl.exists():
        entry["model_bytes"] = espdl.stat().st_size

    from model_registry import load_benchmarks, save_benchmarks, upsert_benchmark
    path = ROOT / "benchmarks.json"
    data = load_benchmarks(path)
    upsert_benchmark(data, args.source, args.board, entry, runtime=args.runtime)
    save_benchmarks(path, data)

    ok = entry["pass_count"] == entry["total"] and entry["dlogits_max"] == 0
    print(f"\n[BENCH] {args.source} @ {args.board} [{args.runtime}]: "
          f"media {entry['inf_ms_mean']} ms "
          f"(min {entry['inf_ms_min']} / max {entry['inf_ms_max']}), "
          f"{entry['pass_count']}/{entry['total']} PASS, "
          f"dlogits<={entry['dlogits_max']} "
          f"{'OK' if ok else '*** REVISAR: no es paridad exacta ***'}")
    print(f"[BENCH] registrado en {path.name} — ver `uv run main.py models`")


if __name__ == "__main__":
    main()
