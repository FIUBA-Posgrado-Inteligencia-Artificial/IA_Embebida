"""Entry point para el pipeline ESP32-CAM. Ver `main.py --help` (la lista de
comandos vive en _DESCRIPTION, más abajo)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Raíz del ejemplo: todos los paths por defecto (pipelines.yaml, models/,
# models.json, firmware/) se resuelven contra ella, no contra el cwd, para
# que main.py funcione invocado desde cualquier directorio.
ROOT = Path(__file__).resolve().parent

# Las utilidades importables viven en pipelib/
sys.path.insert(0, str(ROOT / "pipelib"))


_DESCRIPTION = """\
Entry point para el pipeline ESP32-CAM.

Comandos:
  run <flow>     ejecuta un flow definido en pipelines.yaml
  flows          lista los flows disponibles
  models         imprime la tabla comparativa de modelos
  models clean   reconcilia models.json con el contenido de models/
  models reset   borra todos los artefactos de modelo (requiere --yes)
  bench          detalle de benchmarks on-device por modelo
  menuconfig     abre idf.py menuconfig sobre el proyecto firmware/

Usá `main.py <comando> --help` para ver las opciones de cada comando.

Ejemplos:
  uv run python main.py run fast_debug_esp32s3cam --dry-run
  uv run python main.py models --sort size --reverse
  uv run python main.py models clean --dry-run
"""

_MODELS_DESCRIPTION = """\
Sin subcomando: imprime la tabla comparativa de modelos (accuracy, params,
MACs, tamaño, y tiempos de inferencia medidos en placa si hay benchmarks).

Subcomandos:
  clean   reconcilia models.json con lo que hay en models/ (agrega entradas
          nuevas, elimina huérfanas, actualiza tamaños/fechas)
  reset   borra checkpoints, .espdl, sidecars, headers generados, models.json
          y el model_data.h instalado en firmware/ (requiere --yes)

Ejemplos:
  uv run python main.py models --sort acc --reverse
  uv run python main.py models clean --dry-run
  uv run python main.py models reset --dry-run
"""


_BENCH_DESCRIPTION = """\
Muestra el detalle de los benchmarks on-device (benchmarks.json): una fila por
(modelo, placa, runtime) con tiempos de inferencia, self-test, tamaño del
binario y MACs/s efectivos. Las columnas de métricas aún no medidas (RAM,
PSRAM, ¿entra?) muestran '-' hasta que script06-bench las registre.

Sin argumentos muestra todos los modelos con mediciones; con nombres, solo
esos (el nombre es el del modelo cuantizado, sin extensión).

Ejemplos:
  uv run python main.py bench
  uv run python main.py bench standard_rgb96_pruned_activation_20_ft10_quantized_ptq
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="<comando>")

    run_p = sub.add_parser(
        "run", help="Ejecuta un flow de pipelines.yaml",
        description="Ejecuta un flow (secuencia de stages) definido en pipelines.yaml.",
    )
    run_p.add_argument("flow_name",
                       help="nombre del flow (ver `main.py flows`)")
    run_p.add_argument("--dry-run", action="store_true",
                       help="sólo imprime los comandos que ejecutaría, sin correrlos")
    run_p.add_argument("--config", type=Path, default=ROOT / "pipelines.yaml",
                       help="archivo YAML de flows (default: pipelines.yaml del ejemplo)")

    flows_p = sub.add_parser(
        "flows", help="Lista los flows disponibles",
        description="Lista los flows definidos en el YAML, con sus stages.",
    )
    flows_p.add_argument("--config", type=Path, default=ROOT / "pipelines.yaml",
                         help="archivo YAML de flows (default: pipelines.yaml del ejemplo)")

    models_p = sub.add_parser(
        "models", help="Tabla de modelos, o clean/reset del registro",
        description=_MODELS_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    models_p.add_argument("subcommand", nargs="?", choices=["clean", "reset"],
                          default=None, metavar="{clean,reset}",
                          help="acción sobre el registro; sin subcomando imprime la tabla")
    models_p.add_argument("--dry-run", action="store_true",
                          help="(clean|reset) no modifica nada; solo muestra qué haría")
    models_p.add_argument("--yes", action="store_true",
                          help="(sólo con 'reset') confirma el borrado")
    models_p.add_argument("--sort", choices=["name", "acc", "params", "macs", "size"],
                          default="name",
                          help="(tabla) columna por la que ordenar (default: %(default)s)")
    models_p.add_argument("--reverse", action="store_true",
                          help="(tabla) invierte el orden")
    models_p.add_argument("--dir", type=Path, default=ROOT, dest="base_dir",
                          help="raíz sobre la que operar (default: el directorio del ejemplo)")

    bench_p = sub.add_parser(
        "bench", help="Detalle de benchmarks on-device por modelo",
        description=_BENCH_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bench_p.add_argument("models", nargs="*", metavar="modelo",
                         help="modelos a mostrar (default: todos los que tengan mediciones)")
    bench_p.add_argument("--sort", choices=["board", "runtime", "time", "mmacs"],
                         default="board",
                         help="columna por la que ordenar las filas de cada modelo "
                              "(default: %(default)s)")
    bench_p.add_argument("--reverse", action="store_true",
                         help="invierte el orden")
    bench_p.add_argument("--dir", type=Path, default=ROOT, dest="base_dir",
                         help="raíz sobre la que operar (default: el directorio del ejemplo)")

    sub.add_parser(
        "menuconfig", help="Abre idf.py menuconfig sobre firmware/",
        description="Abre la UI ncurses de idf.py menuconfig sobre el proyecto firmware/.",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "flows":
        return _cmd_flows(args)
    if args.command == "models":
        _validate_models_args(parser, args)
        return _cmd_models(args)
    if args.command == "bench":
        return _cmd_bench(args)
    if args.command == "menuconfig":
        return _cmd_menuconfig()
    parser.error(f"unknown command: {args.command}")
    return 2


def _validate_models_args(parser: argparse.ArgumentParser,
                          args: argparse.Namespace) -> None:
    """Rechaza combinaciones de flags que no aplican al modo elegido, en vez
    de ignorarlas en silencio."""
    if args.subcommand in ("clean", "reset"):
        if args.sort != "name" or args.reverse:
            parser.error(f"--sort/--reverse solo aplican a la tabla, no a '{args.subcommand}'")
        if args.yes and args.subcommand != "reset":
            parser.error("--yes solo aplica a 'reset'")
    else:
        if args.dry_run:
            parser.error("--dry-run solo aplica a 'clean' o 'reset'")
        if args.yes:
            parser.error("--yes solo aplica a 'reset'")


def _cmd_run(args: argparse.Namespace) -> int:
    from pipeline_runner import load_pipelines, run_flow, PipelineError
    try:
        flows = load_pipelines(args.config)
        rc = run_flow(args.flow_name, flows, dry_run=args.dry_run, cwd=ROOT)
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if rc == 0 and not args.dry_run:
        _refresh_registry()
    return rc


def _refresh_registry(cwd=None) -> None:
    from pathlib import Path
    from model_registry import (
        load_registry, save_registry, scan_models_dir, reconcile,
    )
    base = Path(cwd) if cwd is not None else ROOT
    registry = load_registry(base / "models.json")
    disk = scan_models_dir(base / "models")
    updated, _ = reconcile(registry, disk)
    save_registry(base / "models.json", updated)


def _cmd_flows(args: argparse.Namespace) -> int:
    from pipeline_runner import load_pipelines, PipelineError
    try:
        flows = load_pipelines(args.config)
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if not flows:
        print("(no flows defined)")
        return 0

    name_w = max(len("name"), *(len(n) for n in flows))
    print(f"{'name'.ljust(name_w)}  stages  first → last")
    print(f"{'-' * name_w}  ------  ------------")
    for name, stages in flows.items():
        first = stages[0]["stage"]
        last = stages[-1]["stage"]
        print(f"{name.ljust(name_w)}  {len(stages)} stages  {first} → {last}")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    from pathlib import Path
    from model_registry import (
        load_registry, save_registry, scan_models_dir, reconcile,
    )

    base = args.base_dir
    models_dir = base / "models"
    registry_path = base / "models.json"

    if args.subcommand == "clean":
        registry = load_registry(registry_path)
        disk = scan_models_dir(models_dir)
        updated, changes = reconcile(registry, disk)

        if args.dry_run:
            prefix = "would "
        else:
            prefix = ""
            save_registry(registry_path, updated)

        for name in changes["added"]:
            print(f"{prefix}add: {name}")
        for name in changes["removed"]:
            print(f"{prefix}remove: {name}")
        for name in changes["updated"]:
            print(f"{prefix}update: {name}")

        if changes["removed"]:
            print(f"({len(changes['removed'])} removed from registry)")
        if not any(changes.values()):
            print("registry in sync; nothing to do")
        return 0

    if args.subcommand == "reset":
        return _cmd_models_reset(base, dry_run=args.dry_run, confirmed=args.yes)

    # Default: print the comparison table (same format used by stage scripts).
    from esp32cam_utils import collect_available_results, print_comparison_table
    from model_registry import load_benchmarks, benchmarks_by_key
    bench = benchmarks_by_key(load_benchmarks(base / "benchmarks.json"))
    results = collect_available_results(models_dir)
    results.sort(key=_MODELS_SORT_KEYS[args.sort], reverse=args.reverse)
    print_comparison_table(results, benchmarks=bench or None)
    return 0


_MODELS_SORT_KEYS = {
    "name": lambda r: r["name"],
    "acc": lambda r: r.get("accuracy", 0),
    "params": lambda r: r.get("params", 0),
    "macs": lambda r: r.get("macs", 0),
    "size": lambda r: r.get("size_kb", 0),
}


# Columnas de la tabla de bench: (encabezado, ancho mínimo, render(row)).
# `row` junta la entrada de benchmarks.json con metadata del modelo (macs).
# Para agregar una métrica nueva: registrarla acá y emitirla en la entry de
# script06-bench; las entries viejas sin la key muestran '-'.
def _fmt_opt(entry, key, fmt="{}"):
    v = entry.get(key)
    return fmt.format(v) if v is not None else "-"


_BENCH_COLUMNS = [
    ("placa",       10, lambda r: r["board"]),
    ("runtime",     10, lambda r: r.get("runtime", "espdl")),
    ("t_inf [ms]",  10, lambda r: _fmt_opt(r, "inf_ms_mean")),
    ("min/max",     13, lambda r: (f"{r['inf_ms_min']}/{r['inf_ms_max']}"
                                   if "inf_ms_min" in r else "-")),
    ("self-test",   10, lambda r: (f"{r['pass_count']}/{r['total']}"
                                   if "pass_count" in r else "-")),
    ("dlogits",      8, lambda r: _fmt_opt(r, "dlogits_max")),
    ("modelo [KB]", 11, lambda r: (f"{r['model_bytes'] / 1024:.1f}"
                                   if r.get("model_bytes") else "-")),
    ("MMACs/s",      9, lambda r: (f"{r['_macs'] / r['inf_ms_mean'] / 1000:,.1f}"
                                   if r.get("_macs") and r.get("inf_ms_mean") else "-")),
    # Métricas del backlog, aún no emitidas por script06-bench:
    ("RAM [KB]",     9, lambda r: _fmt_opt(r, "ram_kb")),
    ("PSRAM",        6, lambda r: _fmt_opt(r, "psram_required")),
    ("¿entra?",      8, lambda r: _fmt_opt(r, "fits_flash")),
    ("medido",      20, lambda r: _fmt_opt(r, "measured_at")),
]


# Orden de las filas de cada modelo en `bench`. "mmacs" ordena por MACs/s
# (filas sin tiempo o sin MACs van al final).
_BENCH_SORT_KEYS = {
    "board": lambda b: (b["board"], b.get("runtime", "espdl")),
    "runtime": lambda b: (b.get("runtime", "espdl"), b["board"]),
    "time": lambda b: (b.get("inf_ms_mean") is None, b.get("inf_ms_mean", 0)),
    "mmacs": lambda b: ((b.get("_macs") and b.get("inf_ms_mean")) is None,
                        (b["_macs"] / b["inf_ms_mean"])
                        if b.get("_macs") and b.get("inf_ms_mean") else 0),
}


def _cmd_bench(args: argparse.Namespace) -> int:
    from esp32cam_utils import collect_available_results
    from model_registry import load_benchmarks

    base = args.base_dir
    rows = load_benchmarks(base / "benchmarks.json").get("benchmarks", [])
    if args.models:
        unknown = set(args.models) - {b["model"] for b in rows}
        rows = [b for b in rows if b["model"] in args.models]
        for name in sorted(unknown):
            print(f"[WARN] sin mediciones para: {name}", file=sys.stderr)
    if not rows:
        print("(sin benchmarks registrados; correr scripts/script06-bench.py "
              "contra una placa para medir)")
        return 0

    # MACs del modelo (para MACs/s): metadata de los .pth/.espdl en models/.
    macs_by_name = {r["name"]: r.get("macs")
                    for r in collect_available_results(base / "models")}
    for b in rows:
        b["_macs"] = macs_by_name.get(b["model"])

    for model in sorted({b["model"] for b in rows}):
        print(f"\n{model}")
        header = "  " + "  ".join(h.ljust(w) for h, w, _ in _BENCH_COLUMNS)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for b in sorted((b for b in rows if b["model"] == model),
                        key=_BENCH_SORT_KEYS[args.sort], reverse=args.reverse):
            print("  " + "  ".join(str(render(b)).ljust(w)
                                   for _, w, render in _BENCH_COLUMNS))
    return 0


def _cmd_menuconfig() -> int:
    """Lanza script05-build.py --menuconfig-only como subproceso, heredando
    stdin/stdout/stderr para que la UI ncurses de menuconfig funcione."""
    import subprocess
    script = ROOT / "scripts" / "script05-build.py"
    return subprocess.call(
        ["uv", "run", "python", str(script), "--menuconfig-only"], cwd=ROOT
    )


_RESET_MODEL_EXTS = {".pth", ".espdl", ".info", ".json", ".h"}


def _cmd_models_reset(cwd, dry_run: bool, confirmed: bool) -> int:
    """Borra checkpoints, .espdl, sidecars de esp-ppq, headers generados,
    models.json y el model_data.h instalado en el firmware."""
    from pathlib import Path
    cwd = Path(cwd)
    models_dir = cwd / "models"
    registry_path = cwd / "models.json"
    firmware_main = cwd / "firmware" / "main"

    to_delete: list = []
    if models_dir.exists():
        for p in sorted(models_dir.iterdir()):
            if p.is_file() and p.suffix in _RESET_MODEL_EXTS:
                to_delete.append(p)
    if registry_path.exists():
        to_delete.append(registry_path)
    for name in ("model_data.h", "model_data.meta.json"):
        p = firmware_main / name
        if p.exists():
            to_delete.append(p)

    if not to_delete:
        print("(nothing to delete)")
        return 0

    prefix = "would delete" if dry_run else "delete"
    for p in to_delete:
        print(f"{prefix}: {p.relative_to(cwd)}")

    if dry_run:
        return 0

    if not confirmed:
        print(f"\nerror: about to delete {len(to_delete)} files. "
              "Pass --yes to confirm.", file=sys.stderr)
        return 1

    for p in to_delete:
        p.unlink()
    print(f"\ndeleted {len(to_delete)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
