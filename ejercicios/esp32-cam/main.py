"""Entry point para el pipeline ESP32-CAM.

Subcomandos:
  run <flow>          — ejecuta un flow definido en pipelines.yaml
  flows               — lista flows disponibles
  models              — imprime la tabla de modelos
  models clean        — reconcilia models.json con el contenido de models/
  models reset        — borra todos los artefactos de modelo (requiere --yes)
  menuconfig          — abre idf.py menuconfig sobre el proyecto firmware/
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Ejecuta un flow")
    run_p.add_argument("flow_name")
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--config", default="pipelines.yaml")

    flows_p = sub.add_parser("flows", help="Lista flows disponibles")
    flows_p.add_argument("--config", default="pipelines.yaml")

    models_p = sub.add_parser("models", help="Imprime tabla de modelos o reconcilia")
    models_p.add_argument("subcommand", nargs="?", choices=["clean", "reset"], default=None)
    models_p.add_argument("--dry-run", action="store_true",
                          help="(clean|reset) no modifica nada; solo muestra qué haría")
    models_p.add_argument("--yes", action="store_true",
                          help="(sólo con 'reset') confirma el borrado")

    sub.add_parser("menuconfig", help="Abre idf.py menuconfig sobre firmware/")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "flows":
        return _cmd_flows(args)
    if args.command == "models":
        return _cmd_models(args)
    if args.command == "menuconfig":
        return _cmd_menuconfig()
    parser.error(f"unknown command: {args.command}")
    return 2


def _cmd_run(args: argparse.Namespace) -> int:
    from pipeline_runner import load_pipelines, run_flow, PipelineError
    try:
        flows = load_pipelines(args.config)
        rc = run_flow(args.flow_name, flows, dry_run=args.dry_run)
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
    base = Path(cwd) if cwd is not None else Path.cwd()
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

    cwd = Path.cwd()
    models_dir = cwd / "models"
    registry_path = cwd / "models.json"

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
        return _cmd_models_reset(cwd, dry_run=args.dry_run, confirmed=args.yes)

    # Default: print the comparison table (same format used by stage scripts).
    from esp32cam_utils import collect_available_results, print_comparison_table
    print_comparison_table(collect_available_results(models_dir))
    return 0


def _cmd_menuconfig() -> int:
    """Lanza script04-build.py --menuconfig-only como subproceso, heredando
    stdin/stdout/stderr para que la UI ncurses de menuconfig funcione."""
    import subprocess
    from pathlib import Path
    script = Path(__file__).resolve().parent / "script04-build.py"
    return subprocess.call(
        ["uv", "run", "python", str(script), "--menuconfig-only"]
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
