"""Runner declarativo de flows definidos en pipelines.yaml."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import yaml


VALID_STAGES = {"train", "prune", "quantize", "build"}
STAGES_REQUIRING_SOURCE = {"prune", "quantize"}
VALID_QUANTIZE_VARIANTS = {"a", "b", "c"}


class PipelineError(Exception):
    """Error de carga o validación de pipelines.yaml."""


def load_pipelines(config_path: Path) -> dict[str, list[dict]]:
    """Carga y valida pipelines.yaml. Devuelve el mapa nombre→lista_de_stages.

    Si el YAML tiene un `port` a nivel top-level, lo inyecta como default en
    `params.port` de cada stage `build` que no lo defina. Esto evita repetir
    el puerto en cada build stage y centraliza la config del hardware."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise PipelineError(f"pipelines.yaml not found at {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "flows" not in raw:
        raise PipelineError("pipelines.yaml missing 'flows' top-level key")

    flows = raw["flows"]
    if not isinstance(flows, dict):
        raise PipelineError("'flows' must be a mapping")

    default_port = raw.get("port")
    for flow_name, stages in flows.items():
        if not isinstance(stages, list) or not stages:
            raise PipelineError(f"flow '{flow_name}' must be a non-empty list")
        for i, st in enumerate(stages):
            _validate_stage(flow_name, i, st)
            if default_port and st.get("stage") == "build":
                params = st.setdefault("params", {})
                params.setdefault("port", default_port)

    return flows


def _validate_stage(flow_name: str, idx: int, st: dict) -> None:
    prefix = f"flow '{flow_name}' stage #{idx}"
    if not isinstance(st, dict) or "stage" not in st:
        raise PipelineError(f"{prefix}: missing 'stage' key")
    name = st["stage"]
    if name not in VALID_STAGES:
        raise PipelineError(f"{prefix}: unknown stage '{name}' (valid: {sorted(VALID_STAGES)})")
    if name in STAGES_REQUIRING_SOURCE and not st.get("source"):
        raise PipelineError(f"{prefix}: 'source' is required for stage '{name}'")
    if name == "quantize":
        variant = st.get("variant", "a")
        if variant not in VALID_QUANTIZE_VARIANTS:
            raise PipelineError(
                f"{prefix}: variant must be one of {sorted(VALID_QUANTIZE_VARIANTS)}, got '{variant}'"
            )


_SCRIPT_BY_STAGE = {
    "train": "script01-train.py",
    "prune": "script02-prune.py",
    "build": "script04-build.py",
}

_SCRIPT_BY_QUANTIZE_VARIANT = {
    "a": "script03a-quantize.py",
    "b": "script03b-quantize-torchao.py",
    "c": "script03c-quantize-brevitas.py",
}


def build_command(stage: dict) -> list[str]:
    """Traduce un stage YAML a una línea de comando argv."""
    name = stage["stage"]
    if name == "quantize":
        script = _SCRIPT_BY_QUANTIZE_VARIANT[stage.get("variant", "a")]
    else:
        script = _SCRIPT_BY_STAGE[name]

    cmd = ["uv", "run", "python", script]

    if stage.get("source"):
        cmd.extend(["--source", str(stage["source"])])

    for key, value in (stage.get("params") or {}).items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    return cmd


def run_flow(flow_name: str, flows: dict, dry_run: bool = False) -> int:
    """Ejecuta un flow. Retorna el exit code (0 = ok)."""
    if flow_name not in flows:
        raise PipelineError(
            f"flow '{flow_name}' not found. Available: {sorted(flows.keys())}"
        )

    stages = flows[flow_name]
    for idx, stage in enumerate(stages, start=1):
        cmd = build_command(stage)
        tag = f"[{stage['stage']}]"
        header = f"{tag} stage {idx}/{len(stages)}: {' '.join(shlex.quote(c) for c in cmd)}"
        if dry_run:
            print(f"[dry-run] {header}")
            continue

        print(header, flush=True)
        rc = _run_subprocess(cmd, tag)
        if rc != 0:
            print(f"{tag} FAILED with exit code {rc}", file=sys.stderr)
            return rc

    return 0


def _run_subprocess(cmd: list[str], tag: str) -> int:
    """Corre un subprocess y prefija cada línea de stdout/stderr con `tag`.

    Devuelve el exit code.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"{tag} {line.rstrip()}", flush=True)
    return proc.wait()
