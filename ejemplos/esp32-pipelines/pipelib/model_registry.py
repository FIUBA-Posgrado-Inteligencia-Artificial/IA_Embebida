"""Registro de modelos entrenados/cuantizados en examples/esp32-cam/models/."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path


_TRACKED_EXTENSIONS = (".pth", ".espdl")


def infer_stage_from_name(name: str) -> str:
    """Clasifica un modelo por su nombre de archivo (sin path).

    Reglas evaluadas en orden de precedencia (nombres cortos: _praNN =
    pruned activation NN%, _qptq = quantized ptq, etc.):
    - termina en '.espdl' → 'espdl'
    - contiene '_q<modo>' (_qptq/_qqat/_qtao/_qbrv) o el legacy '_quantized'
      → 'quantized'
    - el stem (sin .pth) termina en '_ft<N>' → 'finetuned'
    - contiene '_pr[awn]<NN>' o el legacy '_pruned_' → 'pruned'
    - sino → 'baseline'
    """
    if name.endswith(".espdl"):
        return "espdl"
    if re.search(r"_q(ptq|qat|tao|brv)", name) or "_quantized" in name:
        return "quantized"
    stem = name[:-4] if name.endswith(".pth") else name
    if re.search(r"_ft\d+$", stem):
        return "finetuned"
    if re.search(r"_pr[awn]\d+", name) or "_pruned_" in name:
        return "pruned"
    return "baseline"


def scan_models_dir(models_dir: Path) -> list[dict]:
    """Escanea `models_dir` y devuelve una lista de entradas con metadata.

    Sólo considera archivos con extensiones en _TRACKED_EXTENSIONS.
    Para `.pth` el 'name' es el stem (sin extensión). Para `.espdl` conserva
    la extensión en el nombre para distinguirlo del checkpoint PyTorch homónimo.
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []

    entries = []
    for path in sorted(models_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix not in _TRACKED_EXTENSIONS:
            continue
        name = path.stem if path.suffix == ".pth" else path.name
        stat = path.stat()
        entries.append({
            "name": name,
            "path": str(path),
            "stage": infer_stage_from_name(path.name),
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "notes": "",
        })
    return entries


EMPTY_REGISTRY = {"models": [], "updated_at": ""}


def load_registry(registry_path: Path) -> dict:
    """Lee un registro JSON. Devuelve EMPTY_REGISTRY si no existe."""
    registry_path = Path(registry_path)
    if not registry_path.exists():
        return {"models": [], "updated_at": ""}
    return json.loads(registry_path.read_text(encoding="utf-8"))


def save_registry(registry_path: Path, data: dict) -> None:
    """Escribe el registro JSON atómicamente (tmp + rename)."""
    registry_path = Path(registry_path)
    tmp_path = registry_path.with_suffix(registry_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, registry_path)


def reconcile(registry: dict, disk_entries: list[dict]) -> tuple[dict, dict]:
    """Reconcilia `registry` con lo que hay en disco.

    - Elimina entradas cuyo nombre no aparece en disk_entries.
    - Agrega entradas nuevas (presentes en disco, ausentes en registry).
    - Actualiza size_bytes y modified en entradas existentes; preserva notes.

    Devuelve (registry_actualizado, changes) donde changes tiene las llaves
    'added', 'removed', 'updated' con listas de nombres.
    """
    existing_by_name = {m["name"]: m for m in registry.get("models", [])}
    disk_by_name = {e["name"]: e for e in disk_entries}

    added, removed, updated_names = [], [], []

    new_models = []
    for name, disk_entry in disk_by_name.items():
        if name in existing_by_name:
            prev = existing_by_name[name]
            merged = {**disk_entry, "notes": prev.get("notes", "")}
            new_models.append(merged)
            if (prev.get("size_bytes") != disk_entry.get("size_bytes")
                    or prev.get("modified") != disk_entry.get("modified")):
                updated_names.append(name)
        else:
            new_models.append(disk_entry)
            added.append(name)

    for name in existing_by_name:
        if name not in disk_by_name:
            removed.append(name)

    new_models.sort(key=lambda m: m["name"])
    return (
        {"models": new_models, "updated_at": datetime.now().isoformat(timespec="seconds")},
        {"added": added, "removed": removed, "updated": updated_names},
    )



# ---------------------------------------------------------------------------
# Benchmarks en placa (benchmarks.json)
# ---------------------------------------------------------------------------

EMPTY_BENCHMARKS = {"benchmarks": [], "updated_at": ""}


def load_benchmarks(path: Path) -> dict:
    """Lee benchmarks.json. Devuelve estructura vacía si no existe."""
    path = Path(path)
    if not path.exists():
        return {"benchmarks": [], "updated_at": ""}
    return json.loads(path.read_text(encoding="utf-8"))


def save_benchmarks(path: Path, data: dict) -> None:
    """Escribe benchmarks.json atómicamente (tmp + rename)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def upsert_benchmark(data: dict, model: str, board: str, entry: dict,
                     runtime: str = "espdl") -> dict:
    """Inserta o reemplaza la medición de (model, board, runtime).

    Filas viejas sin campo `runtime` se tratan como runtime="espdl".
    Devuelve `data`."""
    rows = [b for b in data.get("benchmarks", [])
            if not (b.get("model") == model and b.get("board") == board
                    and b.get("runtime", "espdl") == runtime)]
    rows.append({"model": model, "board": board, "runtime": runtime, **entry})
    rows.sort(key=lambda b: (b["model"], b["board"], b.get("runtime", "espdl")))
    data["benchmarks"] = rows
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return data


def benchmarks_by_key(data: dict) -> dict:
    """Indexa {(model, board, runtime): entry} para el render de la tabla.

    Filas sin campo `runtime` (benchmarks.json viejo) se indexan como
    runtime="espdl"."""
    return {(b["model"], b["board"], b.get("runtime", "espdl")): b
            for b in data.get("benchmarks", [])}
