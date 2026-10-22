# Pipeline Stages + Inferencia Interactiva — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separar el pipeline en stages atómicos (`train, prune, finetune, quantize, build, flash, monitor`) y agregar al firmware inferencia interactiva (trigger serie/web, modo single/continuo, webserver con última captura + resultado + tiempos + FPS).

**Architecture:** Parte A (Python): renumerar scripts, extraer finetune de prune a `script03-finetune.py`, `script05-build.py` gana `--action build|flash|monitor`, runner y YAML actualizados. Parte B (firmware `firmware/`): nuevo módulo `capture_state` (estado compartido con mutex), loop de captura en `app_main.c` con modos, `m` por serie, endpoints HTTP nuevos.

**Tech Stack:** Python 3 + uv + pytest; ESP-IDF 5.3 (C), esp32-camera (`fmt2jpg`), esp_http_server, cJSON.

**Spec:** `docs/superpowers/specs/2026-08-07-pipeline-stages-inference-webserver-design.md`

## Global Constraints

- Cwd para todo: `examples/esp32-cam/`. Tests: `uv run pytest tests/ -v`.
- Commits SIEMPRE por paths explícitos (`git commit -- <paths>`): el index puede tener cambios staged de otra sesión (carpeta `examples/esp32-s3-cam/` — NO tocarla, NO commitearla).
- Sin validación de dependencias entre stages en el runner.
- No editar `firmware/main/model_data.h` (generado).
- El firmware debe compilar con `idf.py build` (target esp32); no hace falta flashear para cerrar tareas.
- `firmware_test_webserver/` no se toca.

---

### Task 1: Renumerar scripts y actualizar referencias

**Files:**
- Rename: `script03a-quantize.py` → `script04a-quantize.py`, `script03b-quantize-torchao.py` → `script04b-quantize-torchao.py`, `script03c-quantize-brevitas.py` → `script04c-quantize-brevitas.py`, `script04-build.py` → `script05-build.py`
- Modify: `main.py:154` (path de menuconfig), `tests/test_script04_build.py` (SCRIPT_PATH y renombrar archivo a `tests/test_script05_build.py`), `tests/test_meta_emit.py` y `tests/test_quantize.py` (si referencian `script03*` por path — grep primero)
- Modify: `pipeline_runner.py` (`_SCRIPT_BY_STAGE`, `_SCRIPT_BY_QUANTIZE_VARIANT`)

**Interfaces:**
- Produces: nombres de script nuevos que el resto del plan asume: `script04a-quantize.py`, `script05-build.py`, etc.

- [ ] **Step 1: Renombrar con git mv**

```bash
git mv script03a-quantize.py script04a-quantize.py
git mv script03b-quantize-torchao.py script04b-quantize-torchao.py
git mv script03c-quantize-brevitas.py script04c-quantize-brevitas.py
git mv script04-build.py script05-build.py
git mv tests/test_script04_build.py tests/test_script05_build.py
```

- [ ] **Step 2: Actualizar todas las referencias por texto**

```bash
grep -rln "script03a-quantize\|script03b-quantize\|script03c-quantize\|script04-build" \
  --include="*.py" --include="*.yaml" . | grep -v esp32-s3-cam
```

En cada archivo encontrado (esperados: `pipeline_runner.py`, `main.py`, `tests/test_script05_build.py`, `tests/test_pipeline_runner.py`, `tests/test_main_cli.py`, posiblemente `tests/test_meta_emit.py`, `script04a/b/c` docstrings, `pipelines.yaml`): reemplazar `script03a-quantize.py`→`script04a-quantize.py`, `script03b-quantize-torchao.py`→`script04b-quantize-torchao.py`, `script03c-quantize-brevitas.py`→`script04c-quantize-brevitas.py`, `script04-build.py`→`script05-build.py`. También `prog="script04-build.py"` dentro de `script05-build.py`.

- [ ] **Step 3: Correr tests**

Run: `uv run pytest tests/ -v`
Expected: PASS (mismos tests, paths nuevos).

- [ ] **Step 4: Commit (paths explícitos)**

```bash
git commit -m "refactor(esp32-cam): renumerar scripts (quantize→04, build→05)" \
  -- script04a-quantize.py script04b-quantize-torchao.py script04c-quantize-brevitas.py \
     script05-build.py tests/ main.py pipeline_runner.py pipelines.yaml
```

---

### Task 2: Runner — stages nuevos (`finetune`, `flash`, `monitor`) y `--action`

**Files:**
- Modify: `pipeline_runner.py`
- Test: `tests/test_pipeline_runner.py`

**Interfaces:**
- Produces: `VALID_STAGES = {"train","prune","finetune","quantize","build","flash","monitor"}`; `STAGES_REQUIRING_SOURCE = {"prune","finetune","quantize"}`; `build_command` genera `uv run python script03-finetune.py --source X ...` para finetune y `uv run python script05-build.py --action <stage> ...` para build/flash/monitor; `load_pipelines` inyecta `port` default en stages `flash` y `monitor` (no en `build`).

- [ ] **Step 1: Escribir tests que fallan**

Agregar a `tests/test_pipeline_runner.py` (seguir el estilo existente del archivo):

```python
def test_finetune_stage_command():
    cmd = build_command({"stage": "finetune", "source": "m1", "params": {"epochs": 5, "lr": 2e-4}})
    assert cmd[:4] == ["uv", "run", "python", "script03-finetune.py"]
    assert "--source" in cmd and "m1" in cmd

def test_finetune_requires_source(tmp_path):
    yaml_text = "flows:\n  f:\n    - stage: finetune\n      params: {epochs: 1}\n"
    p = tmp_path / "pipelines.yaml"; p.write_text(yaml_text)
    with pytest.raises(PipelineError):
        load_pipelines(p)

def test_build_flash_monitor_commands():
    assert build_command({"stage": "build"})[3:] == ["script05-build.py", "--action", "build"]
    assert build_command({"stage": "flash"})[3:5] == ["script05-build.py", "--action"]
    assert "flash" in build_command({"stage": "flash"})
    assert "monitor" in build_command({"stage": "monitor"})

def test_port_injected_in_flash_and_monitor_not_build(tmp_path):
    yaml_text = (
        "port: /dev/ttyUSB9\n"
        "flows:\n  f:\n"
        "    - stage: build\n"
        "    - stage: flash\n"
        "    - stage: monitor\n"
    )
    p = tmp_path / "pipelines.yaml"; p.write_text(yaml_text)
    flows = load_pipelines(p)
    build_st, flash_st, mon_st = flows["f"]
    assert "port" not in (build_st.get("params") or {})
    assert flash_st["params"]["port"] == "/dev/ttyUSB9"
    assert mon_st["params"]["port"] == "/dev/ttyUSB9"
```

- [ ] **Step 2: Verificar que fallan**

Run: `uv run pytest tests/test_pipeline_runner.py -v`
Expected: FAIL (stage 'finetune' desconocido, etc.)

- [ ] **Step 3: Implementar en `pipeline_runner.py`**

```python
VALID_STAGES = {"train", "prune", "finetune", "quantize", "build", "flash", "monitor"}
STAGES_REQUIRING_SOURCE = {"prune", "finetune", "quantize"}
_PORT_INJECTED_STAGES = {"flash", "monitor"}

_SCRIPT_BY_STAGE = {
    "train": "script01-train.py",
    "prune": "script02-prune.py",
    "finetune": "script03-finetune.py",
    "build": "script05-build.py",
    "flash": "script05-build.py",
    "monitor": "script05-build.py",
}

_SCRIPT_BY_QUANTIZE_VARIANT = {
    "a": "script04a-quantize.py",
    "b": "script04b-quantize-torchao.py",
    "c": "script04c-quantize-brevitas.py",
}
```

En `load_pipelines`, cambiar la inyección de puerto:

```python
if default_port and st.get("stage") in _PORT_INJECTED_STAGES:
    params = st.setdefault("params", {})
    params.setdefault("port", default_port)
```

En `build_command`, después de elegir el script:

```python
cmd = ["uv", "run", "python", script]
if name in ("build", "flash", "monitor"):
    cmd.extend(["--action", name])
```

- [ ] **Step 4: Correr tests**

Run: `uv run pytest tests/test_pipeline_runner.py -v` — Expected: PASS. Los tests viejos que asuman port inyectado en `build` deben actualizarse en este paso.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(esp32-cam): stages finetune/flash/monitor en el runner" \
  -- pipeline_runner.py tests/test_pipeline_runner.py
```

---

### Task 3: `script02-prune.py` sin finetune

**Files:**
- Modify: `script02-prune.py`
- Test: no hay tests unitarios de este script (requiere torch/dataset); verificación por `--help` y revisión.

**Interfaces:**
- Produces: `script02-prune.py` sin flags `--finetune-epochs` ni `--lr`; output sigue siendo `<source>_pruned_<method>_<ratio_pct>.pth`.

- [ ] **Step 1: Quitar el finetune**

En `parse_args()`: eliminar `--finetune-epochs` y `--lr`. En `main()`: eliminar el bloque

```python
if args.finetune_epochs > 0 and args.method != "none":
    print(f"  Fine-tuning {args.finetune_epochs} epochs...")
    train(model, train_loader, test_loader, args.finetune_epochs, args.lr, device)
```

y quitar `train` del import. Renombrar los prints "Antes/Después del fine-tuning" a "acc post-pruning". Actualizar docstring/epilog (sin `--finetune-epochs`; nota: "el fine-tuning es el stage `finetune` (script03-finetune.py)").

- [ ] **Step 2: Smoke check**

Run: `uv run python script02-prune.py --help`
Expected: sale sin error, sin mención de finetune.

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor(esp32-cam): prune ya no finetunea (stage aparte)" -- script02-prune.py
```

---

### Task 4: Nuevo `script03-finetune.py`

**Files:**
- Create: `script03-finetune.py`
- Test: `tests/test_finetune_naming.py` (solo naming, sin torch)

**Interfaces:**
- Consumes: `esp32cam_utils`: `load_checkpoint(path, device)`, `save_checkpoint(path, model, meta)`, `train(model, train_loader, test_loader, epochs, lr, device)`, `evaluate`, `get_dataloaders(input_mode, resolution, batch_size)`, `get_device`, `MODELS_DIR`, `count_macs`, `count_parameters`, `model_size_kb`, `collect_available_results`, `print_comparison_table`.
- Produces: función `output_name_for(source: str, epochs: int) -> str` → `f"{source}_ft{epochs}"`; CLI `--source --epochs --lr --batch-size --show-table`.

- [ ] **Step 1: Test de naming (falla)**

```python
# tests/test_finetune_naming.py
import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "script03-finetune.py"

def _load_module():
    spec = importlib.util.spec_from_file_location("script03_finetune", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_output_name_appends_ft_epochs():
    mod = _load_module()
    assert mod.output_name_for("depthwise_rgb32_pruned_activation_30", 5) == \
        "depthwise_rgb32_pruned_activation_30_ft5"

def test_output_name_is_repeatable():
    mod = _load_module()
    once = mod.output_name_for("m_pruned_activation_30", 5)
    assert mod.output_name_for(once + "_pruned_activation_20", 3) == \
        "m_pruned_activation_30_ft5_pruned_activation_20_ft3"
```

Nota: `_load_module()` importa el script entero, así que el script no debe ejecutar nada pesado a nivel módulo (todo bajo `if __name__ == "__main__":`) y los imports de torch van adentro de `main()` **no** — van a nivel módulo como en script02 (esp32cam_utils ya importa torch y está instalado; el test lo tolera).

- [ ] **Step 2: Verificar que falla**

Run: `uv run pytest tests/test_finetune_naming.py -v`
Expected: FAIL (archivo no existe).

- [ ] **Step 3: Implementar `script03-finetune.py`**

```python
"""
ESP32-CAM — script03-finetune: Fine-tuning de un checkpoint
============================================================
Reentrena N epochs cualquier checkpoint .pth (tipicamente post-pruning).
Repetible: train -> prune -> finetune -> prune -> finetune.

Usage:
    uv run python script03-finetune.py --source depthwise_rgb32_pruned_activation_30 --epochs 10
"""

import argparse

from esp32cam_utils import (
    MODELS_DIR, collect_available_results, count_macs, count_parameters,
    evaluate, get_dataloaders, get_device, load_checkpoint,
    model_size_kb, print_comparison_table, save_checkpoint, train,
)


def output_name_for(source: str, epochs: int) -> str:
    return f"{source}_ft{epochs}"


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tuning de cualquier checkpoint producido por train/prune.")
    p.add_argument("--source", required=True,
                   help="[required] Checkpoint a finetunear (sin .pth)")
    p.add_argument("--epochs", type=int, default=10, help="Epochs (default: 10)")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    p.add_argument("--batch-size", type=int, default=64, help="Mini-batch (default: 64)")
    p.add_argument("--show-table", action="store_true",
                   help="Imprime la tabla comparativa de modelos al finalizar")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    source_path = MODELS_DIR / f"{args.source}.pth"
    if not source_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {source_path}")

    output_name = output_name_for(args.source, args.epochs)
    output_path = MODELS_DIR / f"{output_name}.pth"

    if output_path.exists():
        print(f"[INFO] Checkpoint de finetune encontrado — cargando {output_path.name}")
        return

    model, meta = load_checkpoint(source_path, device)
    train_loader, test_loader = get_dataloaders(
        meta["input_mode"], meta["resolution"], args.batch_size)

    print(f"Fine-tuning {args.source}: {args.epochs} epochs, lr={args.lr}")
    train(model, train_loader, test_loader, args.epochs, args.lr, device)

    in_channels = 1 if meta["input_mode"] == "gray" else 3
    acc = evaluate(model, test_loader, device)
    print(f"  acc={acc:.4f}  MACs={count_macs(model, (1, in_channels, meta['resolution'], meta['resolution'])):,}  "
          f"{model_size_kb(model):.1f} KB")

    new_meta = dict(meta)
    new_meta.update({
        "accuracy": round(acc, 4),
        "source": args.source,
        "finetune_epochs": args.epochs,
        "finetune_lr": args.lr,
    })
    save_checkpoint(output_path, model, new_meta)

    if args.show_table:
        print()
        print_comparison_table(collect_available_results())


if __name__ == "__main__":
    main()
```

Nota: verificar qué claves devuelve `load_checkpoint` como `meta` (mirar `save_checkpoint` en `esp32cam_utils.py`); `new_meta = dict(meta)` debe conservar `arch`, `input_mode`, `resolution`, `num_classes`, `classes`, `channels`, `normalize_mean`, `normalize_std`. Si `meta` incluye claves no serializables, copiar explícitamente la misma lista de claves que usa `script02-prune.py`.

- [ ] **Step 4: Correr tests**

Run: `uv run pytest tests/test_finetune_naming.py -v` — Expected: PASS.
Run: `uv run python script03-finetune.py --help` — Expected: sin error.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(esp32-cam): script03-finetune — stage de fine-tuning explícito" \
  -- script03-finetune.py tests/test_finetune_naming.py
```

---

### Task 5: `model_registry.py` — categoría `finetuned`

**Files:**
- Modify: `model_registry.py:14-29` (`infer_stage_from_name`)
- Test: `tests/test_model_registry.py`

**Interfaces:**
- Produces: `infer_stage_from_name` devuelve `"finetuned"` si el nombre (sin `.pth`) termina en `_ft<N>`. Precedencia: `espdl > quantized > finetuned > pruned > baseline`.

- [ ] **Step 1: Tests que fallan**

```python
def test_infer_stage_finetuned():
    assert infer_stage_from_name("m_pruned_activation_30_ft5.pth") == "finetuned"

def test_finetuned_loses_to_quantized_and_espdl():
    assert infer_stage_from_name("m_ft5_quantized_ptq.pth") == "quantized"
    assert infer_stage_from_name("m_ft5_quantized_ptq.espdl") == "espdl"

def test_pruned_after_ft_is_pruned():
    # el último paso fue prune: _ft5_pruned_... no termina en _ftN
    assert infer_stage_from_name("m_ft5_pruned_activation_20.pth") == "pruned"
```

- [ ] **Step 2: Verificar que fallan** — `uv run pytest tests/test_model_registry.py -v`

- [ ] **Step 3: Implementar**

```python
import re

def infer_stage_from_name(name: str) -> str:
    if name.endswith(".espdl"):
        return "espdl"
    if "_quantized" in name:
        return "quantized"
    stem = name[:-4] if name.endswith(".pth") else name
    if re.search(r"_ft\d+$", stem):
        return "finetuned"
    if "_pruned_" in name:
        return "pruned"
    return "baseline"
```

Actualizar el docstring con la regla nueva.

- [ ] **Step 4: Correr tests** — `uv run pytest tests/test_model_registry.py -v` — PASS.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(esp32-cam): registry clasifica modelos finetuneados (_ftN)" \
  -- model_registry.py tests/test_model_registry.py
```

---

### Task 6: `script05-build.py` — `--action build|flash|monitor`

**Files:**
- Modify: `script05-build.py`
- Test: `tests/test_script05_build.py`

**Interfaces:**
- Produces: CLI `--action {build,flash,monitor}` (default `build`). `build`: preflight (sin puerto) + `idf.py build`; NO flashea. `flash`: `check_idf_env` + `resolve_port` + `_run_flash`. `monitor`: `check_idf_env` + `resolve_port` + `idf.py monitor`. Flags `--monitor`, `--build-only`, `--monitor-only` eliminadas; `--menuconfig-only` se conserva.

- [ ] **Step 1: Tests que fallan**

Agregar a `tests/test_script05_build.py` (usa el `_load_module()` existente):

```python
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
```

- [ ] **Step 2: Verificar que fallan** — `uv run pytest tests/test_script05_build.py -v`

- [ ] **Step 3: Implementar**

En `build_parser()`: agregar

```python
p.add_argument("--action", default="build", choices=["build", "flash", "monitor"],
               help="Qué hacer: build (compila), flash (flashea lo ya compilado), "
                    "monitor (abre monitor serie). Default: build")
```

y eliminar `--monitor`, `--build-only`, `--monitor-only`. Reescribir `main()`:

```python
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
                install_model_data(args.source, Path.cwd() / "models", firmware_dir / "main")
            if args.preflight:
                check_idf_env()
                check_model_data_present(firmware_dir / "main")
                check_metadata_coherence(firmware_dir / "main")
                check_psram_enabled(firmware_dir)
            else:
                check_idf_env()
            return _run_idf(["idf.py", "build"], cwd=firmware_dir)

        if args.action == "flash":
            check_idf_env()
            port = resolve_port(args.port)
            return _run_flash(firmware_dir, port)

        # monitor
        check_idf_env()
        port = resolve_port(args.port)
        return _run_idf(["idf.py", "-p", port, "monitor"], cwd=firmware_dir)
    except PreflightError as e:
        print(f"preflight error: {e}", file=sys.stderr)
        return 1
```

Eliminar `_run_preflight` (queda inlineado en la rama build, sin `resolve_port`). Actualizar docstring del módulo y `prog=` a `script05-build.py`.

- [ ] **Step 4: Correr tests** — `uv run pytest tests/test_script05_build.py tests/ -v` — PASS. Los tests existentes que ejerciten `--build-only`/`--monitor-only`/`_run_preflight` se actualizan a las ramas nuevas de `--action` conservando qué chequean (p.ej. "monitor resuelve puerto", "build no necesita puerto").

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(esp32-cam): script05-build --action build|flash|monitor" \
  -- script05-build.py tests/test_script05_build.py
```

---

### Task 7: `pipelines.yaml` — flows con los stages nuevos

**Files:**
- Modify: `pipelines.yaml`
- Test: `uv run python main.py run <flow> --dry-run` para cada flow + suite completa.

**Interfaces:**
- Consumes: stages/scripts de Tasks 2, 4, 6.

- [ ] **Step 1: Reescribir los flows**

Reglas de conversión para cada flow existente:
- `prune` pierde `finetune_epochs` y `lr`; después de cada `prune` se agrega un stage `finetune` con `source: <output del prune>` y `params: {epochs: <ex finetune_epochs>, lr: <ex lr>}`.
- Los `source:` de `quantize` en adelante ganan el sufijo `_ft<N>`.
- El stage `build` final se parte en `build` → `flash` → `monitor` (build conserva `source` y `preflight`; flash/monitor sin params, el puerto se inyecta solo).
- `s3_fast_debug`: igual conversión; su `build` tenía `build_only: true` → queda solo `build` (sin flash/monitor); el `port` explícito se elimina (ya no aplica a build). NO tocar nada dentro de `examples/esp32-s3-cam/`.
- Renombrar `fast_debug_only_build` → `fast_debug_flash` (stages: `build` con `preflight: true` + `flash`), `fast_debug_only_monitor` → `fast_debug_monitor` (stage único `monitor`); ídem `good_model_only_build`/`small_model_only_build` → `..._flash`.

Ejemplo del flow `fast_debug` convertido:

```yaml
fast_debug:
  - stage: train
    params: {arch: depthwise, input_mode: rgb, resolution: 32, epochs: 5, lr: 1.0e-3, batch_size: 512}
  - stage: prune
    source: depthwise_rgb32
    params: {method: activation, ratio: 0.3}
  - stage: finetune
    source: depthwise_rgb32_pruned_activation_30
    params: {epochs: 2, lr: 2.0e-4}
  - stage: quantize
    source: depthwise_rgb32_pruned_activation_30_ft2
    variant: a
    params: {mode: ptq, calib_batches: 32}
  - stage: build
    source: depthwise_rgb32_pruned_activation_30_ft2_quantized_ptq
    params: {preflight: true}
  - stage: flash
  - stage: monitor
```

Actualizar el comentario top del YAML (nota del `port`: "se inyecta en stages flash y monitor").

- [ ] **Step 2: Verificar dry-run**

```bash
for f in $(uv run python main.py flows | awk '{print $1}' | grep -v '^$'); do
  uv run python main.py run "$f" --dry-run || echo "FAIL $f"
done
```

Expected: todos imprimen comandos, ninguno FAIL; los stages build/flash/monitor invocan `script05-build.py --action ...` y solo flash/monitor llevan `--port`.

- [ ] **Step 3: Suite completa** — `uv run pytest tests/ -v` — PASS (actualizar `tests/test_main_cli.py` si asume nombres de flows viejos).

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(esp32-cam): pipelines.yaml con stages finetune/build/flash/monitor" \
  -- pipelines.yaml tests/test_main_cli.py
```

---

### Task 8: Firmware — módulo `capture_state`

**Files:**
- Create: `firmware/main/capture_state.c`, `firmware/main/capture_state.h`
- Modify: `firmware/main/CMakeLists.txt` (agregar `capture_state.c` a SRCS)

**Interfaces:**
- Produces (consumido por Tasks 9-11):

```c
// capture_state.h
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "esp_err.h"
#include "inference.h"

typedef enum { CAPTURE_MODE_SINGLE = 0, CAPTURE_MODE_CONTINUOUS = 1 } capture_mode_t;

typedef struct {
    uint32_t           seq;              /* 0 = aún no hubo captura */
    capture_mode_t     mode;
    inference_result_t result;
    uint32_t           t_capture_ms;
    uint32_t           t_preprocess_ms;
    uint32_t           t_inference_ms;
    float              fps;              /* 0 en modo single */
    char               last_error[64];   /* "" si ok */
} capture_status_t;

esp_err_t capture_state_init(void);
capture_mode_t capture_state_get_mode(void);
/* devuelve el modo resultante */
capture_mode_t capture_state_set_mode(capture_mode_t m);
capture_mode_t capture_state_toggle_mode(void);
/* copia el status completo (thread-safe) */
void capture_state_get_status(capture_status_t *out);
void capture_state_publish_result(const inference_result_t *r,
                                  uint32_t t_cap_ms, uint32_t t_pre_ms,
                                  uint32_t t_inf_ms, float fps);
void capture_state_publish_error(const char *msg);
/* guarda copia del JPEG en PSRAM (reemplaza el anterior) */
void capture_state_store_jpeg(const uint8_t *buf, size_t len);
/* lockea y presta el jpeg; devolver con capture_state_release_jpeg().
 * *buf=NULL, *len=0 si no hay. */
void capture_state_borrow_jpeg(const uint8_t **buf, size_t *len);
void capture_state_release_jpeg(void);
```

- [ ] **Step 1: Implementar `capture_state.c`**

```c
#include "capture_state.h"
#include <string.h>
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

static SemaphoreHandle_t s_mutex;
static capture_status_t  s_status;   /* .mode = SINGLE, .seq = 0 */
static uint8_t          *s_jpeg;     /* PSRAM, realloc por captura */
static size_t            s_jpeg_len;
static size_t            s_jpeg_cap;

esp_err_t capture_state_init(void)
{
    s_mutex = xSemaphoreCreateMutex();
    return s_mutex ? ESP_OK : ESP_ERR_NO_MEM;
}

capture_mode_t capture_state_get_mode(void)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    capture_mode_t m = s_status.mode;
    xSemaphoreGive(s_mutex);
    return m;
}

capture_mode_t capture_state_set_mode(capture_mode_t m)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    s_status.mode = m;
    xSemaphoreGive(s_mutex);
    return m;
}

capture_mode_t capture_state_toggle_mode(void)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    s_status.mode = (s_status.mode == CAPTURE_MODE_SINGLE)
                    ? CAPTURE_MODE_CONTINUOUS : CAPTURE_MODE_SINGLE;
    capture_mode_t m = s_status.mode;
    xSemaphoreGive(s_mutex);
    return m;
}

void capture_state_get_status(capture_status_t *out)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    *out = s_status;
    xSemaphoreGive(s_mutex);
}

void capture_state_publish_result(const inference_result_t *r,
                                  uint32_t t_cap_ms, uint32_t t_pre_ms,
                                  uint32_t t_inf_ms, float fps)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    s_status.seq++;
    s_status.result          = *r;
    s_status.t_capture_ms    = t_cap_ms;
    s_status.t_preprocess_ms = t_pre_ms;
    s_status.t_inference_ms  = t_inf_ms;
    s_status.fps             = fps;
    s_status.last_error[0]   = '\0';
    xSemaphoreGive(s_mutex);
}

void capture_state_publish_error(const char *msg)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    strlcpy(s_status.last_error, msg, sizeof s_status.last_error);
    xSemaphoreGive(s_mutex);
}

void capture_state_store_jpeg(const uint8_t *buf, size_t len)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    if (len > s_jpeg_cap) {
        uint8_t *nb = heap_caps_realloc(s_jpeg, len, MALLOC_CAP_SPIRAM);
        if (nb == NULL) { xSemaphoreGive(s_mutex); return; }
        s_jpeg = nb;
        s_jpeg_cap = len;
    }
    memcpy(s_jpeg, buf, len);
    s_jpeg_len = len;
    xSemaphoreGive(s_mutex);
}

void capture_state_borrow_jpeg(const uint8_t **buf, size_t *len)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    *buf = s_jpeg_len ? s_jpeg : NULL;
    *len = s_jpeg_len;
    /* mutex queda tomado hasta release — el caller manda el buffer y libera */
}

void capture_state_release_jpeg(void)
{
    xSemaphoreGive(s_mutex);
}
```

Agregar `capture_state.c` a `SRCS` en `firmware/main/CMakeLists.txt`.

- [ ] **Step 2: Compilar**

```bash
. ~/esp/esp-idf/export.sh && cd firmware && idf.py build
```
Expected: `BUILD_OK` (aún nadie usa el módulo; solo debe compilar y linkear).

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(esp32-cam): firmware — capture_state (estado compartido con mutex)" \
  -- firmware/main/capture_state.c firmware/main/capture_state.h firmware/main/CMakeLists.txt
```

---

### Task 9: Firmware — loop de captura con modos, tiempos y FPS

**Files:**
- Modify: `firmware/main/app_main.c`, `firmware/main/trigger.h`

**Interfaces:**
- Consumes: `capture_state_*` (Task 8), `camera_capture/camera_return`, `preprocess_rgb565`, `inference_run`, `fmt2jpg` de `img_converters.h` (esp32-camera).
- Produces: `run_pipeline(trigger_source_t)` ahora mide etapas, guarda JPEG + resultado en capture_state y reporta por serie; el loop principal atiende modo continuo. `trigger.h` gana `TRIGGER_HTTP = 3`.

- [ ] **Step 1: `trigger.h` — fuente HTTP**

```c
typedef enum {
    TRIGGER_SERIAL = 1,
    TRIGGER_MQTT   = 2,
    TRIGGER_HTTP   = 3,
} trigger_source_t;
```

- [ ] **Step 2: `app_main.c` — pipeline con timing + estado**

Includes nuevos: `#include "capture_state.h"`, `#include "esp_timer.h"`, y bajo `CONFIG_ESP32CAM_CAMERA_ENABLED` `#include "img_converters.h"`. Reemplazar `run_pipeline`/`run_inference_on` por:

```c
/* Devuelve true si hubo resultado. fps<=0 → modo single (no se reporta). */
static bool run_pipeline(trigger_source_t src, float fps)
{
    int64_t t0 = esp_timer_get_time();
    const uint16_t *buf; int actual_w, actual_h;
#ifdef CONFIG_ESP32CAM_CAMERA_ENABLED
    camera_fb_t *fb = NULL;
    if (camera_capture(&fb) != ESP_OK || fb == NULL) {
        capture_state_publish_error("camera_capture failed");
        return false;
    }
    actual_w = (int)fb->width;
    actual_h = (fb->format == PIXFORMAT_RGB565 && actual_w > 0)
               ? (int)fb->len / (actual_w * 2) : (int)fb->height;
    buf = (const uint16_t *)fb->buf;

    /* JPEG para /last.jpg: encodear el RGB565 recién capturado */
    uint8_t *jpg = NULL; size_t jpg_len = 0;
    if (fmt2jpg(fb->buf, fb->len, actual_w, actual_h, PIXFORMAT_RGB565,
                80, &jpg, &jpg_len)) {
        capture_state_store_jpeg(jpg, jpg_len);
        free(jpg);
    }
#else
    if (g_synth_fb == NULL) { capture_state_publish_error("no synth frame"); return false; }
    buf = g_synth_fb; actual_w = SYNTH_W; actual_h = SYNTH_H;
#endif
    int64_t t1 = esp_timer_get_time();   /* fin captura+jpeg */

    uint8_t mean[3] = { MODEL_INPUT_MEAN_0,
#if MODEL_INPUT_CHANNELS == 3
                        MODEL_INPUT_MEAN_1, MODEL_INPUT_MEAN_2
#else
                        0, 0
#endif
                      };
    uint8_t std[3]  = { MODEL_INPUT_STD_0,
#if MODEL_INPUT_CHANNELS == 3
                        MODEL_INPUT_STD_1, MODEL_INPUT_STD_2
#else
                        1, 1
#endif
                      };
    preprocess_rgb565(buf, actual_w, actual_h, g_input_buf,
                      MODEL_INPUT_W, MODEL_INPUT_H, MODEL_INPUT_CHANNELS,
                      mean, std,
#ifdef CONFIG_ESP32CAM_RESIZE_BILINEAR
                      true
#else
                      false
#endif
    );
#ifdef CONFIG_ESP32CAM_CAMERA_ENABLED
    camera_return(fb);
#endif
    int64_t t2 = esp_timer_get_time();   /* fin preprocess */

    inference_result_t result;
    inference_run(g_input_buf, &result);
    int64_t t3 = esp_timer_get_time();   /* fin inferencia */

    uint32_t cap_ms = (uint32_t)((t1 - t0) / 1000);
    uint32_t pre_ms = (uint32_t)((t2 - t1) / 1000);
    uint32_t inf_ms = (uint32_t)((t3 - t2) / 1000);
    capture_state_publish_result(&result, cap_ms, pre_ms, inf_ms, fps > 0 ? fps : 0.0f);

    char text[160], json[768];
    result_format_text(&result, text, sizeof text);
    result_format_json(&result, json, sizeof json);
    printf("src=%s %s cap=%ums pre=%ums inf=%ums",
           src == TRIGGER_SERIAL ? "serial" : src == TRIGGER_HTTP ? "http" : "mqtt",
           text, (unsigned)cap_ms, (unsigned)pre_ms, (unsigned)inf_ms);
    if (fps > 0) printf(" fps=%.1f", fps);
    printf("\n%s\n", json);

#ifdef CONFIG_ESP32CAM_MQTT_ENABLED
    esp32cam_mqtt_publish_result(json);
#endif
    return true;
}
```

- [ ] **Step 3: `app_main.c` — loop principal con modo continuo**

Reemplazar el `while (1)` final de `app_main` (y agregar `capture_state_init()` justo después de `inference_init()`):

```c
    trigger_source_t src;
    float fps = 0.0f;
    while (1) {
        if (capture_state_get_mode() == CAPTURE_MODE_CONTINUOUS) {
            int64_t it0 = esp_timer_get_time();
            /* drenar triggers pendientes para que no se acumulen */
            while (xQueueReceive(trigger_queue, &src, 0) == pdTRUE) {}
            if (run_pipeline(TRIGGER_SERIAL, fps)) {
                int64_t dt_us = esp_timer_get_time() - it0;
                float inst = dt_us > 0 ? 1e6f / (float)dt_us : 0.0f;
                fps = (fps <= 0.0f) ? inst : (0.7f * fps + 0.3f * inst);
            } else {
                vTaskDelay(pdMS_TO_TICKS(200));
            }
        } else {
            fps = 0.0f;
            /* timeout corto para reaccionar si el modo cambia a continuo */
            if (xQueueReceive(trigger_queue, &src, pdMS_TO_TICKS(250)) == pdTRUE) {
                run_pipeline(src, 0.0f);
            }
        }
    }
```

- [ ] **Step 4: Compilar** — `cd firmware && idf.py build` — Expected: BUILD_OK.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(esp32-cam): firmware — loop single/continuo con tiempos y fps" \
  -- firmware/main/app_main.c firmware/main/trigger.h
```

---

### Task 10: Firmware — serie: `m` alterna modo

**Files:**
- Modify: `firmware/main/serial_trigger.c`

**Interfaces:**
- Consumes: `capture_state_toggle_mode()` (Task 8).
- Produces: tecla `m`/`M` → toggle + log; cualquier otro byte → trigger como hoy.

- [ ] **Step 1: Implementar**

En `serial_task` de `serial_trigger.c` (agregar `#include "capture_state.h"`):

```c
        int n = uart_read_bytes(UART_NUM, &byte, 1, portMAX_DELAY);
        if (n <= 0) continue;
        uart_flush_input(UART_NUM);
        if (byte == 'm' || byte == 'M') {
            capture_mode_t m = capture_state_toggle_mode();
            ESP_LOGI(TAG, "mode -> %s",
                     m == CAPTURE_MODE_CONTINUOUS ? "CONTINUOUS" : "SINGLE");
            continue;
        }
        trigger_source_t src = TRIGGER_SERIAL;
        xQueueSend(queue, &src, 0);
```

- [ ] **Step 2: Compilar** — `cd firmware && idf.py build` — BUILD_OK.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(esp32-cam): firmware — tecla m alterna single/continuo" \
  -- firmware/main/serial_trigger.c
```

---

### Task 11: Firmware — HTTP: página, /trigger, /mode, /last.jpg, /result

**Files:**
- Modify: `firmware/main/http_stream_server.c`, `firmware/main/http_stream_server.h`, `firmware/main/app_main.c` (pasar la queue al server)

**Interfaces:**
- Consumes: `capture_state_*`, `trigger_queue` (la misma de serial/MQTT), `STL10 class names` — el firmware ya conoce los nombres vía `MODEL_CLASS_NAMES` si existe en `model_data.h`; verificar con `grep CLASS model_data.h`. Si no existe, usar índices numéricos en el JSON (`"class_names"` omitido) y mostrar `#idx` en la página — NO inventar nombres.
- Produces: `http_stream_server_start(QueueHandle_t trigger_queue)` (firma nueva); endpoints `GET /`, `GET /stream`, `POST /trigger`, `POST /mode` (body `continuous=0|1`), `GET /last.jpg`, `GET /result`.

- [ ] **Step 1: Cambiar firma e includes**

`http_stream_server.h`:

```c
#pragma once
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
esp_err_t http_stream_server_start(QueueHandle_t trigger_queue);
```

En `app_main.c`: `http_stream_server_start(trigger_queue);`. En el stub `#else` del `.c`: misma firma, `(void)trigger_queue;`.

- [ ] **Step 2: Handlers nuevos en `http_stream_server.c`**

Includes: `capture_state.h`, `trigger.h`, `cJSON.h`, `<stdlib.h>`, `<string.h>`. Variable estática `static QueueHandle_t s_trigger_queue;` (asignada en start).

```c
static esp_err_t trigger_handler(httpd_req_t *req)
{
    trigger_source_t src = TRIGGER_HTTP;
    xQueueSend(s_trigger_queue, &src, 0);
    httpd_resp_set_status(req, "204 No Content");
    return httpd_resp_send(req, NULL, 0);
}

static esp_err_t mode_handler(httpd_req_t *req)
{
    char body[32] = {0};
    int n = httpd_req_recv(req, body, sizeof body - 1);
    if (n <= 0) { httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "empty body"); return ESP_FAIL; }
    /* body esperado: continuous=0 | continuous=1 */
    const char *p = strstr(body, "continuous=");
    if (!p) { httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "missing continuous="); return ESP_FAIL; }
    capture_state_set_mode(p[11] == '1' ? CAPTURE_MODE_CONTINUOUS : CAPTURE_MODE_SINGLE);
    httpd_resp_set_status(req, "204 No Content");
    return httpd_resp_send(req, NULL, 0);
}

static esp_err_t last_jpg_handler(httpd_req_t *req)
{
    const uint8_t *buf; size_t len;
    capture_state_borrow_jpeg(&buf, &len);
    if (buf == NULL) {
        capture_state_release_jpeg();
        httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "no capture yet");
        return ESP_FAIL;
    }
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    esp_err_t res = httpd_resp_send(req, (const char *)buf, len);
    capture_state_release_jpeg();
    return res;
}

static esp_err_t result_handler(httpd_req_t *req)
{
    capture_status_t st;
    capture_state_get_status(&st);

    cJSON *root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "seq", st.seq);
    cJSON_AddStringToObject(root, "mode",
        st.mode == CAPTURE_MODE_CONTINUOUS ? "continuous" : "single");
    cJSON_AddStringToObject(root, "last_error", st.last_error);
    if (st.seq > 0) {
        cJSON_AddNumberToObject(root, "top1", st.result.top1_idx);
        cJSON_AddNumberToObject(root, "top1_confidence", st.result.top1_confidence);
        cJSON *probs = cJSON_AddArrayToObject(root, "probs");
        for (int i = 0; i < MODEL_NUM_CLASSES; i++)
            cJSON_AddItemToArray(probs, cJSON_CreateNumber(st.result.probs[i]));
        cJSON_AddNumberToObject(root, "t_capture_ms", st.t_capture_ms);
        cJSON_AddNumberToObject(root, "t_preprocess_ms", st.t_preprocess_ms);
        cJSON_AddNumberToObject(root, "t_inference_ms", st.t_inference_ms);
        cJSON_AddNumberToObject(root, "fps", st.fps);
    }
    char *out = cJSON_PrintUnformatted(root);
    httpd_resp_set_type(req, "application/json");
    esp_err_t res = httpd_resp_send(req, out, HTTPD_RESP_USE_STRLEN);
    cJSON_free(out);
    cJSON_Delete(root);
    return res;
}
```

(Si `model_data.h` define nombres de clase, agregar el array `"class_names"` al JSON con esos strings.)

- [ ] **Step 3: Página `/` nueva**

Reemplazar `INDEX_HTML` (JS vanilla, poll 500 ms; el `<img>` de última captura se refresca solo cuando cambia `seq`):

```c
static const char INDEX_HTML[] =
"<!DOCTYPE html><html><head><meta charset='utf-8'><title>ESP32-CAM</title><style>"
"body{background:#111;color:#eee;font-family:sans-serif;margin:0;padding:1em;text-align:center}"
"img{max-width:100%;border:2px solid #444;border-radius:4px}"
".row{display:flex;flex-wrap:wrap;gap:1em;justify-content:center}"
".col{flex:1 1 320px;max-width:640px}"
"table{margin:0 auto;border-collapse:collapse}td,th{padding:2px 8px;text-align:left}"
"button{padding:.5em 1.5em;font-size:1em;margin:.5em}"
"#err{color:#f66}"
"</style></head><body>"
"<h3>ESP32-CAM — inferencia</h3>"
"<div class='row'>"
"<div class='col'><h4>En vivo</h4><img src='/stream'></div>"
"<div class='col'><h4>Ultima captura</h4><img id='last' alt='(sin captura)'>"
"<div><button onclick='trig()'>Capturar</button>"
"<label><input type='checkbox' id='cont' onchange='mode()'> Continuo</label></div>"
"<div id='err'></div><table id='res'></table></div>"
"</div><script>"
"let seq=0;"
"function trig(){fetch('/trigger',{method:'POST'})}"
"function mode(){fetch('/mode',{method:'POST',body:'continuous='+(cont.checked?1:0)})}"
"async function poll(){try{"
"const r=await(await fetch('/result')).json();"
"cont.checked=(r.mode=='continuous');"
"err.textContent=r.last_error||'';"
"if(r.seq>0){"
"if(r.seq!=seq){seq=r.seq;last.src='/last.jpg?'+seq;}"
"let h='<tr><th>top1</th><td>'+r.top1+' ('+(r.top1_confidence*100).toFixed(1)+'%)</td></tr>';"
"h+='<tr><th>captura</th><td>'+r.t_capture_ms+' ms</td></tr>';"
"h+='<tr><th>preproceso</th><td>'+r.t_preprocess_ms+' ms</td></tr>';"
"h+='<tr><th>inferencia</th><td>'+r.t_inference_ms+' ms</td></tr>';"
"if(r.fps>0)h+='<tr><th>fps</th><td>'+r.fps.toFixed(1)+'</td></tr>';"
"r.probs.forEach((p,i)=>{h+='<tr><td>'+((r.class_names&&r.class_names[i])||('#'+i))+'</td>"
"<td>'+(p*100).toFixed(1)+'%</td></tr>';});"
"res.innerHTML=h;}"
"}catch(e){}setTimeout(poll,500)}"
"poll();"
"</script></body></html>";
```

- [ ] **Step 4: Registrar URIs en `http_stream_server_start`**

```c
    s_trigger_queue = trigger_queue;
    cfg.max_uri_handlers = 8;
    ...
    const httpd_uri_t trigger_uri = { .uri = "/trigger",  .method = HTTP_POST, .handler = trigger_handler };
    const httpd_uri_t mode_uri    = { .uri = "/mode",     .method = HTTP_POST, .handler = mode_handler };
    const httpd_uri_t last_uri    = { .uri = "/last.jpg", .method = HTTP_GET,  .handler = last_jpg_handler };
    const httpd_uri_t result_uri  = { .uri = "/result",   .method = HTTP_GET,  .handler = result_handler };
    httpd_register_uri_handler(server, &trigger_uri);
    httpd_register_uri_handler(server, &mode_uri);
    httpd_register_uri_handler(server, &last_uri);
    httpd_register_uri_handler(server, &result_uri);
```

Los 4 endpoints nuevos van FUERA del `#ifdef CONFIG_ESP32CAM_CAMERA_ENABLED` (funcionan con frame sintético); solo `/stream` queda adentro.

- [ ] **Step 5: Compilar** — `cd firmware && idf.py build` — BUILD_OK.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(esp32-cam): firmware — web con captura, modo continuo y resultado" \
  -- firmware/main/http_stream_server.c firmware/main/http_stream_server.h firmware/main/app_main.c
```

---

### Task 12: Documentación + backlog

**Files:**
- Modify: `CLAUDE.md` (tabla de módulos, sección script03*→04*, stages, preflight, "Qué no hacer"), `README.md`

**Interfaces:** ninguna.

- [ ] **Step 1: Actualizar `CLAUDE.md`**

- Renumeración completa (tabla "Los tres script03*" → "Los tres script04*"; `script04-build.py` → `script05-build.py` con `--action`).
- Stages: `train, prune, finetune, quantize, build, flash, monitor`; prune ya no finetunea; sin validación de dependencias.
- Convenciones de nombres: agregar `_ft<N>` → finetuned al listado.
- Firmware: mencionar `capture_state.{c,h}`, modo single/continuo (`m` por serie, checkbox web), endpoints nuevos.

- [ ] **Step 2: Actualizar `README.md` + backlog**

Sección "Backlog": "Adaptar quantize variantes b (torchao) y c (brevitas) al esquema prune-sin-finetune y agregarlas a flows". Actualizar cualquier ejemplo de comando con nombres de script viejos.

- [ ] **Step 3: Suite completa final** — `uv run pytest tests/ -v` — PASS.

- [ ] **Step 4: Commit + push**

```bash
git commit -m "docs(esp32-cam): renumeración de scripts, stages nuevos y backlog" \
  -- CLAUDE.md README.md
git push
```
