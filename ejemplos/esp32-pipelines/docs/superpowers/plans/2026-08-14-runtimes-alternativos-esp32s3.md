# Runtimes alternativos en ESP32-S3 — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Medir los 4 modelos existentes en ESP32-S3 (devkit N16R8, ttyUSB0, sin cámara) bajo 4 runtimes adicionales — TFLM sin esp-nn, TFLM+esp-nn, onnx2c y ExecuTorch — y volcar todo a la misma tabla de `main.py models`.

**Architecture:** Cada runtime nuevo = (a) un script de export en PC que genera un header C con el modelo + golden images + logits de referencia calculados con la MISMA toolchain en PC, y (b) un proyecto firmware ESP-IDF minimalista (`firmware_rt/<runtime>/`) que corre el self-test por UART con el formato de línea que ya parsea `script06-bench.py`. La dimensión `runtime` se agrega a `benchmarks.json` y a la tabla.

**Tech Stack:** ESP-IDF 5.3.2, `espressif/esp-tflite-micro` (con y sin esp-nn), `onnx2c` (compilado de fuente), ExecuTorch (runtime portable cross-compilado a Xtensa), ai-edge-litert + onnx2tf en venv aislado.

## Global Constraints

- Solo board `esp32s3` (devkit, PSRAM 8MB OK, puerto CH343/CP2102 con auto-reset). NO tocar `examples/esp32-s3-cam/`, `examples/esp32/`, `firmware_test_webserver/`, ni CLAUDE.md raíz.
- Commits SIEMPRE por paths explícitos (el index compartido tiene staged de otra sesión).
- Modelos (fuente = checkpoints ya entrenados en `models/`, NO re-entrenar):
  1. `depthwise_rgb32_pruned_activation_30_ft2` (5ep)
  2. `depthwise_rgb32_pruned_activation_50_ft10` (30ep)
  3. `standard_rgb32_pruned_activation_30_ft10`
  4. `standard_rgb96_pruned_activation_20_ft10` (good_model)
- Formato de self-test por UART idéntico al actual (parseado por `_SELFTEST_LINE` en `script06-bench.py`):
  `  <clase>  -> <pred>  <pct>% ... dlogits<=<N>  <ms>ms  PASS|FAIL` y al final el firmware imprime `ready`ANTES del test (igual que hoy: bench espera "ready", manda 't').
- Golden images: las mismas 10 de STL-10 que usa el flujo espdl (una por clase, margen ≥0.15). La referencia de logits se calcula en PC con el runtime correspondiente (TFLite interpreter para TFLM → dlogits=0 esperado; onnxruntime para onnx2c y ExecuTorch en float → comparar con tolerancia float, reportar `dlogits<=0` si max|Δ|<1e-3 escalado, ver Task de cada firmware).
- Python pesado (TF/onnx2tf, executorch) va en venvs aislados bajo `tools/rt_export/<runtime>/` con su propio `pyproject.toml` + `uv run --project`, para no contaminar el env principal.
- Deliverable final: `uv run main.py models` muestra columnas por runtime y `benchmarks.json` trackeado con todas las mediciones.

## Estructura de archivos nueva

```
firmware_rt/
  common/                 # componente compartido: selftest_uart.c (menú c/t, formato de línea), preprocessing
  tflm/                   # proyecto ESP-IDF (esp-tflite-micro); flag TFLM_ESP_NN=0/1 → build_ref/ y build_espnn/
  onnx2c/                 # proyecto ESP-IDF, model.c generado
  executorch/             # proyecto ESP-IDF, runtime ET vendorizado
tools/rt_export/
  tflite/  (pyproject: onnx2tf, ai-edge-litert, tensorflow-cpu)  → script export_tflite.py
  onnx2c/  (repo onnx2c clonado + build cmake)                  → export_onnx2c.py
  executorch/ (pyproject: executorch)                           → export_executorch.py
```

---

### Task 1: dimensión `runtime` en benchmarks + tabla

**Files:** Modify: `model_registry.py`, `esp32cam_utils.py` (print_comparison_table), `script06-bench.py`, `tests/test_model_registry.py`, `tests/test_bench_parse.py`.

- [ ] Test: `upsert_benchmark` con key (model, board, runtime); entradas viejas sin `runtime` se leen como `runtime="espdl"`.
- [ ] Implementar: campo `runtime` (default "espdl") en load/save/upsert/benchmarks_by_model_board→`benchmarks_by_key(model, board, runtime)`.
- [ ] Tabla: columnas `t_<board>[:<runtime>] [ms]`, orden runtimes: espdl, tflm, tflm_espnn, onnx2c, executorch; solo mostrar columnas con datos. Si queda muy ancha, agrupar: para boards≠esp32s3 solo espdl.
- [ ] `script06-bench.py --runtime {espdl,tflm,tflm_espnn,onnx2c,executorch}` (default espdl) + `--firmware-dir` ya existente para apuntar a `firmware_rt/<rt>/` y `--build-dir` override (tflm usa build_ref/build_espnn).
- [ ] `uv run pytest tests/ -v` verde; commit por paths.

### Task 2: componente firmware común `firmware_rt/common/`

**Files:** Create: `firmware_rt/common/` (componente IDF: `selftest_runner.h/.c`), consumido por los 3 proyectos.

**Interfaz (Produces):**
```c
typedef struct { const char *name; const uint8_t *data; } rt_golden_t; // data = input YA preprocesado al formato del runtime
typedef int (*rt_infer_fn)(const void *input, float *logits_out, int n_classes); // devuelve ms o <0 error
void selftest_run(const rt_golden_t *imgs, int n, rt_infer_fn infer,
                  const float *ref_logits, int n_classes, const char **class_names,
                  float tol); // imprime líneas formato bench y resumen
void selftest_uart_loop(...); // imprime "ready", espera 't' (test) / 'c' (una inferencia), ignora resto
```
- [ ] Implementar; unit-testeable en host no hace falta — validación es on-device. Commit.

### Task 3: export TFLite int8

**Files:** Create: `tools/rt_export/tflite/pyproject.toml`, `tools/rt_export/tflite/export_tflite.py`.

- [ ] venv aislado: deps `tensorflow-cpu`, `onnx2tf`, `ai-edge-litert`, `numpy`, `onnx`.
- [ ] Pipeline: cargar `.pth` (reutilizar builders de `esp32cam_utils.py` vía sys.path), export ONNX → `onnx2tf` (NCHW→NHWC) → converter TFLite full-integer PTQ con representative dataset = mismas imágenes de calibración STL-10 que usa script04a (mismo split/normalización — leer `_NORM_STATS`).
- [ ] Emitir `firmware_rt/tflm/main/model_data.h` (array tflite + len), `golden_data.h` (10 imágenes cuantizadas al input int8 del modelo TFLite + logits int8 de referencia del TFLite interpreter de PC) y `model_meta.json` (runtime, modelo fuente, input scale/zp, accuracy en test set con el interpreter).
- [ ] Correr para los 4 modelos; verificar accuracy PC ≈ la del espdl (±3pt; si standard96 diverge, anotar). Commit (scripts + metas; headers gitignoreados como los actuales).

### Task 4: firmware TFLM (ref y esp-nn)

**Files:** Create: `firmware_rt/tflm/` (proyecto IDF: main + `idf_component.yml` con `espressif/esp-tflite-micro`).

- [ ] `main.c/cc`: MicroInterpreter + MutableOpResolver con las ops necesarias (conv2d, depthwise, pool, fc, reshape, softmax), arena en PSRAM (`heap_caps_malloc` SPIRAM, tamaño por Kconfig, default 2MB), integrar `selftest_runner`. dlogits = logits int8 device vs referencia (esperado 0).
- [ ] Variantes: opción CMake `-DTFLM_ESP_NN=0/1`. Con 0: forzar kernels de referencia (esp-tflite-micro: setear `CONFIG_NN_OPTIMIZATIONS`/`ESP_NN` off — investigar la flag exacta del componente; si no existe knob, compilar con `-DESP_NN=0` en el componente o excluir esp-nn). Build dirs `build_ref/` y `build_espnn/` con `-DSDKCONFIG=<builddir>/sdkconfig`.
- [ ] Build ambos para 1 modelo, flash a ttyUSB0, self-test 10/10. Luego bench 4 modelos × 2 variantes vía `script06-bench.py --runtime tflm|tflm_espnn`. Commit.

### Task 5: onnx2c

**Files:** Create: `tools/rt_export/onnx2c/build.sh` (clona https://github.com/kraiskil/onnx2c, cmake build), `export_onnx2c.py`, `firmware_rt/onnx2c/`.

- [ ] Export ONNX float simplificado (onnxsim) del `.pth` → `onnx2c model.onnx > model.c`. Referencia: logits float de onnxruntime en PC sobre las 10 golden (input float normalizado).
- [ ] Firmware: compilar `model.c` (C puro, float), entrada float, `selftest_runner` con tolerancia float (PASS si max|Δlogit|<1e-3). Pesos float32 en flash: good_model puede ser grande — si `model.c` no compila/entra (>8MB rodata), registrar "no entra" y seguir (la partición factory es 4MB: probablemente ampliar `partitions.csv` propio de este proyecto a app 8MB usando los 16MB del devkit).
- [ ] Bench 4 modelos `--runtime onnx2c`. Commit.

### Task 6: ExecuTorch (spike con timebox)

**Files:** Create: `tools/rt_export/executorch/` (pyproject: `executorch`), `export_executorch.py`, `firmware_rt/executorch/`.

- [ ] Export: `torch.export` + `to_edge` + portable backend → `.pte` embebido como array C. Sin cuantización delegada (XNNPACK no aplica): float con kernels portables.
- [ ] Firmware: cross-compilar el runtime ET (core + portable kernels) con el toolchain Xtensa vía CMake de ESP-IDF (ET soporta CMake embebido; seguir doc "Building from source for embedded"). ESTE ES EL RIESGO GRANDE: timebox ~3 rondas de intento de build; si no linkea, documentar el bloqueo en README (sección backlog) con lo aprendido y cerrar la task como "no viable hoy".
- [ ] Si builda: referencia = logits del runtime ET en PC (python), tolerancia float; bench 4 modelos `--runtime executorch`.

### Task 7: docs + cierre

- [ ] `CLAUDE.md` y `README.md`: sección "Runtimes alternativos (firmware_rt/)" — qué es cada uno, cómo regenerar, tabla final de resultados con análisis (SIMD on/off, int8 vs float, overhead de runtime vs C plano).
- [ ] `benchmarks.json` final trackeado; `uv run main.py models` renderiza todo. Commit final.

## Orden y paralelismo

Task 1 y 2 primero (infra). Task 3→4 (TFLM) es el camino crítico. Task 5 (onnx2c) independiente, puede ir en paralelo con 4. Task 6 al final (mayor riesgo). El flasheo/bench requiere la devkit en ttyUSB0 — serializar el acceso al puerto.
