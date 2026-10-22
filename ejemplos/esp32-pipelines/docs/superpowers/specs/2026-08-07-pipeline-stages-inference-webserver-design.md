# Diseño: etapas de pipeline separadas + inferencia interactiva en firmware

Fecha: 2026-08-07
Directorio: `examples/esp32-cam/`

## Objetivo

1. Reestructurar el pipeline en etapas atómicas: `train`, `prune`, `finetune`,
   `quantize`, `build`, `flash`, `monitor`. `prune` deja de finetunear;
   `finetune` es una etapa explícita y repetible. `build` solo compila;
   flasheo y monitor son etapas propias.
2. Firmware `firmware/`: captura+inferencia disparada por carácter serie o
   botón web, modo single-shot/continuo, y webserver que muestra stream en
   vivo, última captura y resultado completo con tiempos por etapa y FPS.

## Decisiones tomadas (con el usuario)

- `prune` solo poda; el finetune es siempre un stage aparte.
- Un solo script de build con `--action build|flash|monitor` (no tres scripts).
- **Sin validación de dependencias entre stages** — responsabilidad del usuario.
  Los flows parciales (solo build, solo monitor, etc.) siguen siendo válidos.
- Trigger de inferencia por serie **y** por botón en la web.
- Modo single-shot ↔ continuo: tecla `m` por serie, checkbox en la web.
  Continuo = loop de captura+inferencia una tras otra.
- Resultado reporta: clase top-1, las 10 probabilidades STL-10, tiempos de
  captura / preprocesado / inferencia en ms, y FPS en modo continuo.
- La web actualiza por polling de `/result` (JSON) cada ~500 ms; sin SSE/WS.
- Solo la variante esp-ppq (`a`) se usa en los flows por ahora. **Backlog**:
  adaptar torchao (`b`) y brevitas (`c`) al esquema prune-sin-finetune.

## Renumeración de scripts

| Script nuevo | Origen | Stage |
|---|---|---|
| `script01-train.py` | igual | train |
| `script02-prune.py` | igual, **sin** finetune interno | prune |
| `script03-finetune.py` | nuevo | finetune |
| `script04a-quantize.py` | ex `script03a-quantize.py` | quantize (variant a) |
| `script04b-quantize-torchao.py` | ex `script03b` | quantize (variant b) |
| `script04c-quantize-brevitas.py` | ex `script03c` | quantize (variant c) |
| `script05-build.py` | ex `script04-build.py` + `--action` | build / flash / monitor |

## Parte 1 — Pipeline Python

### `pipeline_runner.py`

- `VALID_STAGES = {train, prune, finetune, quantize, build, flash, monitor}`.
- `STAGES_REQUIRING_SOURCE = {prune, finetune, quantize}`.
- Mapa de scripts según la tabla de arriba. `build`, `flash` y `monitor`
  ejecutan `script05-build.py` con `--action <stage>` inyectado.
- El `port:` top-level del YAML se inyecta como default en `flash` y
  `monitor` (ya no en `build`).

### `script02-prune.py`

- Se eliminan `--finetune-epochs` y el lr de finetune. Poda y guarda
  `<source>_pruned_<method>_<ratio>.pth`.

### `script03-finetune.py` (nuevo)

- Args: `--source <nombre>`, `--epochs N`, `--lr`, `--batch-size`.
- Carga el `.pth`, reusa el training loop de `esp32cam_utils.py`, guarda
  `<source>_ft<N>.pth` y registra métricas en `models.json`.
- Repetible: `depthwise_rgb32_pruned_activation_30_ft5_pruned_activation_20_ft5`.

### `model_registry.py`

- Nueva categoría `finetuned`: nombre que termina en `_ft<N>` (y no matchea
  `quantized`/`espdl`). Orden de precedencia de clasificación:
  `espdl > quantized > finetuned > pruned > baseline`.

### `script05-build.py`

- `--action build|flash|monitor` (default `build`).
  - `build`: preflight de IDF + coherencia header↔sidecar + PSRAM; compila.
    No requiere puerto.
  - `flash`: preflight de puerto (`resolve_port`); `idf.py -p <port> flash`.
  - `monitor`: solo monitor serie.
- Las flags actuales `monitor`/`monitor_only` desaparecen de los flows
  (reemplazadas por stages).

### `pipelines.yaml`

- Flows reescritos, ej.:

```yaml
fast_debug:
  - {stage: train, params: {...}}
  - {stage: prune,    source: depthwise_rgb32, params: {method: activation, ratio: 0.3}}
  - {stage: finetune, source: depthwise_rgb32_pruned_activation_30, params: {epochs: 2, lr: 2.0e-4}}
  - {stage: quantize, source: depthwise_rgb32_pruned_activation_30_ft2, variant: a, params: {...}}
  - {stage: build,    source: ..._quantized_ptq}
  - {stage: flash}
  - {stage: monitor}
```

- Los `source:` siguen siendo strings mantenidos a mano (sin validación cruzada).

### Tests

- Actualizar `tests/` a los nuevos nombres/stages; tests nuevos para
  `script03-finetune.py` (naming del output, registro) y para
  `--action` en `script05-build.py`. `uv run pytest tests/` debe pasar.

## Parte 2 — Firmware `firmware/`

### Estado compartido (nuevo módulo `capture_state.{c,h}`)

Protegido por mutex:

- `mode`: `SINGLE` | `CONTINUOUS`.
- Último JPEG capturado (buffer en PSRAM, copiado del fb).
- Último resultado: `top1_idx`, `probs[10]`, `t_capture_ms`,
  `t_preprocess_ms`, `t_inference_ms`, `fps` (solo continuo), `capture_seq`
  (contador monotónico).

### Loop de captura/inferencia

- Tarea única que ejecuta: captura → copia JPEG al estado → preprocesado →
  inferencia → publica resultado (serie + estado). Mide cada etapa con
  `esp_timer_get_time()`.
- En `SINGLE`: espera trigger (serie o HTTP) en una queue/semáforo.
- En `CONTINUOUS`: itera sin pausa; calcula FPS con media móvil y lo
  publica en cada iteración.

### Serie (`serial_trigger.c`)

- `m` → alterna modo (log del modo nuevo).
- Cualquier otra tecla → trigger (solo tiene efecto en `SINGLE`).
- Reporte por serie en cada inferencia: top-1, 10 probabilidades, tiempos
  por etapa, FPS si continuo.

### HTTP (`http_stream_server.c`)

- `GET /` — página única: `<img src=/stream>` (en vivo), `<img>` de la
  última captura (refresca cuando cambia `capture_seq`), tabla de
  resultado, botón **Capturar** (POST `/trigger`), checkbox **Continuo**
  (POST `/mode`). JS vanilla, polling de `/result` cada 500 ms.
- `GET /stream` — MJPEG (existente).
- `POST /trigger` — dispara una captura (204).
- `POST /mode` — body `continuous=0|1` (204).
- `GET /last.jpg` — último JPEG capturado (404 si aún no hay).
- `GET /result` — JSON: `{seq, mode, top1, class_names[10], probs[10],
  t_capture_ms, t_preprocess_ms, t_inference_ms, fps}`.

### Manejo de errores

- Captura/inferencia fallida: se loguea y el resultado previo queda
  intacto; `/result` incluye `last_error` string (vacío si ok).
- `/last.jpg` y `/stream` compiten por la cámara: la copia del JPEG al
  estado se hace inmediatamente tras `esp_camera_fb_get`/`return`, nunca
  se sirve el fb en vivo desde dos handlers.

## Documentación

- Actualizar `CLAUDE.md` (tabla de módulos, stages, renumeración) y
  `README.md` del ejemplo. Backlog (README): variantes quantize b/c.

## Fuera de alcance

- Validación de dependencias entre stages.
- Variantes quantize torchao/brevitas en flows.
- MQTT (queda como está).
- `firmware_test_webserver/` no se toca.
