# IA Embebida

Repositorio del curso de Inteligencia Artificial Embebida — CEIA, FIUBA Posgrado.

El curso cubre técnicas de compresión de redes neuronales (cuantización, pruning,
factorización, NAS, knowledge distillation) y su despliegue en dispositivos de
recursos limitados.

## Ejercicios

Cada tema vive bajo `ejercicios/<tema>/` con su propio `pyproject.toml` y entorno
virtual. El proyecto usa [uv](https://docs.astral.sh/uv/) como gestor de
dependencias — al ejecutar `uv sync` se crea automáticamente el `.venv`.

| Carpeta | Framework |
|---------|-----------|
| `ejercicios/quantization/` | TensorFlow / QKeras |
| `ejercicios/pruning/` | PyTorch |
| `ejercicios/factorization/` | PyTorch |
| `ejercicios/optimization/` | PyTorch |
| `ejercicios/nas/` | PyTorch |
| `ejercicios/knowledge_distillation/` | PyTorch |
| `ejercicios/esp32-cam/` | PyTorch + esp-ppq + torchao + brevitas |

Instalar dependencias y correr un script:

```bash
cd ejercicios/pruning
uv sync
uv run python script01.py
```

Los scripts se numeran `script01.py`, `script02.py`, … dentro de cada carpeta,
ordenados de menor a mayor complejidad. La carpeta `ejercicios/datasets/` es un
caché compartido de datasets que se descargan automáticamente la primera vez.

## Presentación

Para levantar un servidor local y abrir la presentación en el navegador:

```bash
./run_presentation.sh
```

El script busca un puerto libre (a partir del 8000), inicia un servidor HTTP y
abre `clases/main.html` en el navegador. Las slides **deben servirse vía HTTP**
(no como `file://`) porque Reveal.js carga los archivos de sección de forma
dinámica.

Presionar `Ctrl+C` para detener el servidor.
