# IA Embebida

Repositorio del curso de Inteligencia Artificial Embebida, CEIA/CESE, FIUBA Posgrado.

El curso cubre técnicas de compresión de redes neuronales (cuantización, pruning,
factorización, NAS, knowledge distillation) y su despliegue en dispositivos de
recursos limitados.

## Ejercicios

Cada tema vive bajo `ejemplos/<tema>/` con su propio `pyproject.toml` y entorno
virtual.
El proyecto usa [uv](https://docs.astral.sh/uv/) como gestor de dependencias.
Al ejecutar `uv sync` se crea automáticamente el `.venv`.

| Carpeta | Framework |
|---------|-----------|
| `ejemplos/quantization/` | TensorFlow / QKeras |
| `ejemplos/pruning/` | PyTorch |
| `ejemplos/factorization/` | PyTorch |
| `ejemplos/optimization/` | PyTorch |
| `ejemplos/nas/` | PyTorch |
| `ejemplos/knowledge_distillation/` | PyTorch |
| `ejemplos/esp32-pipelines/` | PyTorch + esp-ppq + torchao + brevitas |

Instalar dependencias y correr un script:

```bash
cd ejemplos/pruning
uv sync
uv run python script01.py
```

Los scripts se numeran `script01.py`, `script02.py`,... dentro de cada carpeta,
ordenados de menor a mayor complejidad.
La carpeta `ejemplos/datasets/` es un caché compartido de datasets que se
descargan automáticamente la primera vez.

## Presentación

Para levantar un servidor local y abrir la presentación en el navegador:

```bash
./run_presentation.sh
```

El script busca un puerto libre (a partir del 8000), inicia un servidor HTTP y
abre `clases/main.html` en el navegador.
Las slides **deben servirse vía HTTP** (no como `file://`) porque Reveal.js
carga los archivos de sección de forma dinámica.

Presionar `Ctrl+C` para detener el servidor.
