# IA Embebida

Repositorio del curso de Inteligencia Artificial Embebida — FIUBA Posgrado.

## Ejercicios

### Instalar dependencias

El proyecto usa [uv](https://docs.astral.sh/uv/) para la gestión de dependencias. Al ejecutar `uv sync` se crea automáticamente el entorno virtual (`.venv`):

```bash
cd ejercicios && uv sync
```

### Ejecutar un script

```bash
cd ejercicios && uv run python script06b.py
```

## Presentación

Para levantar un servidor local y abrir la presentación en el navegador:

```bash
./run_presentation.sh
```

El script busca un puerto libre (a partir del 8000), inicia un servidor HTTP y abre la presentación en Google Chrome. Presionar `Ctrl+C` para detener el servidor.