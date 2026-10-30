"""Exporta las métricas de todos los modelos entrenados a metricas_modelos.csv.

Los .pth de models/ no se versionan, así que este CSV es lo que queda en el repo
para graficar el compromiso accuracy / tamaño / operaciones (por ejemplo, la
frontera de Pareto de la diapo "Muchas combinaciones posibles"). Para los
modelos que además se desplegaron, agrega el tiempo medido en la ESP32-CAM y el
tamaño del .espdl, tomados de benchmarks.json.

Uso (desde examples/esp32-pipelines/):
    uv run python tools/export_metricas.py
"""

import contextlib
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipelib"))

import torch  # noqa: E402

from esp32cam_utils import count_macs, count_parameters, load_checkpoint  # noqa: E402

SALIDA = ROOT / "metricas_modelos.csv"


def medidas_esp32cam() -> dict:
    """Tiempo (ms) y tamaño del .espdl por modelo base, en ESP32-CAM con ESP-DL."""
    bench = json.loads((ROOT / "benchmarks.json").read_text())["benchmarks"]
    out = {}
    for b in bench:
        if b["board"] == "esp32cam" and b["runtime"] == "espdl":
            base = b["model"].removesuffix("_qptq")
            out[base] = (b["inf_ms_mean"], b["model_bytes"], b["measured_at"])
    return out


def main() -> None:
    medidas = medidas_esp32cam()
    filas = []
    for path in sorted((ROOT / "models").glob("*.pth")):
        # load_checkpoint imprime la ruta de cada .pth: con cientos, molesta.
        with contextlib.redirect_stdout(io.StringIO()):
            model, meta = load_checkpoint(path, torch.device("cpu"))
        res = meta["resolution"]
        in_ch = 1 if meta["input_mode"] == "gray" else 3
        ms, nbytes, fecha = medidas.get(path.stem, ("", "", ""))
        filas.append({
            "modelo": path.stem,
            "accuracy": f"{meta['accuracy']:.4f}",
            "parametros": count_parameters(model),
            "macs": count_macs(model, (1, in_ch, res, res)),
            "resolucion": res,
            "esp32cam_ms": ms,
            "espdl_bytes": nbytes,
            "medido": fecha,
        })

    with SALIDA.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"{len(filas)} modelos -> {SALIDA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
