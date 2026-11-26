"""Figuras de la sección "Factorización".

Genera, con el estilo oscuro de las diapos:
  - una foto comprimida con la SVD a distintos rangos (la del gato de la demo de la clase 1);
  - el espectro de la primera capa del MLP (energía acumulada, pesos iniciales vs. entrenados);
  - accuracy, MACs y parámetros vs. rango de la SVD en el MLP;
  - accuracy vs. MACs de las convoluciones factorizadas.
Las tres últimas leen los resultados de los ejemplos:
  examples/factorization/outputs/svd_resultados.json   (script01.py)
  examples/factorization/outputs/convs_resultados.json (script02.py)

Uso (desde presentations/):
    uv run --no-project --with matplotlib --with numpy --with pillow python scripts/factorizacion_figuras.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIGURAS = ROOT / "presentations" / "figures"
SALIDAS = ROOT / "examples" / "factorization" / "outputs"
FOTO = ROOT / "examples" / "esp32-pipelines" / "demo" / "imagenes" / "gato.png"

BG = "#282a36"
GRID = "#44475a"
FG = "white"
VERDE, NARANJA, CIAN, ROSA, VIOLETA = "#50fa7b", "#ffb86c", "#8be9fd", "#ff79c6", "#bd93f9"


def ejes(ax, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.tick_params(colors=FG, labelsize=10)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_xlabel(xlabel, color=FG, fontsize=12)
    ax.set_ylabel(ylabel, color=FG, fontsize=12)


def leyenda(ax, **kw):
    ax.legend(fontsize=9, facecolor=GRID, edgecolor=GRID, labelcolor=FG, **kw)


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(FIGURAS / nombre, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"Figura guardada en: presentations/figures/{nombre}")


def figura_svd_imagen():
    """La foto en grises reconstruida con los k valores singulares más grandes."""
    X = np.asarray(Image.open(FOTO).convert("L"), dtype=float) / 255
    m, n = X.shape
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), facecolor=BG)
    for ax, k in zip(axes, [2, 8, 24, min(m, n)]):
        Xk = (U[:, :k] * S[:k]) @ Vt[:k, :]
        ax.imshow(np.clip(Xk, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if k == min(m, n):
            titulo = f"Original ({m}×{n})\n{m * n:,} valores".replace(",", " ")
        else:
            titulo = f"Rango {k}\nguarda el {k * (m + n) / (m * n):.0%} de los valores"
        ax.set_title(titulo.replace("%", " %"), color=FG, fontsize=11)
    guardar(fig, "factorizacion-svd_imagen.png")


def figura_espectro(res):
    """Energía acumulada de fc1 (784→64): pesos iniciales vs. entrenados."""
    r, limite = 16, res["rango_limite"]["fc1"]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=BG)
    for clave, etiqueta, color in (("entrenada", "Pesos entrenados", VERDE),
                                   ("inicial", "Pesos iniciales (aleatorios)", CIAN)):
        S = np.array(res["espectro"][clave])
        acumulada = np.cumsum(S ** 2) / np.sum(S ** 2)
        k = np.arange(1, len(S) + 1)
        ax.plot(k, acumulada * 100, color=color, linewidth=2.5, label=etiqueta)
        ax.annotate(f"{acumulada[r - 1]:.0%}".replace("%", " %"), (r, acumulada[r - 1] * 100),
                    textcoords="offset points", xytext=(8, -14), color=color, fontsize=11)
    ax.axvline(r, color=NARANJA, linestyle="--", linewidth=1.5, label=f"r = {r}")
    ax.axvline(limite, color=ROSA, linestyle=":", linewidth=1.5, label=f"Rango límite ≈ {limite:.0f}")
    ejes(ax, "Cantidad de valores singulares que se conservan (k)", "Energía acumulada (%)")
    ax.set_title("fc1, Linear(784→64): energía (Σσ²) de los k más grandes", color=FG, fontsize=12)
    leyenda(ax, loc="lower right")
    guardar(fig, "factorizacion-espectro.png")


def figura_svd_mlp(res):
    """Accuracy vs. rango y MACs/parámetros como % del baseline."""
    rangos = sorted(int(r) for r in res["rangos"])
    d = [res["rangos"][str(r)] for r in rangos]
    base = res["baseline"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=BG)

    a1.axhline(base["acc_ft"] * 100, color=CIAN, linestyle="--", linewidth=1.5,
               label=f"Baseline + FT ({base['acc_ft']:.1%})".replace("%", " %"))
    a1.plot(rangos, [x["acc_sin_ft"] * 100 for x in d], "o--", color=NARANJA, linewidth=2, label="SVD sin FT")
    a1.plot(rangos, [x["acc_ft"] * 100 for x in d], "s-", color=VERDE, linewidth=2.5, label="SVD con FT")
    ejes(a1, "Rango r", "Accuracy (%)")
    a1.set_xticks(rangos)
    a1.set_title("Accuracy", color=FG, fontsize=12)
    leyenda(a1, loc="lower right")

    x = np.arange(len(rangos))
    macs = [v["macs"] / base["macs"] * 100 for v in d]
    params = [v["params"] / base["params"] * 100 for v in d]
    a2.bar(x - 0.2, macs, 0.4, color=VIOLETA, label="MACs")
    a2.bar(x + 0.2, params, 0.4, color=ROSA, label="Parámetros")
    for xi, v in zip(x, macs):
        a2.text(xi - 0.2, v + 1.5, f"{v:.0f} %", color=FG, ha="center", fontsize=9)
    ejes(a2, "Rango r", "% del baseline")
    a2.set_xticks(x, [str(r) for r in rangos])
    a2.set_ylim(0, 100)
    a2.set_title("MACs y parámetros (dos capas)", color=FG, fontsize=12)
    leyenda(a2, loc="upper left")
    guardar(fig, "factorizacion-svd_mlp.png")


def figura_convs(res):
    """Accuracy vs. MACs de cada variante de las convoluciones."""
    colores = {"—": CIAN, "después de entrenar": NARANJA, "diseño": VERDE}
    etiquetas = {"—": "Estándar", "después de entrenar": "Tucker-2: después de entrenar",
                 "diseño": "Al diseñar"}
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=BG)
    vistos = set()
    for m in res["modelos"]:
        c = m["cuando"]
        ax.scatter(m["macs"] / 1e6, m["acc"] * 100, color=colores[c], s=70, zorder=3,
                   label=None if c in vistos else etiquetas[c])
        vistos.add(c)
        ax.annotate(m["nombre"].replace("%)", " %)"), (m["macs"] / 1e6, m["acc"] * 100), textcoords="offset points",
                    xytext=(7, 5), color=FG, fontsize=9)
    ejes(ax, "MACs (millones)", "Accuracy (%)")
    ax.set_xlim(0, 2.25)
    ax.set_title("Convoluciones: accuracy vs. MACs", color=FG, fontsize=12)
    leyenda(ax, loc="lower right")
    guardar(fig, "factorizacion-convs.png")


def main():
    svd = json.loads((SALIDAS / "svd_resultados.json").read_text())
    convs = json.loads((SALIDAS / "convs_resultados.json").read_text())
    figura_svd_imagen()
    figura_espectro(svd)
    figura_svd_mlp(svd)
    figura_convs(convs)


if __name__ == "__main__":
    main()
