"""Figuras de la sección "Cuantización".

Genera, con el estilo oscuro de las diapos:
  - función de cuantización uniforme y no uniforme;
  - straight-through estimator (STE): forward vs. lo que usa el backward;
  - soft quantization (aproximación suave con distintas temperaturas);
  - resultados del ejemplo con las tres formas de cuantizar:
      · script02.py a 8 bits (PTQ), leídos de
        examples/quantization/outputs/cnn_3pasos_resultados.json;
      · script03.py de 2 a 8 bits, y a 8 y a 4 bits (PTQ y QAT), leídos de
        examples/quantization/outputs/cnn_barrido_resultados.json.

Uso (desde presentations/):
    uv run --no-project --with matplotlib --with numpy python scripts/cuantizacion_figuras.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGURAS = ROOT / "presentations" / "figures"
SALIDAS = ROOT / "examples" / "quantization" / "outputs"
BARRIDO = SALIDAS / "cnn_barrido_resultados.json"
FIJO = SALIDAS / "cnn_3pasos_resultados.json"

BG = "#282a36"
GRID = "#44475a"
FG = "white"
VERDE, NARANJA, CIAN, ROSA, VIOLETA = "#50fa7b", "#ffb86c", "#8be9fd", "#ff79c6", "#bd93f9"


def ejes(ax, xlabel="x", ylabel="Q(x)", ceros=True):
    ax.set_facecolor(BG)
    if ceros:
        ax.axhline(0, color=GRID, linewidth=1)
        ax.axvline(0, color=GRID, linewidth=1)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.tick_params(colors=FG, labelsize=10)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_xlabel(xlabel, color=FG, fontsize=12)
    ax.set_ylabel(ylabel, color=FG, fontsize=12)


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(FIGURAS / nombre, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"Figura guardada en: presentations/figures/{nombre}")


def escalera(ax, umbrales, niveles, color):
    """Dibuja una función constante por tramos (umbrales crecientes)."""
    xs = np.concatenate([[-4.2], umbrales, [4.2]])
    for i, y in enumerate(niveles):
        ax.plot([xs[i], xs[i + 1]], [y, y], color=color, linewidth=2.5)
        if i > 0:
            ax.plot([xs[i], xs[i]], [niveles[i - 1], y], color=color, linewidth=1, linestyle=":")


def funcion_uniforme():
    fig, ax = plt.subplots(figsize=(5, 3.6), facecolor=BG)
    ejes(ax)
    d = 1.0
    niveles = np.arange(-4, 5) * d
    umbrales = (niveles[:-1] + niveles[1:]) / 2
    escalera(ax, umbrales, niveles, VERDE)
    ax.plot([-4.2, 4.2], [-4.2, 4.2], color=FG, linewidth=1, linestyle="--", alpha=0.4)
    ax.annotate("", xy=(2.5, 2), xytext=(1.5, 2),
                arrowprops=dict(arrowstyle="<->", color=NARANJA))
    ax.text(2.0, 2.25, "Δ", color=NARANJA, fontsize=13, ha="center")
    ax.annotate("", xy=(3.1, 3), xytext=(3.1, 2),
                arrowprops=dict(arrowstyle="<->", color=NARANJA))
    ax.text(3.3, 2.4, "Δ", color=NARANJA, fontsize=13)
    ax.set_title("Uniforme: Q(x) = round(x / Δ) · Δ", color=FG, fontsize=12)
    guardar(fig, "cuantizacion-funcion_uniforme.png")


def funcion_no_uniforme():
    fig, ax = plt.subplots(figsize=(5, 3.6), facecolor=BG)
    ejes(ax)
    pos = np.array([0.25, 0.75, 1.75, 3.5])        # niveles más densos cerca de 0
    niveles = np.concatenate([-pos[::-1], pos])
    umbrales = (niveles[:-1] + niveles[1:]) / 2
    escalera(ax, umbrales, niveles, ROSA)
    ax.plot([-4.2, 4.2], [-4.2, 4.2], color=FG, linewidth=1, linestyle="--", alpha=0.4)
    ax.set_title("No uniforme: niveles más densos cerca de 0", color=FG, fontsize=12)
    guardar(fig, "cuantizacion-funcion_no_uniforme.png")


def ste():
    x = np.linspace(-3, 3, 2000)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=BG)
    ejes(a1, "x", "salida")
    a1.plot(x, np.round(x), color=VERDE, linewidth=2.5, label="forward: Q(x) = round(x)")
    a1.plot(x, x, color=NARANJA, linewidth=2, linestyle="--", label="backward (STE): como si fuera x")
    a1.set_title("Forward vs. backward", color=FG, fontsize=12)
    a1.legend(loc="upper left", fontsize=9, facecolor=GRID, edgecolor=GRID, labelcolor=FG)

    ejes(a2, "x", "derivada")
    a2.plot(x, np.zeros_like(x), color=VERDE, linewidth=2.5, label="real: 0 (o no definida en los saltos)")
    for u in np.arange(-2.5, 3, 1):
        a2.axvline(u, color=VERDE, linewidth=1, linestyle=":", alpha=0.8)
    a2.plot(x, np.ones_like(x), color=NARANJA, linewidth=2, linestyle="--", label="STE: 1")
    a2.set_ylim(-0.5, 1.5)
    a2.set_title("Derivada", color=FG, fontsize=12)
    a2.legend(loc="center left", fontsize=9, facecolor=GRID, edgecolor=GRID, labelcolor=FG)
    guardar(fig, "cuantizacion-ste.png")


def soft():
    x = np.linspace(-2.5, 2.5, 2000)
    fig, ax = plt.subplots(figsize=(5, 3.4), facecolor=BG)
    ejes(ax)
    ax.plot(x, np.round(x), color=FG, linewidth=1.5, alpha=0.5, label="round(x)")
    centros = np.arange(-1.5, 2, 1)
    for t, c, nombre in [(0.3, CIAN, "T = 0.3"), (0.1, VIOLETA, "T = 0.1"), (0.03, ROSA, "T = 0.03")]:
        y = -2 + sum(1 / (1 + np.exp(-(x - u) / t)) for u in centros)
        ax.plot(x, y, color=c, linewidth=2, label=nombre)
    ax.set_title("Soft quantization: suma de sigmoides", color=FG, fontsize=12)
    ax.legend(loc="upper left", fontsize=9, facecolor=GRID, edgecolor=GRID, labelcolor=FG)
    guardar(fig, "cuantizacion-soft.png")


def script02_8bits():
    d = json.loads(FIJO.read_text())
    nombres = {"Paso1-Uniforme": "Uniforme",
               "Paso2a-Mixta/max": "Bits enteros\npor capa\n(máx.)",
               "Paso2b-Mixta/p95": "Bits enteros\npor capa\n(p95)",
               "Paso2c-Mixta/p75": "Bits enteros\npor capa\n(p75)",
               "Paso3-Completa/p95": "Bits totales\npor capa\n(8 / 4, p95)"}
    colores = [CIAN, VERDE, VERDE, VERDE, VIOLETA]
    pasos = d["steps"]
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=BG)
    ejes(ax, "", "Accuracy PTQ (%)", ceros=False)
    x = np.arange(len(pasos))
    barras = ax.bar(x, [p["ptq_acc"] * 100 for p in pasos], 0.6, color=colores)
    for b, p in zip(barras, pasos):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() - 0.5,
                f"{b.get_height():.1f} %\n{p['mem_bytes'] / 1024:.0f} KB", ha="center", va="top",
                color=BG, fontsize=10, fontweight="bold")
    ax.axhline(d["float_acc"] * 100, color=ROSA, linestyle="--", linewidth=1.2,
               label=f"FP32 ({d['float_acc'] * 100:.1f} %, {d['fp32_mem_kb']:.0f} KB)")
    ax.set_xticks(x, [nombres[p["tag"]] for p in pasos], color=FG, fontsize=9)
    ax.set_ylim(75, 92)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, facecolor=GRID, edgecolor=GRID, labelcolor=FG)
    guardar(fig, "cuantizacion-script02_8bits.png")


def barrido():
    d = json.loads(BARRIDO.read_text())
    formas = [("paso1", "Uniforme", CIAN), ("paso2", "Bits enteros por capa (p95)", VERDE),
              ("paso3", "Bits totales por capa (p95)", VIOLETA)]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), facecolor=BG, sharey=True)
    for ax, (clave, nombre, color) in zip(axs, formas):
        ejes(ax, "Bits", "Accuracy (%)" if clave == "paso1" else "", ceros=False)
        r = d["results"][clave]
        bits = sorted(int(b) for b in r)
        ax.plot(bits, [r[str(b)]["ptq_acc"] * 100 for b in bits], "x--", color=color, alpha=0.7,
                linewidth=1.8, markersize=8, label="PTQ")
        ax.plot(bits, [r[str(b)]["qat_acc"] * 100 for b in bits], "o-", color=color, linewidth=2.5,
                markersize=7, label="QAT")
        ax.axhline(d["float_acc"] * 100, color=ROSA, linestyle=":", linewidth=1.5, label="FP32")
        ax.set_xticks(bits)
        ax.set_ylim(0, 100)
        ax.set_title(nombre, color=FG, fontsize=12)
        ax.legend(loc="lower right", fontsize=9, facecolor=GRID, edgecolor=GRID, labelcolor=FG)
    guardar(fig, "cuantizacion-barrido.png")


def formas_8vs4():
    d = json.loads(BARRIDO.read_text())
    pasos = [("paso1", "Uniforme"), ("paso2", "Bits enteros\npor capa (p95)"),
             ("paso3", "Bits totales\npor capa\n(N extremos, N/2 resto)")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=BG, sharey=True)
    x = np.arange(len(pasos))
    for ax, bits in zip(axs, ["8", "4"]):
        ejes(ax, "", "Accuracy (%)" if bits == "8" else "", ceros=False)
        ptq = [d["results"][p][bits]["ptq_acc"] * 100 for p, _ in pasos]
        qat = [d["results"][p][bits]["qat_acc"] * 100 for p, _ in pasos]
        b1 = ax.bar(x - 0.2, ptq, 0.4, color=CIAN, label="PTQ")
        b2 = ax.bar(x + 0.2, qat, 0.4, color=NARANJA, label="QAT")
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.1f}",
                    ha="center", color=FG, fontsize=9)
        ax.axhline(d["float_acc"] * 100, color=ROSA, linestyle="--", linewidth=1.2,
                   label=f"FP32 ({d['float_acc'] * 100:.1f} %)")
        ax.set_xticks(x, [n for _, n in pasos], color=FG, fontsize=9)
        ax.set_ylim(60, 95)
        ax.set_title(f"{bits} bits", color=FG, fontsize=13)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=10, facecolor=GRID, edgecolor=GRID,
               labelcolor=FG)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGURAS / "cuantizacion-formas_8vs4.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("Figura guardada en: presentations/figures/cuantizacion-formas_8vs4.png")


if __name__ == "__main__":
    funcion_uniforme()
    funcion_no_uniforme()
    ste()
    soft()
    script02_8bits()
    barrido()
    formas_8vs4()
