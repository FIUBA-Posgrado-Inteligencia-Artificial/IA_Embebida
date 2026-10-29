"""Figuras de la sección "Cuantización".

Genera, con el estilo oscuro de las diapos:
  - función de cuantización uniforme y no uniforme;
  - straight-through estimator (STE): forward vs. lo que usa el backward;
  - soft quantization (aproximación suave con distintas temperaturas);
  - resultados del ejemplo de 3 pasos a 8 y a 4 bits (PTQ y QAT), leídos de
    examples/quantization/outputs/cnn_barrido_resultados.json (lo genera
    script03.py).

Uso (desde presentations/):
    uv run --no-project --with matplotlib --with numpy python scripts/cuantizacion_figuras.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGURAS = ROOT / "presentations" / "figures"
BARRIDO = ROOT / "examples" / "quantization" / "outputs" / "cnn_barrido_resultados.json"

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


def tres_pasos():
    d = json.loads(BARRIDO.read_text())
    pasos = [("paso1", "Paso 1\nuniforme"), ("paso2", "Paso 2\nbits enteros\npor capa (p95)"),
             ("paso3", "Paso 3\nbits por capa\n(N b extremos, N/2 resto)")]
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
    fig.savefig(FIGURAS / "cuantizacion-3pasos_8vs4.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("Figura guardada en: presentations/figures/cuantizacion-3pasos_8vs4.png")


if __name__ == "__main__":
    funcion_uniforme()
    funcion_no_uniforme()
    ste()
    soft()
    tres_pasos()
