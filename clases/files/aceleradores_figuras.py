"""Figuras de la sección extra "Aceleradores en FPGA y ASIC".

Genera, con el estilo oscuro de las diapos:
  - costo por unidad vs. volumen para FPGA y ASIC (valores ilustrativos);
  - un systolic array de 2×2 (output-stationary) multiplicando dos matrices
    de 2×2, un cuadro por ciclo de reloj.

Uso (desde presentations/):
    uv run --no-project --with matplotlib --with numpy python scripts/aceleradores_figuras.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
FIGURAS = ROOT / "presentations" / "figures"

BG = "#282a36"
GRID = "#44475a"
FG = "white"
VERDE, NARANJA, CIAN, ROSA, VIOLETA = "#50fa7b", "#ffb86c", "#8be9fd", "#ff79c6", "#bd93f9"


def guardar(fig, nombre):
    fig.savefig(FIGURAS / nombre, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"Figura guardada en: presentations/figures/{nombre}")


def costo_volumen():
    # Valores ilustrativos: solo importa la forma de las curvas
    nre = {"FPGA": 0.1e6, "ASIC": 5e6}
    unidad = {"FPGA": 100.0, "ASIC": 5.0}
    n = np.logspace(2, 7, 400)

    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=BG)
    ax.set_facecolor(BG)
    for nombre, color in [("FPGA", CIAN), ("ASIC", NARANJA)]:
        ax.plot(n, nre[nombre] / n + unidad[nombre], color=color, linewidth=2.5,
                label=f"{nombre}: NRE {nre[nombre] / 1e6:g} M USD + {unidad[nombre]:g} USD/unidad")
    cruce = (nre["ASIC"] - nre["FPGA"]) / (unidad["FPGA"] - unidad["ASIC"])
    ax.axvline(cruce, color=ROSA, linestyle="--", linewidth=1.2)
    ax.text(cruce * 1.15, 2e3, f"cruce ≈ {cruce / 1e3:.0f} mil unidades", color=ROSA, fontsize=11)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
    ax.set_xlabel("Unidades fabricadas (escala log)", color=FG, fontsize=12)
    ax.set_ylabel("Costo por unidad, USD (escala log)", color=FG, fontsize=12)
    ax.tick_params(colors=FG, labelsize=10)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.legend(loc="upper right", fontsize=10, facecolor=GRID, edgecolor=GRID, labelcolor=FG)
    ax.set_title("Valores ilustrativos", color=FG, fontsize=11, alpha=0.7)
    fig.tight_layout()
    guardar(fig, "aceleradores-costo_volumen.png")


def systolic():
    """C = A·B en un array de 2×2 output-stationary.

    El PE (i, j) acumula C[i][j]. En el ciclo t recibe A[i][k] desde la
    izquierda y B[k][j] desde arriba, con k = t - i - j: las entradas entran
    escalonadas (skew) para que cada par llegue al mismo PE en el mismo ciclo.
    """
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    N = K = 2
    ciclos = 2 * N + K - 2                      # 4 ciclos: t = 0..3
    acum = np.zeros((N, N), dtype=int)

    for t in range(ciclos):
        fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=BG)
        ax.set_facecolor(BG)
        ax.set_xlim(-3.6, 4.6)
        ax.set_ylim(-1.2, 3.4)
        ax.set_aspect("equal")
        ax.axis("off")

        # posiciones: PE (i, j) en (x = j*1.6, y = 1.6 - i*1.6)
        def pos(i, j):
            return j * 1.6, 1.6 - i * 1.6

        # flechas entre PEs
        for i in range(N):
            for j in range(N):
                x, y = pos(i, j)
                if j < N - 1:
                    ax.annotate("", xy=(x + 1.0, y), xytext=(x + 0.6, y),
                                arrowprops=dict(arrowstyle="->", color=GRID, lw=1.5))
                if i < N - 1:
                    ax.annotate("", xy=(x, y - 1.0), xytext=(x, y - 0.6),
                                arrowprops=dict(arrowstyle="->", color=GRID, lw=1.5))

        for i in range(N):
            for j in range(N):
                x, y = pos(i, j)
                k = t - i - j
                activo = 0 <= k < K
                if activo:
                    acum[i, j] += A[i, k] * B[k, j]
                ax.add_patch(FancyBboxPatch((x - 0.55, y - 0.55), 1.1, 1.1,
                                            boxstyle="round,pad=0.02",
                                            facecolor="#3a3d4f" if activo else BG,
                                            edgecolor=VERDE if activo else GRID, linewidth=2))
                ax.text(x, y + 0.3, f"PE {i}{j}", color=FG, fontsize=9, ha="center", alpha=0.7)
                ax.text(x, y - 0.12, f"{acum[i, j]}", color=VERDE, fontsize=16, ha="center",
                        fontweight="bold")
                if activo:
                    ax.text(x, y - 0.42, f"+{A[i, k]}·{B[k, j]}", color=FG, fontsize=9, ha="center")

        # A entrando por la izquierda (fila i, escalonada i ciclos)
        for i in range(N):
            _, y = pos(i, 0)
            for k in range(K):
                llegada = i + k                  # ciclo en que A[i][k] entra al PE (i, 0)
                d = llegada - t                  # cuántos lugares le faltan
                if d > 0:
                    ax.text(-0.9 - 0.7 * (d - 1), y, f"{A[i, k]}", color=CIAN, fontsize=14,
                            ha="center", va="center", fontweight="bold")
        # B entrando por arriba (columna j, escalonada j ciclos)
        for j in range(N):
            x, _ = pos(0, j)
            for k in range(K):
                d = j + k - t
                if d > 0:
                    ax.text(x, 2.5 + 0.55 * (d - 1), f"{B[k, j]}", color=NARANJA, fontsize=14,
                            ha="center", va="center", fontweight="bold")

        ax.text(-3.5, 2.95, f"Ciclo {t + 1} de {ciclos}", color=FG, fontsize=13, fontweight="bold")
        ax.text(-3.5, -0.9, "A entra por la izquierda", color=CIAN, fontsize=10)
        ax.text(-3.5, -1.15, "B entra por arriba", color=NARANJA, fontsize=10)
        ax.text(2.6, 0.8, "A = [[1, 2], [3, 4]]\nB = [[5, 6], [7, 8]]\nC = A·B", color=FG,
                fontsize=10, va="center", family="monospace")
        if t == ciclos - 1:
            ax.text(2.6, -0.3, "C = [[19, 22],\n     [43, 50]]", color=VERDE, fontsize=10,
                    va="center", family="monospace")
        fig.tight_layout()
        guardar(fig, f"aceleradores-systolic_{t + 1}.png")

    assert (acum == A @ B).all()


if __name__ == "__main__":
    costo_volumen()
    systolic()
