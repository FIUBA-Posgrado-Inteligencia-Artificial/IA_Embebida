"""Fronteras de Pareto de los modelos de examples/esp32-pipelines.

Lee metricas_modelos.csv (lo genera tools/export_metricas.py en ese ejemplo) y
genera dos figuras, accuracy vs. MACs y accuracy vs. parámetros (cada modelo
entrenado es un punto, con la frontera de Pareto y los dos modelos de la demo
resaltados), y una con los tres objetivos a la vez (3D y mapa de calor). No
dice qué técnicas se aplicaron a cada uno: en la diapo de la clase 1 la idea es
solo mostrar que hay muchas combinaciones.

Uso (desde presentations/):
    uv run --no-project --with matplotlib --with numpy python scripts/pareto_modelos.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "examples" / "esp32-pipelines" / "metricas_modelos.csv"
FIGURAS = ROOT / "presentations" / "figures"

DEMO = {
    "std_rgb96_pra20_ft10": ("grande", "#ffb86c"),
    "dw_rgb32_pra50_ft10": ("chico", "#8be9fd"),
}

# (columna del CSV, divisor, etiqueta del eje x, archivo de salida)
EJES = [
    ("macs", 1e6, "MACs por imagen (millones, escala log)", "demo_detalle-pareto-macs.png"),
    ("parametros", 1e3, "Parámetros (miles, escala log)", "demo_detalle-pareto-params.png"),
]

BG = "#282a36"


def frontera(puntos):
    """Puntos no dominados: nadie tiene menor costo y más accuracy a la vez."""
    out, mejor = [], -1.0
    for p in sorted(puntos, key=lambda p: (p["x"], -p["acc"])):
        if p["acc"] > mejor:
            out.append(p)
            mejor = p["acc"]
    return out


def graficar(filas, columna, divisor, xlabel, salida) -> None:
    puntos = [{"nombre": r["modelo"], "acc": float(r["accuracy"]) * 100,
               "x": int(r[columna]) / divisor} for r in filas]
    front = frontera(puntos)

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=BG)
    ax.set_facecolor(BG)

    ax.scatter([p["x"] for p in puntos], [p["acc"] for p in puntos],
               s=14, color="#6272a4", alpha=0.6, label=f"Modelos ({len(puntos)})", zorder=1)
    ax.plot([p["x"] for p in front], [p["acc"] for p in front],
            color="#50fa7b", linewidth=2, drawstyle="steps-post", zorder=2)
    ax.scatter([p["x"] for p in front], [p["acc"] for p in front],
               s=30, color="#50fa7b", label="Frontera de Pareto", zorder=3)

    for p in puntos:
        if p["nombre"] in DEMO:
            etiqueta, color = DEMO[p["nombre"]]
            ax.scatter(p["x"], p["acc"], s=160, marker="*", color=color,
                       edgecolor="white", linewidth=0.8, zorder=4)
            ax.annotate(f"{etiqueta} (demo)", (p["x"], p["acc"]),
                        textcoords="offset points", xytext=(8, -16),
                        color=color, fontsize=12, fontweight="bold")

    ax.axhline(10, color="#ff5555", linewidth=1, linestyle="--", alpha=0.7)
    ax.text(0.02, 12, "azar (10 %)", color="#ff5555", fontsize=10,
            transform=ax.get_yaxis_transform())

    ax.set_xscale("log")
    ax.set_xlabel(xlabel, color="white", fontsize=12)
    ax.set_ylabel("Accuracy (%)", color="white", fontsize=12)
    ax.set_ylim(0, 70)
    ax.tick_params(colors="white", labelsize=10)
    ax.grid(True, color="#44475a", linewidth=0.6, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#44475a")
    ax.legend(loc="lower right", fontsize=10, facecolor="#44475a",
              edgecolor="#6272a4", labelcolor="white")

    fig.tight_layout()
    fig.savefig(salida, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"Figura guardada en: {salida.relative_to(ROOT)}")


def frontera_3d(puntos):
    """Puntos no dominados con 3 objetivos: menos MACs, menos parámetros, más accuracy."""
    def domina(q, p):
        return (q["macs"] <= p["macs"] and q["params"] <= p["params"] and q["acc"] >= p["acc"]
                and (q["macs"], q["params"], q["acc"]) != (p["macs"], p["params"], p["acc"]))
    return [p for p in puntos if not any(domina(q, p) for q in puntos)]


def ejes_log(ax, eje, valores):
    """Ticks en potencias de 10 para un eje que ya está en log10."""
    exps = range(int(np.floor(min(valores))), int(np.ceil(max(valores))) + 1)
    getattr(ax, f"set_{eje}ticks")(list(exps))
    getattr(ax, f"set_{eje}ticklabels")([f"{10.0 ** e:g}" for e in exps])


def graficar_3_objetivos(filas, salida) -> None:
    """Tres objetivos a la vez: 3D a la izquierda, mapa de calor (color = accuracy) a la derecha."""
    puntos = [{"nombre": r["modelo"], "acc": float(r["accuracy"]) * 100,
               "macs": int(r["macs"]) / 1e6, "params": int(r["parametros"]) / 1e3} for r in filas]
    front = frontera_3d(puntos)
    en_front = {id(p) for p in front}
    resto = [p for p in puntos if id(p) not in en_front]
    lm = lambda ps: np.log10([p["macs"] for p in ps])
    lp = lambda ps: np.log10([p["params"] for p in ps])
    acc = lambda ps: [p["acc"] for p in ps]

    fig = plt.figure(figsize=(13, 5.4), facecolor=BG)

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_facecolor(BG)
    ax.scatter(lm(resto), lp(resto), acc(resto), s=10, color="#6272a4", alpha=0.5,
               label=f"Dominados ({len(resto)})")
    ax.scatter(lm(front), lp(front), acc(front), s=30, color="#50fa7b",
               label=f"Frontera de Pareto ({len(front)})")
    for p in puntos:
        if p["nombre"] in DEMO:
            etiqueta, color = DEMO[p["nombre"]]
            ax.scatter(np.log10(p["macs"]), np.log10(p["params"]), p["acc"], s=160, marker="*",
                       color=color, edgecolor="white", linewidth=0.8)
            ax.text(np.log10(p["macs"]), np.log10(p["params"]), p["acc"] - 9, etiqueta,
                    color=color, fontsize=11, fontweight="bold")
    ejes_log(ax, "x", lm(puntos))
    ejes_log(ax, "y", lp(puntos))
    ax.set_xlabel("MACs (millones)", color="white", fontsize=10)
    ax.set_ylabel("Parámetros (miles)", color="white", fontsize=10)
    ax.set_zlabel("Accuracy (%)", color="white", fontsize=10)
    ax.set_zlim(0, 70)
    ax.view_init(elev=22, azim=-55)
    ax.tick_params(colors="white", labelsize=8)
    for eje in (ax.xaxis, ax.yaxis, ax.zaxis):
        eje.set_pane_color((0.27, 0.28, 0.35, 0.35))
        eje._axinfo["grid"]["color"] = "#44475a"
    ax.legend(loc="upper left", fontsize=9, facecolor="#44475a", edgecolor="#6272a4",
              labelcolor="white")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor(BG)
    orden = sorted(puntos, key=lambda p: p["acc"])      # los mejores arriba
    sc = ax2.scatter([p["macs"] for p in orden], [p["params"] for p in orden],
                     c=acc(orden), cmap="plasma", vmin=30, vmax=65, s=22, zorder=2)
    ax2.scatter([p["macs"] for p in front], [p["params"] for p in front], s=70,
                facecolors="none", edgecolors="#50fa7b", linewidth=1.3, zorder=3,
                label="Frontera de Pareto")
    for p in puntos:
        if p["nombre"] in DEMO:
            etiqueta, color = DEMO[p["nombre"]]
            ax2.scatter(p["macs"], p["params"], s=160, marker="*", color=color,
                        edgecolor="white", linewidth=0.8, zorder=4)
            ax2.annotate(etiqueta, (p["macs"], p["params"]), textcoords="offset points",
                         xytext=(10, -16) if etiqueta == "chico" else (-58, 8), color=color, fontsize=11, fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    for eje in (ax2.xaxis, ax2.yaxis):
        eje.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax2.set_xlabel("MACs por imagen (millones, escala log)", color="white", fontsize=11)
    ax2.set_ylabel("Parámetros (miles, escala log)", color="white", fontsize=11)
    ax2.tick_params(colors="white", labelsize=9)
    ax2.grid(True, color="#44475a", linewidth=0.6, alpha=0.7)
    for spine in ax2.spines.values():
        spine.set_color("#44475a")
    ax2.legend(loc="lower right", fontsize=9, facecolor="#44475a", edgecolor="#6272a4",
               labelcolor="white")
    cb = fig.colorbar(sc, ax=ax2, pad=0.02, extend="min")
    cb.set_label("Accuracy (%)", color="white", fontsize=11)
    cb.ax.tick_params(colors="white", labelsize=9)
    cb.outline.set_edgecolor("#44475a")

    fig.tight_layout()
    fig.savefig(salida, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"Figura guardada en: {salida.relative_to(ROOT)} ({len(front)} en la frontera)")


def main() -> None:
    with CSV.open() as f:
        filas = list(csv.DictReader(f))
    for columna, divisor, xlabel, archivo in EJES:
        graficar(filas, columna, divisor, xlabel, FIGURAS / archivo)
    graficar_3_objetivos(filas, FIGURAS / "demo_detalle-pareto-3d.png")


if __name__ == "__main__":
    main()
