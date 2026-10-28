"""Fronteras de Pareto de los modelos de examples/esp32-pipelines.

Lee metricas_modelos.csv (lo genera tools/export_metricas.py en ese ejemplo) y
genera dos figuras, accuracy vs. MACs y accuracy vs. parámetros: cada modelo
entrenado es un punto, con la frontera de Pareto y los dos modelos de la demo
resaltados. No dice qué técnicas se aplicaron a cada uno: en la diapo de la
clase 1 la idea es solo mostrar que hay muchas combinaciones.

Uso (desde presentations/):
    uv run --with matplotlib python scripts/pareto_modelos.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

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


def main() -> None:
    with CSV.open() as f:
        filas = list(csv.DictReader(f))
    for columna, divisor, xlabel, archivo in EJES:
        graficar(filas, columna, divisor, xlabel, FIGURAS / archivo)


if __name__ == "__main__":
    main()
