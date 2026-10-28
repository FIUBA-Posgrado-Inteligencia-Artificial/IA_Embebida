"""Frontera de Pareto accuracy vs. MACs de los modelos de examples/esp32-pipelines.

Lee metricas_modelos.csv (lo genera tools/export_metricas.py en ese ejemplo) y
grafica cada modelo entrenado como un punto, resaltando la frontera de Pareto y
los dos modelos de la demo. No dice qué técnicas se aplicaron a cada uno: en la
diapo de la clase 1 la idea es solo mostrar que hay muchas combinaciones.

Uso (desde presentations/):
    uv run --with matplotlib python scripts/pareto_modelos.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "examples" / "esp32-pipelines" / "metricas_modelos.csv"
SALIDA = ROOT / "presentations" / "figures" / "demo_detalle-pareto.png"

DEMO = {
    "std_rgb96_pra20_ft10": ("grande", "#ffb86c"),
    "dw_rgb32_pra50_ft10": ("chico", "#8be9fd"),
}

BG = "#282a36"


def frontera(puntos):
    """Puntos no dominados: nadie tiene menos MACs y más accuracy a la vez."""
    out, mejor = [], -1.0
    for p in sorted(puntos, key=lambda p: (p["macs"], -p["acc"])):
        if p["acc"] > mejor:
            out.append(p)
            mejor = p["acc"]
    return out


def main() -> None:
    with CSV.open() as f:
        puntos = [{"nombre": r["modelo"], "acc": float(r["accuracy"]) * 100,
                   "macs": int(r["macs"]) / 1e6} for r in csv.DictReader(f)]
    front = frontera(puntos)

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=BG)
    ax.set_facecolor(BG)

    ax.scatter([p["macs"] for p in puntos], [p["acc"] for p in puntos],
               s=14, color="#6272a4", alpha=0.6, label=f"Modelos ({len(puntos)})", zorder=1)
    ax.plot([p["macs"] for p in front], [p["acc"] for p in front],
            color="#50fa7b", linewidth=2, drawstyle="steps-post", zorder=2)
    ax.scatter([p["macs"] for p in front], [p["acc"] for p in front],
               s=30, color="#50fa7b", label="Frontera de Pareto", zorder=3)

    for p in puntos:
        if p["nombre"] in DEMO:
            etiqueta, color = DEMO[p["nombre"]]
            ax.scatter(p["macs"], p["acc"], s=160, marker="*", color=color,
                       edgecolor="white", linewidth=0.8, zorder=4)
            ax.annotate(f"{etiqueta} (demo)", (p["macs"], p["acc"]),
                        textcoords="offset points", xytext=(8, -16),
                        color=color, fontsize=12, fontweight="bold")

    ax.axhline(10, color="#ff5555", linewidth=1, linestyle="--", alpha=0.7)
    ax.text(0.12, 12, "azar (10 %)", color="#ff5555", fontsize=10)

    ax.set_xscale("log")
    ax.set_xlabel("MACs por imagen (millones, escala log)", color="white", fontsize=12)
    ax.set_ylabel("Accuracy (%)", color="white", fontsize=12)
    ax.set_ylim(0, 70)
    ax.tick_params(colors="white", labelsize=10)
    ax.grid(True, color="#44475a", linewidth=0.6, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#44475a")
    ax.legend(loc="lower right", fontsize=10, facecolor="#44475a",
              edgecolor="#6272a4", labelcolor="white")

    fig.tight_layout()
    fig.savefig(SALIDA, dpi=150, facecolor=BG)
    print(f"Figura guardada en: {SALIDA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
