"""
Sección 10 — Neural Architecture Search (NAS): Búsqueda en Grilla
==================================================================
⚠️  ENFOQUE INGENUO / BRUTE FORCE
Este script recorre exhaustivamente un espacio de arquitecturas entrenando cada
candidato desde cero. Su propósito es mostrar POR QUÉ esto no escala en la
práctica y justificar el uso de algoritmos de búsqueda más inteligentes.
No es un ejemplo representativo de NAS moderno.

Con 10 candidatos × 5 epochs cada uno, el script ya tarda varios minutos.
Un espacio real tiene miles de candidatos y entrenamientos completos (cientos
de epochs): DARTS lo resuelve con una búsqueda por gradiente en un único ciclo.

Espacio de búsqueda: variaciones de la CNN LeNet-5 con distintos anchos de canal.
  conv1_filters ∈ {4, 8, 16}
  conv2_filters ∈ {8, 16, 32}
  dense_units   ∈ {32, 64, 128}

(subconjunto de ~10 candidatos representativos — ya es suficiente para ver el problema)

Alternativas modernas:
  - DARTS (script02.py): búsqueda por gradiente, O(1) pases en lugar de O(N).
  - OFA / Weight sharing: un único supernet en lugar de N entrenamientos.

Uso:
    cd examples/nas
    uv run python script01.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pytorch_utils import (
    count_macs,
    count_parameters,
    evaluate,
    get_dataloaders,
    get_device,
    model_size_kb,
    save_results,
    train,
)

OUTPUT_DIR     = "outputs"
EPOCHS_SEARCH  = 5    # epochs por candidato (rápido, solo para ranking)
BATCH_SIZE     = 256
LR             = 1e-3

# ---------------------------------------------------------------------------
# Arquitectura paramétrica
# ---------------------------------------------------------------------------

class CandidateCNN(nn.Module):
    """
    CNN LeNet-5-like con anchos configurables.
    conv1 → MaxPool → conv2 → MaxPool → Flatten → dense1 → dense2 → output.
    """
    def __init__(self, conv1_filters: int, conv2_filters: int, dense_units: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_filters, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5)
        self.pool  = nn.MaxPool2d(2)
        flat_dim   = conv2_filters * 4 * 4
        self.fc1   = nn.Linear(flat_dim, dense_units)
        self.fc2   = nn.Linear(dense_units, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Espacio de búsqueda
# ---------------------------------------------------------------------------

# Candidatos representativos: desde muy pequeño hasta del tamaño del baseline.
SEARCH_SPACE = [
    {"conv1_filters": 4,  "conv2_filters": 8,  "dense_units": 32},
    {"conv1_filters": 4,  "conv2_filters": 8,  "dense_units": 64},
    {"conv1_filters": 4,  "conv2_filters": 16, "dense_units": 32},
    {"conv1_filters": 8,  "conv2_filters": 8,  "dense_units": 32},
    {"conv1_filters": 8,  "conv2_filters": 16, "dense_units": 64},
    {"conv1_filters": 8,  "conv2_filters": 32, "dense_units": 64},
    {"conv1_filters": 8,  "conv2_filters": 16, "dense_units": 128},
    {"conv1_filters": 16, "conv2_filters": 16, "dense_units": 64},
    {"conv1_filters": 16, "conv2_filters": 32, "dense_units": 64},
    {"conv1_filters": 16, "conv2_filters": 32, "dense_units": 128},  # ~baseline
]


def config_label(cfg: dict) -> str:
    return f"c1={cfg['conv1_filters']} c2={cfg['conv2_filters']} d={cfg['dense_units']}"


# ---------------------------------------------------------------------------
# Frente de Pareto
# ---------------------------------------------------------------------------

def pareto_front(accs: list[float], macs: list[int]) -> list[int]:
    """
    Retorna los índices de los puntos en el frente de Pareto:
    maximizar accuracy, minimizar MACs.
    Un punto i es Pareto-óptimo si ningún j tiene acc[j]>=acc[i] AND macs[j]<=macs[i]
    con al menos una desigualdad estricta.
    """
    n = len(accs)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if accs[j] >= accs[i] and macs[j] <= macs[i]:
                if accs[j] > accs[i] or macs[j] < macs[i]:
                    dominated[i] = True
                    break
    return [i for i in range(n) if not dominated[i]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "⚠️ " * 20)
    print("ENFOQUE INGENUO: este script evalúa cada arquitectura por separado.")
    print(f"Con {len(SEARCH_SPACE)} candidatos × {EPOCHS_SEARCH} epochs, ya es costoso.")
    print("En un espacio real (miles de arqs.) esto es inviable → ver script02.py (DARTS).")
    print("⚠️ " * 20)
    print(f"\nEspacio de búsqueda: {len(SEARCH_SPACE)} candidatos × {EPOCHS_SEARCH} epochs cada uno")
    print("=" * 65)

    candidates = []
    for i, cfg in enumerate(SEARCH_SPACE):
        label = config_label(cfg)
        print(f"\n[{i+1}/{len(SEARCH_SPACE)}] {label}")

        model = CandidateCNN(**cfg).to(device)
        params = count_parameters(model)
        macs   = count_macs(model)

        train(model, train_loader, test_loader, epochs=EPOCHS_SEARCH,
              lr=LR, device=device, verbose=False)
        acc = evaluate(model, test_loader, device)
        size_kb = model_size_kb(model)

        print(f"  acc={acc:.4f}  params={params:,}  MACs={macs:,}  {size_kb:.1f} KB")

        candidates.append({
            "config":  cfg,
            "label":   label,
            "acc":     round(acc, 4),
            "params":  params,
            "macs":    macs,
            "size_kb": round(size_kb, 2),
        })

    # ---------------------------------------------------------------
    # Análisis del frente de Pareto
    # ---------------------------------------------------------------
    accs = [c["acc"]  for c in candidates]
    macs = [c["macs"] for c in candidates]
    pareto_idx = pareto_front(accs, macs)

    # Budget winner: mayor accuracy con MACs ≤ umbral (mediana de MACs)
    macs_median = float(np.median(macs))
    budget_candidates = [c for c in candidates if c["macs"] <= macs_median]
    budget_winner = max(budget_candidates, key=lambda c: c["acc"])

    # Accuracy winner: mayor accuracy sin restricciones
    acc_winner = max(candidates, key=lambda c: c["acc"])

    print(f"\n{'='*65}")
    print("TABLA (ordenada por MACs)")
    print(f"{'Config':<35} {'Acc':>7} {'MACs':>10} {'KB':>7}  Pareto")
    print("-" * 65)
    for c in sorted(candidates, key=lambda x: x["macs"]):
        is_pareto = "  ★" if candidates.index(c) in pareto_idx else ""
        print(f"{c['label']:<35} {c['acc']:>7.4f} {c['macs']:>10,} {c['size_kb']:>7.1f}{is_pareto}")

    print(f"\n★ Pareto-óptimos: {len(pareto_idx)} candidatos")
    print(f"  Budget winner (MACs ≤ {macs_median/1e6:.2f}M):  {budget_winner['label']}  acc={budget_winner['acc']:.4f}")
    print(f"  Accuracy winner:  {acc_winner['label']}  acc={acc_winner['acc']:.4f}")

    # ---------------------------------------------------------------
    # Gráfico: scatter accuracy vs MACs + Pareto front
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    # Todos los candidatos
    ax.scatter(
        [c["macs"] / 1e6 for c in candidates],
        [c["acc"]         for c in candidates],
        color="#4C72B0", s=60, zorder=3, label="Candidatos"
    )

    # Pareto front
    pareto_pts = sorted([candidates[i] for i in pareto_idx], key=lambda c: c["macs"])
    ax.plot(
        [c["macs"] / 1e6 for c in pareto_pts],
        [c["acc"]         for c in pareto_pts],
        "o-", color="#DD8452", linewidth=2, markersize=8, zorder=4, label="Frente de Pareto"
    )

    # Etiquetas especiales
    for c in [budget_winner, acc_winner]:
        ax.annotate(
            ("Budget\nwinner" if c is budget_winner else "Acc\nwinner"),
            xy=(c["macs"] / 1e6, c["acc"]),
            xytext=(15, -20 if c is budget_winner else 10),
            textcoords="offset points",
            fontsize=8,
            color="#55A868" if c is budget_winner else "#C44E52",
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    # Línea de presupuesto de MACs
    ax.axvline(macs_median / 1e6, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Budget MACs ({macs_median/1e6:.1f}M)")

    ax.set_xlabel("MACs (millones)")
    ax.set_ylabel("Accuracy en test")
    ax.set_title(f"Micro-NAS: Frente de Pareto — {len(SEARCH_SPACE)} candidatos, {EPOCHS_SEARCH} epochs c/u\n"
                 "(En NAS real: DARTS = gradiente en lugar de grilla; OFA = 1 supernet en lugar de N modelos)")
    ax.legend(fontsize=8)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "nas_pareto.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\n[INFO] Gráfico guardado en '{plot_path}'")
    plt.close()

    # Guardar JSON
    save_results(
        os.path.join(OUTPUT_DIR, "nas_resultados.json"),
        {
            "candidates":    candidates,
            "pareto_idx":    pareto_idx,
            "budget_winner": budget_winner,
            "acc_winner":    acc_winner,
            "macs_budget":   macs_median,
        }
    )

    print("\n[CONCLUSIÓN — POR QUÉ ESTE ENFOQUE NO ESCALA]")
    print(f"  Evaluamos {len(SEARCH_SPACE)} arquitecturas en ~{len(SEARCH_SPACE) * EPOCHS_SEARCH} epochs totales.")
    print(f"  El frente de Pareto tiene {len(pareto_idx)} candidatos óptimos.")
    print(f"  Con conv1∈{{4,8,16}}, conv2∈{{8,16,32}}, dense∈{{32,64,128}} hay 3³=27 combinaciones.")
    print(f"  En NAS real: espacio de búsqueda con >10^18 combinaciones.")
    print(f"  → DARTS (script02.py) resuelve esto en O(1) pases con búsqueda diferenciable.")
    print(f"  → OFA / Weight sharing: un único supernet en lugar de {len(SEARCH_SPACE)} entrenamientos.")


if __name__ == "__main__":
    main()
