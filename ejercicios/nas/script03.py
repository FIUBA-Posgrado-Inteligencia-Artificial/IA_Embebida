"""
Sección 10c — NAS multi-objetivo: frente de Pareto en un MLP
=============================================================
Muestra el concepto central de NAS multi-objetivo: buscar el mejor equilibrio
entre accuracy y eficiencia (número de parámetros).

Dataset: clasificación sintética (sklearn make_classification) —
         800 muestras, 20 features, 4 clases, estructura no lineal.
         Suficientemente difícil para que el tamaño del modelo importe.
Red:     MLP con capas ocultas configurables (PyTorch).

Espacio de búsqueda:
  n_hidden_layers ∈ {1, 2, 3}
  hidden_size     ∈ {8, 16, 32, 64, 128}
  → 15 configuraciones

El script entrena cada configuración brevemente, calcula el frente de Pareto
(maximizar accuracy, minimizar parámetros) y genera un scatter plot.

Mensaje clave: no existe una única "mejor" red. La elección depende de las
restricciones del hardware destino. El frente de Pareto resume todas las
opciones óptimas de una vez.

Uso:
    cd examples/nas
    uv run python script03.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR  = "outputs"
EPOCHS      = 150
LR          = 1e-3
BATCH_SIZE  = 64

N_LAYERS_OPTIONS    = [1, 2, 3]
HIDDEN_SIZE_OPTIONS = [8, 16, 32, 64, 128]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset():
    """
    Genera un dataset sintético de clasificación con 4 clases y estructura
    no lineal: suficientemente difícil para que el tamaño del modelo importe,
    suficientemente pequeño para entrenar en segundos.
    """
    X, y = make_classification(
        n_samples=800, n_features=20, n_informative=15, n_redundant=3,
        n_classes=4, n_clusters_per_class=2, random_state=42,
    )
    X = X.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return (
        torch.tensor(X_train), torch.tensor(y_train),
        torch.tensor(X_test),  torch.tensor(y_test),
    )


# ---------------------------------------------------------------------------
# Arquitectura paramétrica
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_features: int, n_classes: int,
                 n_hidden_layers: int, hidden_size: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(prev, hidden_size), nn.ReLU()]
            prev = hidden_size
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------

def train_model(model: nn.Module,
                X_train: torch.Tensor, y_train: torch.Tensor,
                epochs: int, lr: float) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    n = X_train.size(0)

    model.train()
    for _ in range(epochs):
        # Mini-batches
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate(model: nn.Module,
             X_test: torch.Tensor, y_test: torch.Tensor) -> float:
    model.eval()
    preds = model(X_test).argmax(dim=1)
    return (preds == y_test).float().mean().item()


# ---------------------------------------------------------------------------
# Frente de Pareto
# ---------------------------------------------------------------------------

def pareto_front(accs: list[float], params: list[int]) -> list[int]:
    """
    Índices de los puntos Pareto-óptimos:
    maximizar accuracy, minimizar parámetros.
    """
    n = len(accs)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if accs[j] >= accs[i] and params[j] <= params[i]:
                if accs[j] > accs[i] or params[j] < params[i]:
                    dominated[i] = True
                    break
    return [i for i in range(n) if not dominated[i]]


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

def plot_pareto(candidates: list[dict], pareto_idx: list[int],
                budget_winner: dict, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    accs   = [c["acc"]    for c in candidates]
    params = [c["params"] for c in candidates]

    # Todos los candidatos
    ax.scatter(params, accs, color="#4C72B0", s=60, zorder=3, label="Candidatos")

    # Frente de Pareto (ordenado por params)
    pf = sorted([candidates[i] for i in pareto_idx], key=lambda c: c["params"])
    ax.plot(
        [c["params"] for c in pf],
        [c["acc"]    for c in pf],
        "o-", color="#DD8452", linewidth=2, markersize=8, zorder=4,
        label="Frente de Pareto",
    )

    # Etiquetas de los puntos Pareto
    for c in pf:
        ax.annotate(
            c["label"],
            xy=(c["params"], c["acc"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7,
            color="#DD8452",
        )

    # Budget winner
    bw = budget_winner
    ax.scatter([bw["params"]], [bw["acc"]], color="#55A868", s=120,
               zorder=5, marker="*", label=f"Budget winner ({bw['label']})")

    # Línea de presupuesto
    budget_line = np.median(params)
    ax.axvline(budget_line, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Límite presupuesto ({int(budget_line):,} params)")

    ax.set_xlabel("Parámetros")
    ax.set_ylabel("Accuracy en test")
    ax.set_title(
        "NAS multi-objetivo: frente de Pareto — MLP sobre Wine dataset\n"
        "(accuracy vs. parámetros, 15 configuraciones)"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[INFO] Figura guardada en '{save_path}'")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(42)

    X_train, y_train, X_test, y_test = load_dataset()
    in_features = X_train.shape[1]  # 13
    n_classes   = int(y_train.max().item()) + 1  # 3

    # Construir espacio de búsqueda
    search_space = [
        {"n_hidden_layers": nl, "hidden_size": hs}
        for nl in N_LAYERS_OPTIONS
        for hs in HIDDEN_SIZE_OPTIONS
    ]

    print(f"\nDataset sintético: {X_train.shape[0]} train / {X_test.shape[0]} test  "
          f"({in_features} features, {n_classes} clases)")
    print(f"Espacio de búsqueda: {len(search_space)} configuraciones × {EPOCHS} epochs c/u")
    print("=" * 60)

    candidates = []
    for i, cfg in enumerate(search_space):
        label = f"L{cfg['n_hidden_layers']}×H{cfg['hidden_size']}"
        model = MLP(in_features, n_classes, **cfg)
        n_params = count_params(model)

        train_model(model, X_train, y_train, EPOCHS, LR)
        acc = evaluate(model, X_test, y_test)

        print(f"  [{i+1:2d}/{len(search_space)}] {label:<10}  acc={acc:.4f}  params={n_params:,}")
        candidates.append({"label": label, "acc": acc, "params": n_params, **cfg})

    # Frente de Pareto
    accs   = [c["acc"]    for c in candidates]
    params = [c["params"] for c in candidates]
    pidx   = pareto_front(accs, params)

    # Budget winner: mejor accuracy con params ≤ mediana
    budget = float(np.median(params))
    budget_candidates = [c for c in candidates if c["params"] <= budget]
    budget_winner = max(budget_candidates, key=lambda c: c["acc"])

    print(f"\n{'='*60}")
    print(f"★ Puntos Pareto-óptimos: {len(pidx)}")
    for i in sorted(pidx, key=lambda i: candidates[i]["params"]):
        c = candidates[i]
        print(f"   {c['label']:<10}  acc={c['acc']:.4f}  params={c['params']:,}")

    print(f"\n  Budget winner (params ≤ {int(budget):,}): "
          f"{budget_winner['label']}  acc={budget_winner['acc']:.4f}")

    plot_pareto(
        candidates, pidx, budget_winner,
        os.path.join(OUTPUT_DIR, "nas_pareto_mlp.png"),
    )

    print("\n[CONCLUSIÓN]")
    print("  El frente de Pareto resume todas las redes óptimas.")
    print("  No hay una sola 'mejor red': la elección depende del")
    print("  presupuesto de parámetros (o latencia) del hardware destino.")


if __name__ == "__main__":
    main()
