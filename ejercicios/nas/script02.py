"""
Sección 10b — DARTS simplificado
=================================
Implementa el mecanismo central de DARTS (Differentiable Architecture Search)
sobre Fashion MNIST en tres fases:

  1. Búsqueda  — entrena pesos W y parámetros de arquitectura α conjuntamente.
                 Cada capa es una MixedOp: suma ponderada de 4 operaciones
                 con pesos softmax(α).
  2. Derivación — argmax(α) por capa → arquitectura discreta.
  3. Reentrenamiento — entrena la arquitectura derivada desde cero.

Esta es una simplificación didáctica:
  - Se usa una red secuencial de 3 capas con una MixedOp cada una.
  - La optimización conjunta de W y α (en lugar de bilevel) reduce complejidad.
  - El espacio de operaciones es pequeño (4 ops) y las capas son independientes.

Para la versión completa con celdas normal/reducción, múltiples nodos y
optimización bilevel (gradiente de segundo orden), ver:
  Paper: https://arxiv.org/abs/1806.09055
  Repo:  https://github.com/quark0/darts

Uso:
    cd examples/nas
    uv run python script02.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_utils import (
    evaluate,
    get_dataloaders,
    get_device,
    save_results,
)

OUTPUT_DIR       = "outputs"
EPOCHS_SEARCH    = 15   # epochs de búsqueda (W + α)
EPOCHS_RETRAIN   = 20   # epochs de reentrenamiento de la arquitectura derivada
BATCH_SIZE       = 256
LR_WEIGHTS       = 1e-3
LR_ARCH          = 3e-4
CHANNELS         = 32   # canales internos fijos para todas las ops

OP_NAMES = ["conv3x3", "conv5x5", "maxpool3x3", "identity"]


# ---------------------------------------------------------------------------
# Operaciones candidatas
# ---------------------------------------------------------------------------

def make_op(name: str, C_in: int, C_out: int, stride: int = 1) -> nn.Module:
    """Construye una operación por nombre. Todas preservan resolución espacial."""
    if name == "conv3x3":
        return nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        )
    if name == "conv5x5":
        return nn.Sequential(
            nn.Conv2d(C_in, C_out, 5, stride=stride, padding=2, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        )
    if name == "maxpool3x3":
        # MaxPool no cambia canales; si C_in != C_out, añade proyección 1×1
        layers: list[nn.Module] = [nn.MaxPool2d(3, stride=stride, padding=1)]
        if C_in != C_out:
            layers.append(nn.Conv2d(C_in, C_out, 1, bias=False))
            layers.append(nn.BatchNorm2d(C_out))
        return nn.Sequential(*layers)
    if name == "identity":
        if C_in == C_out and stride == 1:
            return nn.Identity()
        # Si los canales difieren o hay stride, proyectar con 1×1
        return nn.Sequential(
            nn.Conv2d(C_in, C_out, 1, stride=stride, bias=False),
            nn.BatchNorm2d(C_out),
        )
    raise ValueError(f"Operación desconocida: {name}")


# ---------------------------------------------------------------------------
# MixedOp: suma ponderada de todas las operaciones
# ---------------------------------------------------------------------------

class MixedOp(nn.Module):
    """
    Suma ponderada de N operaciones candidatas:
        output = Σ softmax(α_i) · op_i(x)
    Los α son parámetros externos (gestionados por DARTSNet).
    """
    def __init__(self, C_in: int, C_out: int, stride: int = 1):
        super().__init__()
        self.ops = nn.ModuleList([
            make_op(name, C_in, C_out, stride) for name in OP_NAMES
        ])

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(alpha, dim=0)
        return sum(w * op(x) for w, op in zip(weights, self.ops))


# ---------------------------------------------------------------------------
# DARTSNet: 3 capas MixedOp + clasificador
# ---------------------------------------------------------------------------

class DARTSNet(nn.Module):
    """
    Red con 3 capas de MixedOp apiladas.
    Estructura:
        stem (conv 1→C) → MixedOp×3 → AdaptiveAvgPool → FC(10)

    Los parámetros α se guardan separados de los pesos del modelo para
    permitir tasas de aprendizaje distintas.
    """
    def __init__(self, channels: int = CHANNELS):
        super().__init__()
        C = channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
        # 3 capas MixedOp; la segunda y tercera reducen resolución (stride=2)
        self.mixed1 = MixedOp(C, C, stride=1)
        self.mixed2 = MixedOp(C, C, stride=2)   # 28→14
        self.mixed3 = MixedOp(C, C, stride=2)   # 14→7
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(C, 10)

        # Parámetros de arquitectura: uno por (capa, operación)
        # Forma: (n_layers=3, n_ops=4). Inicializados en 0 → softmax uniforme.
        self.arch_params = nn.Parameter(
            torch.zeros(3, len(OP_NAMES)), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.mixed1(x, self.arch_params[0])
        x = self.mixed2(x, self.arch_params[1])
        x = self.mixed3(x, self.arch_params[2])
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def arch_weights(self) -> np.ndarray:
        """Devuelve softmax(α) como array numpy (3×4) para visualización."""
        with torch.no_grad():
            return F.softmax(self.arch_params, dim=1).cpu().numpy()

    def derive(self) -> list[str]:
        """Retorna la operación ganadora (argmax α) por capa."""
        idx = self.arch_params.argmax(dim=1).tolist()
        return [OP_NAMES[i] for i in idx]


# ---------------------------------------------------------------------------
# Red derivada: arquitectura discreta tras argmax
# ---------------------------------------------------------------------------

class DerivedNet(nn.Module):
    """Red con las operaciones ganadoras fijas (sin MixedOp)."""
    def __init__(self, ops: list[str], channels: int = CHANNELS):
        super().__init__()
        C = channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
        self.op1  = make_op(ops[0], C, C, stride=1)
        self.op2  = make_op(ops[1], C, C, stride=2)
        self.op3  = make_op(ops[2], C, C, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(C, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación locales
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    correct = total = 0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train_darts_epoch(
    model: DARTSNet,
    train_loader: torch.utils.data.DataLoader,
    w_optimizer: torch.optim.Optimizer,
    arch_optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Un epoch de búsqueda DARTS: alterna update de W y α por mini-batch."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    train_iter = iter(train_loader)

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # — Update pesos W —
        w_optimizer.zero_grad()
        loss_w = criterion(model(x), y)
        loss_w.backward()
        w_optimizer.step()

        # — Update α con otro mini-batch (aproximación bilevel) —
        try:
            x_a, y_a = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_a, y_a = next(train_iter)
        x_a, y_a = x_a.to(device), y_a.to(device)

        arch_optimizer.zero_grad()
        loss_a = criterion(model(x_a), y_a)
        loss_a.backward()
        arch_optimizer.step()

        with torch.no_grad():
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


# ---------------------------------------------------------------------------
# Figura: heatmap α + curvas de accuracy
# ---------------------------------------------------------------------------

def plot_results(
    arch_weights: np.ndarray,
    derived_ops: list[str],
    search_accs: list[float],
    retrain_accs: list[float],
    save_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Subplot izquierdo: heatmap de pesos α finales
    ax = axes[0]
    im = ax.imshow(arch_weights, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(OP_NAMES)))
    ax.set_xticklabels(OP_NAMES, rotation=15, ha="right", fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"Capa {i+1}" for i in range(3)])
    ax.set_title("Pesos de arquitectura softmax(α)\n(celda más brillante = operación ganadora)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Marcar ganadora por capa
    for i, op in enumerate(derived_ops):
        j = OP_NAMES.index(op)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="#50fa7b", linewidth=2.5))
        ax.text(j, i, "★", ha="center", va="center", color="#50fa7b", fontsize=14)

    # Subplot derecho: curvas de accuracy
    ax2 = axes[1]
    ax2.plot(range(1, len(search_accs) + 1), search_accs,
             "o-", color="#8be9fd", linewidth=2, label=f"Búsqueda ({len(search_accs)} epochs)")
    ax2.plot(range(1, len(retrain_accs) + 1), retrain_accs,
             "s-", color="#50fa7b", linewidth=2, label=f"Reentrenamiento ({len(retrain_accs)} epochs)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy en train")
    ax2.set_title("Accuracy: búsqueda vs reentrenamiento")
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"DARTS simplificado — Arquitectura derivada: {' → '.join(derived_ops)}",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[INFO] Figura guardada en '{save_path}'")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = get_device()
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------------
    # Fase 1: Búsqueda (DARTS)
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("FASE 1 — Búsqueda DARTS (W + α conjuntamente)")
    print("=" * 65)

    model = DARTSNet(channels=CHANNELS).to(device)

    # Separar parámetros: pesos de red vs parámetros de arquitectura
    weight_params = [p for n, p in model.named_parameters() if n != "arch_params"]
    arch_params   = [model.arch_params]

    w_optimizer    = torch.optim.Adam(weight_params, lr=LR_WEIGHTS, weight_decay=1e-4)
    arch_optimizer = torch.optim.Adam(arch_params,   lr=LR_ARCH)

    search_accs: list[float] = []
    for epoch in range(1, EPOCHS_SEARCH + 1):
        train_acc = train_darts_epoch(model, train_loader, w_optimizer, arch_optimizer, device)
        test_acc  = evaluate(model, test_loader, device)
        search_accs.append(train_acc)

        weights = model.arch_weights()
        ops_str = " | ".join(
            f"L{i+1}: {OP_NAMES[weights[i].argmax()]}({weights[i].max():.2f})"
            for i in range(3)
        )
        print(f"  Epoch {epoch:2d}/{EPOCHS_SEARCH} — train={train_acc:.4f}  test={test_acc:.4f}  [{ops_str}]")

    # ---------------------------------------------------------------
    # Fase 2: Derivación
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("FASE 2 — Derivación (argmax α)")
    print("=" * 65)

    derived_ops = model.derive()
    arch_weights_final = model.arch_weights()
    print("  Arquitectura derivada:")
    for i, op in enumerate(derived_ops):
        w = arch_weights_final[i]
        print(f"    Capa {i+1}: {op}  (α = [{', '.join(f'{v:.3f}' for v in w)}])")

    # ---------------------------------------------------------------
    # Fase 3: Reentrenamiento
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("FASE 3 — Reentrenamiento de la arquitectura derivada")
    print("=" * 65)

    derived_model = DerivedNet(derived_ops, channels=CHANNELS).to(device)
    retrain_optimizer = torch.optim.Adam(
        derived_model.parameters(), lr=LR_WEIGHTS, weight_decay=1e-4
    )

    retrain_accs: list[float] = []
    for epoch in range(1, EPOCHS_RETRAIN + 1):
        train_acc = train_one_epoch(derived_model, train_loader, retrain_optimizer, device)
        test_acc  = evaluate(derived_model, test_loader, device)
        retrain_accs.append(train_acc)
        print(f"  Epoch {epoch:2d}/{EPOCHS_RETRAIN} — train={train_acc:.4f}  test={test_acc:.4f}")

    final_test_acc = evaluate(derived_model, test_loader, device)

    # ---------------------------------------------------------------
    # Resultados y figura
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("RESULTADOS")
    print("=" * 65)
    print(f"  Arquitectura derivada: {' → '.join(derived_ops)}")
    print(f"  Accuracy final (reentrenamiento): {final_test_acc:.4f}")

    plot_results(
        arch_weights_final,
        derived_ops,
        search_accs,
        retrain_accs,
        os.path.join(OUTPUT_DIR, "nas_darts.png"),
    )

    save_results(
        os.path.join(OUTPUT_DIR, "darts_resultados.json"),
        {
            "derived_ops":    derived_ops,
            "arch_weights":   arch_weights_final.tolist(),
            "search_accs":    search_accs,
            "retrain_accs":   retrain_accs,
            "final_test_acc": round(final_test_acc, 4),
        },
    )

    print("\n[CONCLUSIÓN]")
    print(f"  DARTS aprendió α en {EPOCHS_SEARCH} epochs y derivó la arquitectura:")
    print(f"  {' → '.join(derived_ops)}")
    print(f"  En NAS real (paper), el espacio es exponencialmente mayor")
    print(f"  (celdas con múltiples nodos, ~10⁸ arquitecturas posibles),")
    print(f"  pero el mismo mecanismo diferenciable mantiene el costo bajo.")


if __name__ == "__main__":
    main()
