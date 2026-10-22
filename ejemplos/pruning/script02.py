"""
Sección 7 — Pruning: Criterio basado en magnitud de pesos
===========================================================
Aplica pruning estructurado (a nivel de filtros Conv2d) usando como criterio
la norma L1 de los pesos de cada filtro. Es el criterio más simple posible:
importancia ∝ ‖W_filtro‖₁.

Flujo:
  1. Entrenar CNN baseline en Fashion MNIST.
  2. Calcular la importancia de cada filtro por norma L1 de sus pesos.
  3. Aplicar pruning iterativo: eliminar los filtros menos importantes
     en pasos de 10% hasta llegar al 70% de filtros eliminados.
  4. Fine-tune breve (2 epochs) después de cada paso.
  5. Comparar la curva accuracy vs sparsity.

Mensaje clave:
  La magnitud de los pesos es un proxy simple para la importancia de un filtro.
  Funciona razonablemente bien, pero ignora cómo la red realmente usa cada filtro
  durante la inferencia. Ver script03.py para un criterio basado en activaciones.

Uso:
    cd examples/pruning
    uv run python script02.py
"""

import copy
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from pytorch_utils import (
    count_macs,
    count_parameters,
    evaluate,
    get_dataloaders,
    get_device,
    model_size_kb,
    print_model_stats,
    save_results,
    train,
)

OUTPUT_DIR       = "outputs"
EPOCHS_BASELINE  = 10
EPOCHS_FINETUNE  = 2
BATCH_SIZE       = 256
LR               = 1e-3
SPARSITY_STEPS   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # fracción de filtros a eliminar


# ---------------------------------------------------------------------------
# Arquitectura (igual que script01 para comparabilidad)
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    """CNN LeNet-5 pequeña para Fashion MNIST."""
    def __init__(self, conv1_out=6, conv2_out=16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=5)
        self.pool  = nn.MaxPool2d(2)
        flat_dim   = conv2_out * 4 * 4
        self.fc1   = nn.Linear(flat_dim, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# Pruning estructurado por magnitud de pesos
# ---------------------------------------------------------------------------

def filter_importance_magnitude(conv: nn.Conv2d) -> torch.Tensor:
    """
    Importancia de cada filtro de salida = norma L1 de sus pesos.
    weight shape: (out_channels, in_channels, kH, kW)
    Retorna tensor de shape (out_channels,).
    """
    return conv.weight.data.abs().sum(dim=(1, 2, 3))


def build_pruned_cnn(base_model: CNN, n_keep1: int, n_keep2: int,
                     keep1: torch.Tensor, keep2: torch.Tensor) -> CNN:
    """
    Construye una CNN más pequeña conservando sólo los filtros indicados
    y copiando los pesos correspondientes del modelo base.
    """
    pruned = CNN(conv1_out=n_keep1, conv2_out=n_keep2)
    with torch.no_grad():
        pruned.conv1.weight.copy_(base_model.conv1.weight.data[keep1])
        if base_model.conv1.bias is not None:
            pruned.conv1.bias.copy_(base_model.conv1.bias.data[keep1])

        pruned.conv2.weight.copy_(base_model.conv2.weight.data[keep2][:, keep1, :, :])
        if base_model.conv2.bias is not None:
            pruned.conv2.bias.copy_(base_model.conv2.bias.data[keep2])

        # fc1: columnas = canales keep2, cada canal ocupa 4×4=16 entradas
        keep2_flat = torch.cat([keep2 * 16 + i for i in range(16)]).sort().values
        pruned.fc1.weight.copy_(base_model.fc1.weight.data[:, keep2_flat])
        pruned.fc1.bias.copy_(base_model.fc1.bias.data)
        pruned.fc2.weight.copy_(base_model.fc2.weight.data)
        pruned.fc2.bias.copy_(base_model.fc2.bias.data)
        pruned.fc3.weight.copy_(base_model.fc3.weight.data)
        pruned.fc3.bias.copy_(base_model.fc3.bias.data)
    return pruned


def prune_by_magnitude(model: CNN, sparsity: float) -> CNN:
    """
    Aplica pruning estructurado eliminando la fracción `sparsity` de filtros
    con menor norma L1 en cada capa Conv2d.
    """
    imp1 = filter_importance_magnitude(model.conv1)
    n_keep1 = max(1, int(model.conv1.out_channels * (1 - sparsity)))
    keep1 = imp1.topk(n_keep1).indices.sort().values

    imp2 = filter_importance_magnitude(model.conv2)
    n_keep2 = max(1, int(model.conv2.out_channels * (1 - sparsity)))
    keep2 = imp2.topk(n_keep2).indices.sort().values

    return build_pruned_cnn(model, n_keep1, n_keep2, keep1, keep2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Baseline
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 1 — Entrenando CNN baseline")
    print("="*60)
    baseline = CNN().to(device)
    train(baseline, train_loader, test_loader, epochs=EPOCHS_BASELINE, lr=LR, device=device)
    base_acc = evaluate(baseline, test_loader, device)
    print(f"\nBaseline  acc={base_acc:.4f}")
    print_model_stats(baseline, "Baseline")

    results = {
        "baseline": {
            "acc": round(base_acc, 4),
            "params": count_parameters(baseline),
            "macs": count_macs(baseline),
            "size_kb": round(model_size_kb(baseline), 2),
            "sparsity": 0.0,
        }
    }

    # ------------------------------------------------------------------
    # 2. Pruning iterativo por magnitud
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 2 — Pruning iterativo por magnitud de pesos (L1)")
    print("="*60)
    print(f"\n  Criterio: importancia(filtro) = ‖W_filtro‖₁")
    print(f"  Se eliminan los filtros con MENOR importancia.\n")
    print(f"  {'Sparsity':>9} | {'Acc antes FT':>13} | {'Acc después FT':>14} | {'Paráms':>9} | {'MACs':>10}")
    print("  " + "-" * 65)

    sparsity_vals, accs_before, accs_after, param_counts, mac_counts = [], [], [], [], []

    for sparsity in SPARSITY_STEPS:
        pruned = prune_by_magnitude(baseline, sparsity).to(device)
        acc_before_ft = evaluate(pruned, test_loader, device)
        train(pruned, train_loader, test_loader, epochs=EPOCHS_FINETUNE, lr=LR/5,
              device=device, verbose=False)
        acc_after_ft = evaluate(pruned, test_loader, device)

        params = count_parameters(pruned)
        macs   = count_macs(pruned)
        print(f"  {sparsity:>8.0%} | {acc_before_ft:>13.4f} | {acc_after_ft:>14.4f} | {params:>9,} | {macs:>10,}")

        sparsity_vals.append(sparsity)
        accs_before.append(acc_before_ft)
        accs_after.append(acc_after_ft)
        param_counts.append(params)
        mac_counts.append(macs)

        results[f"magnitude_{int(sparsity*100)}pct"] = {
            "sparsity": sparsity,
            "acc_before_ft": round(acc_before_ft, 4),
            "acc_after_ft":  round(acc_after_ft, 4),
            "params": params,
            "macs":   macs,
        }

    # ------------------------------------------------------------------
    # 3. Gráfico
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    x = [0.0] + sparsity_vals
    y_base   = [base_acc] + accs_before
    y_ft     = [base_acc] + accs_after
    macs_all = [results["baseline"]["macs"]] + mac_counts

    ax = axes[0]
    ax.plot([s * 100 for s in x], y_base, "o--", color="#DD8452", label="Sin fine-tuning", markersize=6)
    ax.plot([s * 100 for s in x], y_ft,   "s-",  color="#4C72B0", label="Con fine-tuning (2 epochs)", markersize=6)
    ax.axhline(base_acc, color="gray", linestyle=":", linewidth=1, label=f"Baseline ({base_acc:.4f})")
    ax.set_xlabel("Filtros eliminados (%)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Nivel de Pruning\n(criterio: magnitud de pesos L1)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    ax = axes[1]
    ax.plot([s * 100 for s in x], [m / 1e6 for m in macs_all], "o-", color="#55A868", markersize=6)
    ax.set_xlabel("Filtros eliminados (%)")
    ax.set_ylabel("MACs (millones)")
    ax.set_title("Reducción de MACs vs Sparsity\n(pruning estructurado sí reduce MACs)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Pruning por Magnitud de Pesos (L1) — Fashion MNIST CNN", fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "pruning_magnitud.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\n[INFO] Gráfico guardado en '{plot_path}'")
    plt.close()

    save_results(os.path.join(OUTPUT_DIR, "pruning_magnitud_resultados.json"), results)

    print("\n[CONCLUSIÓN]")
    best_sparsity = SPARSITY_STEPS[accs_after.index(max(accs_after))]
    best_acc = max(accs_after)
    mac_reduction = 1 - mac_counts[SPARSITY_STEPS.index(best_sparsity)] / results["baseline"]["macs"]
    print(f"  Criterio: norma L1 de pesos por filtro.")
    print(f"  Mejor balance en {best_sparsity:.0%} sparsity: acc={best_acc:.4f}, reducción MACs={mac_reduction:.1%}.")
    print(f"  Limitación: la magnitud de los pesos no captura cuánto contribuye cada filtro")
    print(f"  a las predicciones reales del modelo. Ver script03.py (activaciones).")


if __name__ == "__main__":
    main()
