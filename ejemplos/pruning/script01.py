"""
Sección 7 — Pruning
====================
Demuestra la diferencia entre pruning no estructurado y estructurado sobre una CNN
entrenada en Fashion MNIST.

Flujo:
  1. Entrenar una CNN de referencia (baseline).
  2. Pruning NO ESTRUCTURADO: zeroing de pesos individuales por magnitud
     → sparsidad 50% y 75%.
  3. Pruning ESTRUCTURADO: eliminar los filtros más débiles (L1) de cada capa Conv2d
     → reducir realmente el número de canales (50% de filtros eliminados).
  4. Comparar: accuracy, parámetros, MACs y tamaño del modelo.

Mensaje clave: la sparsidad no estructurada NO reduce los MACs en hardware genérico.
El pruning estructurado sí reduce los MACs y el tamaño del modelo.

Uso:
    cd examples/pruning
    uv run python script01.py
"""

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
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

OUTPUT_DIR = "outputs"
EPOCHS_BASELINE = 10
EPOCHS_FINETUNE = 3
BATCH_SIZE = 256
LR = 1e-3

FASHION_MNIST_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ---------------------------------------------------------------------------
# Arquitectura CNN (LeNet-5 style, misma que sección 6)
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    """
    CNN pequeña para Fashion MNIST.
    Conv2d(1→6, 5×5, valid) → MaxPool → Conv2d(6→16, 5×5, valid) → MaxPool
    → Flatten → Linear(256→120) → Linear(120→84) → Linear(84→10)
    """
    def __init__(self, conv1_out=6, conv2_out=16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=5)   # 28→24
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=5)  # 12→8
        self.pool  = nn.MaxPool2d(2)
        flat_dim   = conv2_out * 4 * 4
        self.fc1   = nn.Linear(flat_dim, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # → (B, 6, 12, 12)
        x = self.pool(torch.relu(self.conv2(x)))  # → (B, 16, 4, 4)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# Pruning no estructurado
# ---------------------------------------------------------------------------

def apply_unstructured_pruning(model: nn.Module, sparsity: float) -> nn.Module:
    """
    Aplica L1-unstructured pruning a todos los Conv2d y Linear con el nivel
    de sparsidad indicado, y luego hace permanente la máscara (elimina los
    hooks de pruning para que el modelo sea evaluble sin overheads).
    """
    model = copy.deepcopy(model)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=sparsity)
            prune.remove(module, "weight")  # hace permanente la máscara
    return model


def get_sparsity(model: nn.Module) -> float:
    """Fracción real de pesos iguales a cero en Conv2d y Linear."""
    zeros, total = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            zeros += (m.weight == 0).sum().item()
            total += m.weight.numel()
    return zeros / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Pruning estructurado (por filtros Conv2d)
# ---------------------------------------------------------------------------

def get_filter_importance(conv: nn.Conv2d) -> torch.Tensor:
    """L1-norm de cada filtro de salida de una capa Conv2d."""
    # weight shape: (out_channels, in_channels, kH, kW)
    return conv.weight.data.abs().sum(dim=(1, 2, 3))


def build_pruned_cnn(base_model: nn.Module, prune_ratio: float) -> nn.Module:
    """
    Construye una nueva CNN más pequeña eliminando los `prune_ratio` filtros
    más débiles de cada capa Conv2d (por norma L1). Reemplaza también las
    capas Linear que dependen de la dimensión de salida de las Conv2d.

    Retorna un nuevo modelo con pesos copiados de las neuronas supervivientes.
    """
    conv1 = base_model.conv1
    conv2 = base_model.conv2

    # --- Seleccionar filtros a conservar en cada capa ---
    imp1 = get_filter_importance(conv1)
    n_keep1 = max(1, int(conv1.out_channels * (1 - prune_ratio)))
    keep1 = imp1.topk(n_keep1).indices.sort().values

    imp2 = get_filter_importance(conv2)
    n_keep2 = max(1, int(conv2.out_channels * (1 - prune_ratio)))
    keep2 = imp2.topk(n_keep2).indices.sort().values

    # --- Construir nuevo modelo con los canales reducidos ---
    pruned = CNN(conv1_out=n_keep1, conv2_out=n_keep2)

    with torch.no_grad():
        # Conv1: seleccionar filtros de salida keep1
        pruned.conv1.weight.copy_(conv1.weight.data[keep1])
        if conv1.bias is not None:
            pruned.conv1.bias.copy_(conv1.bias.data[keep1])

        # Conv2: seleccionar entradas keep1 y salidas keep2
        pruned.conv2.weight.copy_(conv2.weight.data[keep2][:, keep1, :, :])
        if conv2.bias is not None:
            pruned.conv2.bias.copy_(conv2.bias.data[keep2])

        # fc1: las entradas ahora son n_keep2 * 4 * 4
        # El fc1 original tiene 256 entradas (16*4*4). Seleccionamos las
        # columnas correspondientes a los canales keep2.
        # Columnas: cada canal k ocupa 16 entradas (4x4 = 16)
        keep2_flat = torch.cat([keep2 * 16 + i for i in range(16)]).sort().values
        pruned.fc1.weight.copy_(base_model.fc1.weight.data[:, keep2_flat])
        pruned.fc1.bias.copy_(base_model.fc1.bias.data)

        # fc2 y fc3 no cambian
        pruned.fc2.weight.copy_(base_model.fc2.weight.data)
        pruned.fc2.bias.copy_(base_model.fc2.bias.data)
        pruned.fc3.weight.copy_(base_model.fc3.weight.data)
        pruned.fc3.bias.copy_(base_model.fc3.bias.data)

    return pruned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    # ------------------------------------------------------------------
    # 1. Baseline
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 1 — Entrenando CNN baseline")
    print("="*60)
    baseline = CNN().to(device)
    train(baseline, train_loader, test_loader, epochs=EPOCHS_BASELINE, lr=LR, device=device)
    base_acc = evaluate(baseline, test_loader, device)
    print(f"\nBaseline  →  acc={base_acc:.4f}")
    print_model_stats(baseline, "Baseline")

    results = {}
    results["baseline"] = {
        "acc": round(base_acc, 4),
        "params": count_parameters(baseline),
        "macs": count_macs(baseline),
        "size_kb": round(model_size_kb(baseline), 2),
        "sparsity": 0.0,
    }

    # ------------------------------------------------------------------
    # 2. Pruning NO ESTRUCTURADO
    # ------------------------------------------------------------------
    for sparsity_target in [0.50, 0.75]:
        label = f"unstructured_{int(sparsity_target*100)}pct"
        print(f"\n{'='*60}")
        print(f"PASO 2 — Pruning NO ESTRUCTURADO  sparsidad={sparsity_target:.0%}")
        print("="*60)

        pruned = apply_unstructured_pruning(baseline, sparsity_target)
        real_sparsity = get_sparsity(pruned)
        print(f"  Sparsidad real después del pruning: {real_sparsity:.2%}")

        # Fine-tuning
        print(f"  Fine-tuning {EPOCHS_FINETUNE} epochs...")
        train(pruned, train_loader, test_loader, epochs=EPOCHS_FINETUNE, lr=LR/5, device=device)

        acc = evaluate(pruned, test_loader, device)
        print(f"  Acc={acc:.4f}")
        print_model_stats(pruned, label)
        macs = count_macs(pruned)
        print(f"  ⚠ MACs={macs:,}  (idéntico al baseline — la sparsidad no estructurada")
        print(f"     no reduce los MACs en hardware genérico)")

        results[label] = {
            "acc": round(acc, 4),
            "params": count_parameters(pruned),
            "macs": macs,
            "size_kb": round(model_size_kb(pruned), 2),
            "sparsity": round(real_sparsity, 4),
        }

    # ------------------------------------------------------------------
    # 3. Pruning ESTRUCTURADO (50% filtros eliminados)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PASO 3 — Pruning ESTRUCTURADO  (50% filtros eliminados)")
    print("="*60)

    pruned_struct = build_pruned_cnn(baseline, prune_ratio=0.5)
    pruned_struct = pruned_struct.to(device)

    acc_before = evaluate(pruned_struct, test_loader, device)
    print(f"  Acc antes del fine-tuning: {acc_before:.4f}")
    print_model_stats(pruned_struct, "Structured-50% (antes FT)")

    print(f"  Fine-tuning {EPOCHS_FINETUNE} epochs...")
    train(pruned_struct, train_loader, test_loader, epochs=EPOCHS_FINETUNE, lr=LR/5, device=device)

    acc_struct = evaluate(pruned_struct, test_loader, device)
    print(f"  Acc después del fine-tuning: {acc_struct:.4f}")
    print_model_stats(pruned_struct, "Structured-50% (después FT)")

    results["structured_50pct"] = {
        "acc": round(acc_struct, 4),
        "params": count_parameters(pruned_struct),
        "macs": count_macs(pruned_struct),
        "size_kb": round(model_size_kb(pruned_struct), 2),
        "sparsity": "structured",
    }

    # ------------------------------------------------------------------
    # 4. Tabla comparativa
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("TABLA COMPARATIVA")
    print("="*60)
    header = f"{'Modelo':<30} {'Acc':>7} {'Paráms':>10} {'MACs':>12} {'KB':>8} {'Sparsidad':>12}"
    print(header)
    print("-" * len(header))
    labels_order = ["baseline", "unstructured_50pct", "unstructured_75pct", "structured_50pct"]
    display_names = {
        "baseline":           "Baseline",
        "unstructured_50pct": "No estructurado  50%",
        "unstructured_75pct": "No estructurado  75%",
        "structured_50pct":   "Estructurado     50%",
    }
    for k in labels_order:
        r = results[k]
        sp = f"{r['sparsity']:.1%}" if isinstance(r['sparsity'], float) else r['sparsity']
        print(f"{display_names[k]:<30} {r['acc']:>7.4f} {r['params']:>10,} {r['macs']:>12,} {r['size_kb']:>8.1f} {sp:>12}")

    # ------------------------------------------------------------------
    # 5. Gráfico
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labels_plot = ["Baseline", "Unstruct.\n50%", "Unstruct.\n75%", "Struct.\n50%"]
    accs  = [results[k]["acc"]  for k in labels_order]
    macs  = [results[k]["macs"] for k in labels_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#4C72B0", "#DD8452", "#C44E52", "#55A868"]

    ax1.bar(labels_plot, accs, color=colors)
    ax1.axhline(base_acc, color="black", linestyle="--", linewidth=1, label="Baseline acc")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy por estrategia de pruning")
    ax1.set_ylim(min(accs) - 0.05, 1.0)
    ax1.legend()

    ax2.bar(labels_plot, [m / 1e6 for m in macs], color=colors)
    ax2.set_ylabel("MACs (millones)")
    ax2.set_title("MACs por estrategia de pruning")

    fig.suptitle("Pruning no estructurado vs estructurado — Fashion MNIST CNN", fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "pruning_comparacion.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\n[INFO] Gráfico guardado en '{plot_path}'")
    plt.close()

    # Guardar JSON
    save_results(os.path.join(OUTPUT_DIR, "pruning_resultados.json"), results)

    print("\n[CONCLUSIÓN]")
    macs_base   = results["baseline"]["macs"]
    macs_struct = results["structured_50pct"]["macs"]
    macs_unstr  = results["unstructured_75pct"]["macs"]
    print(f"  Baseline:             {macs_base:,} MACs")
    print(f"  Unstruct. 75%:        {macs_unstr:,} MACs  ({macs_unstr/macs_base:.1%} del baseline)")
    print(f"  Estructurado 50% FT:  {macs_struct:,} MACs  ({macs_struct/macs_base:.1%} del baseline)")
    print("  → Eliminar pesos individuales no reduce los MACs.")
    print("  → Eliminar filtros enteros sí reduce los MACs (y el modelo es genuinamente más pequeño).")


if __name__ == "__main__":
    main()
