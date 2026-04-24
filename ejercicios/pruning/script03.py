"""
Sección 7 — Pruning: Criterio basado en activaciones
======================================================
Aplica pruning estructurado (a nivel de filtros Conv2d) usando como criterio
la activación media de cada filtro sobre un batch de calibración.

El criterio de activación captura cuánto usa realmente la red cada filtro
durante la inferencia, en lugar de limitarse a la magnitud de los pesos.

Flujo:
  1. Entrenar CNN baseline en Fashion MNIST.
  2. Calibración: pasar un batch por la red y registrar la activación media
     de cada filtro (‖activación‖₁ promediada sobre muestras y posiciones espaciales).
  3. Pruning iterativo: eliminar los filtros con MENOR activación media
     en pasos de 10% hasta el 70%.
  4. Fine-tune breve (2 epochs) después de cada paso.
  5. Comparar directamente con el criterio de magnitud de pesos (script02.py).

Mensaje clave:
  Usar activaciones como criterio de importancia captura mejor qué filtros
  contribuyen a las predicciones reales, dando mejor accuracy que el criterio
  de magnitud de pesos al mismo nivel de sparsity.

Uso:
    cd examples/pruning
    uv run python script03.py
"""

import copy
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

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

OUTPUT_DIR          = "outputs"
EPOCHS_BASELINE     = 10
EPOCHS_FINETUNE     = 2
BATCH_SIZE          = 256
LR                  = 1e-3
CALIBRATION_BATCHES = 5   # batches para estimar activaciones medias
SPARSITY_STEPS      = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


# ---------------------------------------------------------------------------
# Arquitectura (igual que script01/b para comparabilidad)
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
# Criterio de importancia: magnitud de pesos (para comparación)
# ---------------------------------------------------------------------------

def filter_importance_magnitude(conv: nn.Conv2d) -> torch.Tensor:
    """Norma L1 de los pesos por filtro."""
    return conv.weight.data.abs().sum(dim=(1, 2, 3))


# ---------------------------------------------------------------------------
# Criterio de importancia: activaciones medias (calibración)
# ---------------------------------------------------------------------------

def compute_activation_importance(model: CNN, loader, device: torch.device,
                                   n_batches: int = CALIBRATION_BATCHES
                                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pasa `n_batches` por el modelo y calcula la activación media por filtro
    en conv1 y conv2.

    Activación media de filtro k = promedio de |act[k, h, w]| sobre muestras,
    posiciones espaciales (h, w). Equivalente a ‖activación‖₁ / (B·H·W).

    Retorna (imp1, imp2) — tensores de importancia para conv1 y conv2.
    """
    model.eval()
    acts1_sum = torch.zeros(model.conv1.out_channels, device=device)
    acts2_sum = torch.zeros(model.conv2.out_channels, device=device)
    count = 0

    hooks = []
    captured = {}

    def hook1(module, inp, out):
        # out shape: (B, C_out, H, W)
        captured["conv1"] = out.detach()

    def hook2(module, inp, out):
        captured["conv2"] = out.detach()

    hooks.append(model.conv1.register_forward_hook(hook1))
    hooks.append(model.conv2.register_forward_hook(hook2))

    with torch.no_grad():
        for i, (X, _) in enumerate(loader):
            if i >= n_batches:
                break
            X = X.to(device)
            model(X)
            # Activación ANTES de ReLU está en captured; usamos abs para L1
            a1 = captured["conv1"].abs()  # (B, C1, H1, W1)
            a2 = captured["conv2"].abs()  # (B, C2, H2, W2)
            acts1_sum += a1.mean(dim=(0, 2, 3))  # media sobre B, H, W → (C1,)
            acts2_sum += a2.mean(dim=(0, 2, 3))
            count += 1

    for h in hooks:
        h.remove()

    return acts1_sum / count, acts2_sum / count


# ---------------------------------------------------------------------------
# Construcción del modelo podado
# ---------------------------------------------------------------------------

def build_pruned_cnn(base_model: CNN, n_keep1: int, n_keep2: int,
                     keep1: torch.Tensor, keep2: torch.Tensor) -> CNN:
    """Construye una CNN más pequeña conservando sólo los filtros indicados."""
    pruned = CNN(conv1_out=n_keep1, conv2_out=n_keep2)
    with torch.no_grad():
        pruned.conv1.weight.copy_(base_model.conv1.weight.data[keep1])
        if base_model.conv1.bias is not None:
            pruned.conv1.bias.copy_(base_model.conv1.bias.data[keep1])

        pruned.conv2.weight.copy_(base_model.conv2.weight.data[keep2][:, keep1, :, :])
        if base_model.conv2.bias is not None:
            pruned.conv2.bias.copy_(base_model.conv2.bias.data[keep2])

        keep2_flat = torch.cat([keep2 * 16 + i for i in range(16)]).sort().values
        pruned.fc1.weight.copy_(base_model.fc1.weight.data[:, keep2_flat])
        pruned.fc1.bias.copy_(base_model.fc1.bias.data)
        pruned.fc2.weight.copy_(base_model.fc2.weight.data)
        pruned.fc2.bias.copy_(base_model.fc2.bias.data)
        pruned.fc3.weight.copy_(base_model.fc3.weight.data)
        pruned.fc3.bias.copy_(base_model.fc3.bias.data)
    return pruned


def prune_by_importance(model: CNN, importance1: torch.Tensor,
                         importance2: torch.Tensor, sparsity: float) -> CNN:
    """Poda los filtros con menor importancia (cualquier criterio)."""
    n_keep1 = max(1, int(model.conv1.out_channels * (1 - sparsity)))
    keep1 = importance1.topk(n_keep1).indices.sort().values

    n_keep2 = max(1, int(model.conv2.out_channels * (1 - sparsity)))
    keep2 = importance2.topk(n_keep2).indices.sort().values

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

    results = {"baseline": {"acc": round(base_acc, 4),
                             "params": count_parameters(baseline),
                             "macs": count_macs(baseline)}}

    # ------------------------------------------------------------------
    # 2. Calcular importancia por activaciones (calibración)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 2 — Calibración: activación media por filtro")
    print("="*60)
    act_imp1, act_imp2 = compute_activation_importance(baseline, train_loader, device)
    mag_imp1 = filter_importance_magnitude(baseline.conv1)
    mag_imp2 = filter_importance_magnitude(baseline.conv2)
    print(f"  Conv1 ({baseline.conv1.out_channels} filtros) — activación media mín/máx: "
          f"{act_imp1.min():.3f} / {act_imp1.max():.3f}")
    print(f"  Conv2 ({baseline.conv2.out_channels} filtros) — activación media mín/máx: "
          f"{act_imp2.min():.3f} / {act_imp2.max():.3f}")

    # ------------------------------------------------------------------
    # 3. Pruning iterativo: activaciones vs magnitud de pesos
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 3 — Comparación: activaciones vs magnitud de pesos")
    print("="*60)
    print(f"\n  {'Sparsity':>9} | {'Activaciones (FT)':>18} | {'Magnitud (FT)':>14} | {'Δ':>8}")
    print("  " + "-" * 57)

    act_accs, mag_accs, mac_counts = [], [], []

    for sparsity in SPARSITY_STEPS:
        # Pruning por activaciones
        pruned_act = prune_by_importance(baseline, act_imp1, act_imp2, sparsity).to(device)
        train(pruned_act, train_loader, test_loader, epochs=EPOCHS_FINETUNE,
              lr=LR/5, device=device, verbose=False)
        acc_act = evaluate(pruned_act, test_loader, device)

        # Pruning por magnitud
        pruned_mag = prune_by_importance(baseline, mag_imp1, mag_imp2, sparsity).to(device)
        train(pruned_mag, train_loader, test_loader, epochs=EPOCHS_FINETUNE,
              lr=LR/5, device=device, verbose=False)
        acc_mag = evaluate(pruned_mag, test_loader, device)

        macs = count_macs(pruned_act)
        delta = acc_act - acc_mag
        print(f"  {sparsity:>8.0%} | {acc_act:>18.4f} | {acc_mag:>14.4f} | {delta:>+8.4f}")

        act_accs.append(acc_act)
        mag_accs.append(acc_mag)
        mac_counts.append(macs)

        results[f"activation_{int(sparsity*100)}pct"] = {
            "sparsity": sparsity, "acc_ft": round(acc_act, 4),
            "params": count_parameters(pruned_act), "macs": macs,
        }
        results[f"magnitude_{int(sparsity*100)}pct"] = {
            "sparsity": sparsity, "acc_ft": round(acc_mag, 4),
        }

    # ------------------------------------------------------------------
    # 4. Gráficos
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    svals = [s * 100 for s in SPARSITY_STEPS]

    ax = axes[0]
    ax.plot(svals, act_accs, "o-",  color="#55A868", label="Activaciones (este script)", linewidth=2)
    ax.plot(svals, mag_accs, "s--", color="#DD8452", label="Magnitud de pesos (script02)", linewidth=2)
    ax.axhline(base_acc, color="gray", linestyle=":", label=f"Baseline ({base_acc:.4f})")
    ax.set_xlabel("Filtros eliminados (%)")
    ax.set_ylabel("Accuracy (con fine-tuning)")
    ax.set_title("Activaciones vs Magnitud de pesos\ncomo criterio de importancia")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_xticks(svals)

    ax = axes[1]
    deltas = [a - m for a, m in zip(act_accs, mag_accs)]
    colors = ["#55A868" if d >= 0 else "#C44E52" for d in deltas]
    ax.bar(svals, deltas, color=colors, width=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Filtros eliminados (%)")
    ax.set_ylabel("Δ accuracy (Activaciones − Magnitud)")
    ax.set_title("Ventaja del criterio de activaciones\n(positivo = activaciones gana)")
    ax.set_xticks(svals)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Criterio de Pruning: Activaciones vs Magnitud de Pesos", fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "pruning_activaciones.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\n[INFO] Gráfico guardado en '{plot_path}'")
    plt.close()

    save_results(os.path.join(OUTPUT_DIR, "pruning_activaciones_resultados.json"), results)

    print("\n[CONCLUSIÓN]")
    wins = sum(1 for d in deltas if d > 0)
    avg_delta = sum(deltas) / len(deltas)
    print(f"  Criterio de activaciones supera al de magnitud en {wins}/{len(SPARSITY_STEPS)} niveles.")
    print(f"  Ventaja promedio: {avg_delta:+.4f} accuracy.")
    print(f"  Razón: la activación media captura cuánto 'usa' la red cada filtro,")
    print(f"  mientras que la magnitud de los pesos puede ser alta aunque la activación sea baja.")


if __name__ == "__main__":
    main()
