"""
Sección 8 — Factorización
==========================
Demuestra dos formas de factorización en redes neuronales:

  Parte A — SVD en capas Dense (MLP):
    Descompone W = U Σ Vᵀ y aproxima la capa con dos capas más delgadas.
    Muestra la relación entre el rango de aproximación y la pérdida de accuracy.

  Parte B — Convolución Separable por Profundidad (Depthwise Separable Conv):
    Reemplaza Conv2d(k_in→k_out, 3×3) por DepthwiseConv + PointwiseConv,
    reduciendo los MACs teóricos en un factor de ≈ 1/k_out + 1/9.

Uso:
    cd examples/factorization
    uv run python script01.py
"""

import copy
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
    print_model_stats,
    save_results,
    train,
)

OUTPUT_DIR   = "outputs"
EPOCHS_TRAIN = 10
EPOCHS_FT    = 3
BATCH_SIZE   = 256
LR           = 1e-3


# ---------------------------------------------------------------------------
# Arquitecturas
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """MLP: Flatten → Linear(784→64) → Linear(64→32) → Linear(32→16) → Linear(16→10)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 64), nn.ReLU(),
            nn.Linear(64, 32),  nn.ReLU(),
            nn.Linear(32, 16),  nn.ReLU(),
            nn.Linear(16, 10),
        )
    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """CNN estándar: Conv2d(1→6, 5×5) → MaxPool → Conv2d(6→16, 5×5) → MaxPool → FC."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 120), nn.ReLU(),
            nn.Linear(120, 84),  nn.ReLU(),
            nn.Linear(84, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


class CNNSeparable(nn.Module):
    """
    CNN con convoluciones separables:
    DepthwiseConv(1→1, 5×5) + PointwiseConv(1→6, 1×1) (en lugar de Conv2d(1→6, 5×5))
    DepthwiseConv(6→6, 5×5) + PointwiseConv(6→16, 1×1) (en lugar de Conv2d(6→16, 5×5))
    El resto igual.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Bloque 1: depthwise (1→1, 5×5) + pointwise (1→6, 1×1)
            nn.Conv2d(1, 1,  kernel_size=5, groups=1),  # depthwise (in=1, trivialmente igual)
            nn.Conv2d(1, 6,  kernel_size=1),              # pointwise
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Bloque 2: depthwise (6→6, 5×5) + pointwise (6→16, 1×1)
            nn.Conv2d(6, 6,  kernel_size=5, groups=6),   # depthwise separada
            nn.Conv2d(6, 16, kernel_size=1),              # pointwise
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 120), nn.ReLU(),
            nn.Linear(120, 84),  nn.ReLU(),
            nn.Linear(84, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Parte A — SVD en capas Linear
# ---------------------------------------------------------------------------

def factorize_linear_layer(linear: nn.Linear, rank: int) -> nn.Sequential:
    """
    Factoriza una capa Linear(in→out) en dos capas delgadas usando SVD truncado:
        W ≈ U[:, :rank] @ diag(S[:rank]) @ Vt[:rank, :]
    Retorna nn.Sequential(Linear(in→rank, no bias), Linear(rank→out, con bias)).
    """
    W = linear.weight.data  # shape: (out, in)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    # Truncar
    U_r  = U[:, :rank]
    S_r  = S[:rank]
    Vt_r = Vt[:rank, :]
    # Primera capa: Vt_r  (rank × in)
    # Segunda capa: U_r @ diag(S_r)  (out × rank)
    W1 = Vt_r                               # (rank, in)
    W2 = U_r * S_r.unsqueeze(0)             # (out, rank)

    fc1 = nn.Linear(linear.in_features,  rank, bias=False)
    fc2 = nn.Linear(rank, linear.out_features, bias=linear.bias is not None)
    with torch.no_grad():
        fc1.weight.copy_(W1)
        fc2.weight.copy_(W2)
        if linear.bias is not None:
            fc2.bias.copy_(linear.bias.data)
    return nn.Sequential(fc1, fc2)


def apply_svd_to_mlp(mlp: nn.Module, rank: int) -> nn.Module:
    """Reemplaza todos los nn.Linear internos del MLP por factorizaciones SVD de rango `rank`."""
    mlp_svd = copy.deepcopy(mlp)
    # La red tiene: Flatten, Linear×4 con ReLU entre ellas
    # Iteramos sobre los módulos directos de mlp_svd.net
    new_layers = []
    for layer in mlp_svd.net:
        if isinstance(layer, nn.Linear):
            # Sólo factorizamos si rank < min(in, out)
            if rank < min(layer.in_features, layer.out_features):
                new_layers.append(factorize_linear_layer(layer, rank))
            else:
                new_layers.append(layer)
        else:
            new_layers.append(layer)
    mlp_svd.net = nn.Sequential(*new_layers)
    return mlp_svd


def count_params_svd_mlp(mlp_svd: nn.Module) -> int:
    """Cuenta parámetros de un MLP con bloques SVD (puede tener Sequential anidados)."""
    return sum(p.numel() for p in mlp_svd.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    # ======================================================================
    # PARTE A — SVD en capas Dense del MLP
    # ======================================================================
    print("\n" + "="*60)
    print("PARTE A — SVD en capas Dense del MLP")
    print("="*60)

    mlp = MLP().to(device)
    print(f"Entrenando MLP baseline ({EPOCHS_TRAIN} epochs)...")
    train(mlp, train_loader, test_loader, epochs=EPOCHS_TRAIN, lr=LR, device=device)
    base_acc_mlp = evaluate(mlp, test_loader, device)
    print_model_stats(mlp, "MLP Baseline")

    results["mlp_baseline"] = {
        "acc": round(base_acc_mlp, 4),
        "params": count_parameters(mlp),
        "macs": count_macs(mlp, (1, 1, 28, 28)),
    }

    # Espectro de valores singulares de la primera Linear (784→64)
    W0 = mlp.net[1].weight.data.cpu()  # (64, 784)
    _, S0, _ = torch.linalg.svd(W0, full_matrices=False)
    S0_np = S0.numpy()

    # Experimento SVD para distintos rangos
    ranks = [4, 8, 16, 32]
    svd_accs_before_ft = []
    svd_accs_after_ft  = []

    for rank in ranks:
        mlp_svd = apply_svd_to_mlp(mlp, rank).to(device)
        acc_bf = evaluate(mlp_svd, test_loader, device)
        svd_accs_before_ft.append(acc_bf)
        print(f"  SVD rank={rank:2d}  acc (sin FT)={acc_bf:.4f}  params={count_params_svd_mlp(mlp_svd):,}")

        # Fine-tuning breve
        train(mlp_svd, train_loader, test_loader, epochs=EPOCHS_FT, lr=LR/5, device=device, verbose=False)
        acc_af = evaluate(mlp_svd, test_loader, device)
        svd_accs_after_ft.append(acc_af)
        print(f"           acc (con FT)={acc_af:.4f}")

        results[f"mlp_svd_rank{rank}"] = {
            "acc_no_ft": round(acc_bf, 4),
            "acc_ft":    round(acc_af, 4),
            "params":    count_params_svd_mlp(mlp_svd),
        }

    # ======================================================================
    # PARTE B — Convolución Separable por Profundidad en CNN
    # ======================================================================
    print("\n" + "="*60)
    print("PARTE B — Convolución Separable por Profundidad (CNN)")
    print("="*60)

    # CNN estándar
    cnn = CNN().to(device)
    print(f"Entrenando CNN estándar ({EPOCHS_TRAIN} epochs)...")
    train(cnn, train_loader, test_loader, epochs=EPOCHS_TRAIN, lr=LR, device=device)
    base_acc_cnn = evaluate(cnn, test_loader, device)
    print_model_stats(cnn, "CNN Estándar")

    results["cnn_baseline"] = {
        "acc": round(base_acc_cnn, 4),
        "params": count_parameters(cnn),
        "macs": count_macs(cnn),
    }

    # CNN separable (sin pesos del baseline — entrenada desde cero)
    cnn_sep = CNNSeparable().to(device)
    acc_sep_before = evaluate(cnn_sep, test_loader, device)
    print(f"\nCNN Separable (sin entrenar) acc={acc_sep_before:.4f}")

    print(f"Entrenando CNN Separable ({EPOCHS_TRAIN} epochs)...")
    train(cnn_sep, train_loader, test_loader, epochs=EPOCHS_TRAIN, lr=LR, device=device)
    acc_sep = evaluate(cnn_sep, test_loader, device)
    print_model_stats(cnn_sep, "CNN Separable")

    results["cnn_separable"] = {
        "acc": round(acc_sep, 4),
        "params": count_parameters(cnn_sep),
        "macs": count_macs(cnn_sep),
    }

    macs_std = results["cnn_baseline"]["macs"]
    macs_sep = results["cnn_separable"]["macs"]
    print(f"\n  CNN estándar:   {macs_std:,} MACs  ({count_parameters(cnn):,} params)")
    print(f"  CNN separable:  {macs_sep:,} MACs  ({count_parameters(cnn_sep):,} params)")
    print(f"  Reducción MACs: {1 - macs_sep/macs_std:.1%}")

    # ======================================================================
    # Figuras
    # ======================================================================

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Espectro de valores singulares
    ax = axes[0]
    ax.bar(range(len(S0_np)), S0_np, color="#4C72B0")
    ax.axvline(x=15.5, color="red", linestyle="--", label="rank=16")
    ax.set_xlabel("Índice del valor singular")
    ax.set_ylabel("Valor singular")
    ax.set_title("Espectro SVD: capa Linear(784→64)")
    ax.legend()
    # Anotar energía capturada por top-16
    energy_16 = S0_np[:16].sum() / S0_np.sum()
    ax.text(20, S0_np[0]*0.8, f"Top-16 captura\n{energy_16:.1%} de la energía",
            fontsize=8, color="red")

    # (b) Accuracy SVD vs rango
    ax = axes[1]
    ax.axhline(base_acc_mlp, color="gray", linestyle="--", label="Baseline MLP")
    ax.plot(ranks, svd_accs_before_ft, "o--", label="Sin fine-tuning", color="#DD8452")
    ax.plot(ranks, svd_accs_after_ft,  "s-",  label="Con fine-tuning", color="#55A868")
    ax.set_xlabel("Rango SVD")
    ax.set_ylabel("Accuracy")
    ax.set_title("Parte A: Accuracy vs Rango SVD (MLP)")
    ax.legend()
    ax.set_xticks(ranks)

    # (c) CNN estándar vs separable
    ax = axes[2]
    labels = ["CNN\nEstándar", "CNN\nSeparable"]
    accs_b = [results["cnn_baseline"]["acc"], results["cnn_separable"]["acc"]]
    macs_b = [results["cnn_baseline"]["macs"] / 1e6, results["cnn_separable"]["macs"] / 1e6]
    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, accs_b,  width, label="Accuracy",   color="#4C72B0")
    ax2b  = ax.twinx()
    bars2 = ax2b.bar(x + width/2, macs_b, width, label="MACs (M)",   color="#DD8452", alpha=0.8)
    ax.set_ylabel("Accuracy")
    ax2b.set_ylabel("MACs (millones)")
    ax.set_title("Parte B: CNN Estándar vs Separable")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    fig.suptitle("Factorización: SVD (Dense) + Convolución Separable (CNN)", fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "factorizacion_resultados.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\n[INFO] Gráfico guardado en '{plot_path}'")
    plt.close()

    save_results(os.path.join(OUTPUT_DIR, "factorizacion_resultados.json"), results)

    # ======================================================================
    # Tabla final
    # ======================================================================
    print(f"\n{'='*60}")
    print("TABLA COMPARATIVA")
    print("="*60)
    print(f"{'Modelo':<30} {'Acc':>7} {'Paráms':>10} {'MACs':>12}")
    print("-" * 63)
    def row(name, acc, params, macs):
        print(f"{name:<30} {acc:>7.4f} {params:>10,} {macs:>12,}")

    row("MLP Baseline",         results["mlp_baseline"]["acc"],        results["mlp_baseline"]["params"],        results["mlp_baseline"]["macs"])
    for r in ranks:
        row(f"  MLP SVD rank={r} (FT)", results[f"mlp_svd_rank{r}"]["acc_ft"], results[f"mlp_svd_rank{r}"]["params"],  results["mlp_baseline"]["macs"])
    row("CNN Estándar",         results["cnn_baseline"]["acc"],         results["cnn_baseline"]["params"],         results["cnn_baseline"]["macs"])
    row("CNN Separable",        results["cnn_separable"]["acc"],        results["cnn_separable"]["params"],        results["cnn_separable"]["macs"])

    print(f"\n[CONCLUSIÓN]")
    print(f"  SVD: rango bajo → menos parámetros, pero se recupera con fine-tuning.")
    print(f"  Separable: {1 - macs_sep/macs_std:.1%} menos MACs, accuracy comparable.")
    print(f"  La factorización es más efectiva cuando los pesos tienen estructura de bajo rango.")


if __name__ == "__main__":
    main()
