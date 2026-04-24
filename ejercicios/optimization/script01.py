"""
Sección 9 — Optimización Algorítmica: BatchNorm Folding
=========================================================
Demuestra que el plegado (folding) de BatchNorm en una capa Conv2d precedente
es una optimización puramente algorítmica:
  - Cero pérdida de accuracy (las salidas son matemáticamente equivalentes).
  - Reducción real de parámetros (se eliminan los 4 tensores de BN: γ, β, μ, σ²).
  - Reducción de operaciones (se elimina el paso BN en inferencia).
  - Aceleración medible en inferencia.

La operación Conv → BN es equivalente a una única Conv con pesos y bias modificados:
    W_fold = W * (γ / √(σ² + ε))
    b_fold = (b - μ) * (γ / √(σ² + ε)) + β

Uso:
    cd examples/optimization
    uv run python script01.py
"""

import copy
import os
import time

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

OUTPUT_DIR   = "outputs"
EPOCHS_TRAIN = 5
BATCH_SIZE   = 256
LR           = 1e-3
TIMING_REPS  = 200   # repeticiones para medir tiempos de inferencia


# ---------------------------------------------------------------------------
# Arquitectura con BatchNorm explícito
# ---------------------------------------------------------------------------

class CNNWithBN(nn.Module):
    """
    CNN pequeña con BatchNorm explícito después de cada Conv.
    Conv2d → BN → ReLU → MaxPool (×2) → Flatten → Linear → Linear.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 28×28 → 28×28
        self.bn1   = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # 14×14 → 14×14
        self.bn2   = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(16 * 7 * 7, 64)
        self.fc2   = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # 28→14
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # 14→7
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class CNNFolded(nn.Module):
    """
    Misma arquitectura que CNNWithBN pero SIN capas BatchNorm.
    Los parámetros de BN se absorben en los Conv2d mediante BN folding.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8,  kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=True)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(16 * 7 * 7, 64)
        self.fc2   = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# BN Folding
# ---------------------------------------------------------------------------

def fold_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Crea un nuevo Conv2d con bias=True cuyos pesos y bias absorben la BN.

    Derivación:
        BN(Conv(x)) = γ * (Conv(x) - μ) / √(σ²+ε) + β
                    = (γ/√(σ²+ε)) * Conv(x) + [β - γ*μ/√(σ²+ε)]
                    = Conv_fold(x)   donde:
                        W_fold = W * scale          (scale por canal de salida)
                        b_fold = (b_conv - μ) * scale + β
    """
    assert bn.running_mean is not None, "BN debe estar en modo eval con estadísticas calculadas."

    gamma = bn.weight.data                 # (C_out,)
    beta  = bn.bias.data                   # (C_out,)
    mean  = bn.running_mean.data           # (C_out,)
    var   = bn.running_var.data            # (C_out,)
    eps   = bn.eps

    scale = gamma / torch.sqrt(var + eps)  # (C_out,)

    # Pesos: escalar cada filtro de salida por su scale
    # conv.weight shape: (C_out, C_in, kH, kW)
    W_fold = conv.weight.data * scale.view(-1, 1, 1, 1)

    # Bias del conv original (puede ser None)
    b_conv = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels)
    b_fold = (b_conv - mean) * scale + beta

    # Construir nuevo Conv2d con bias
    conv_fold = nn.Conv2d(
        conv.in_channels, conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True,
    )
    conv_fold.weight.data.copy_(W_fold)
    conv_fold.bias.data.copy_(b_fold)
    return conv_fold


def build_folded_model(model_with_bn: CNNWithBN) -> CNNFolded:
    """
    Construye un CNNFolded copiando los pesos de FC y plegando los BN.
    Pone el modelo en eval() antes de plegar (necesario para estadísticas de BN).
    """
    model_with_bn.eval()
    folded = CNNFolded()

    with torch.no_grad():
        folded.conv1 = fold_bn_into_conv(model_with_bn.conv1, model_with_bn.bn1)
        folded.conv2 = fold_bn_into_conv(model_with_bn.conv2, model_with_bn.bn2)
        folded.fc1.weight.copy_(model_with_bn.fc1.weight)
        folded.fc1.bias.copy_(model_with_bn.fc1.bias)
        folded.fc2.weight.copy_(model_with_bn.fc2.weight)
        folded.fc2.bias.copy_(model_with_bn.fc2.bias)

    return folded


# ---------------------------------------------------------------------------
# Medición de velocidad de inferencia
# ---------------------------------------------------------------------------

def measure_inference_time(model: nn.Module, device: torch.device,
                            batch_size: int = 256, reps: int = TIMING_REPS) -> float:
    """
    Mide el tiempo promedio de inferencia (ms) sobre un batch de tamaño `batch_size`.
    Hace un warm-up de 10 iteraciones antes de medir.
    """
    model.eval()
    dummy = torch.randn(batch_size, 1, 28, 28, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - t0) / reps * 1000
    return elapsed_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Entrenar CNN con BatchNorm
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 1 — Entrenando CNN con BatchNorm")
    print("="*60)
    model_bn = CNNWithBN().to(device)
    train(model_bn, train_loader, test_loader, epochs=EPOCHS_TRAIN, lr=LR, device=device)
    model_bn.eval()

    acc_bn = evaluate(model_bn, test_loader, device)
    print(f"\nAcc con BN: {acc_bn:.4f}")
    print_model_stats(model_bn, "Con BatchNorm", input_size=(1, 1, 28, 28))

    # ------------------------------------------------------------------
    # 2. Plegar BatchNorm
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 2 — Plegando BatchNorm (BN Folding)")
    print("="*60)
    model_folded = build_folded_model(model_bn).to(device)
    model_folded.eval()

    acc_folded = evaluate(model_folded, test_loader, device)
    print(f"\nAcc plegado: {acc_folded:.4f}")
    print_model_stats(model_folded, "BN Plegado", input_size=(1, 1, 28, 28))

    diff = abs(acc_bn - acc_folded)
    print(f"\nDiferencia de accuracy: {diff:.6f}  {'✓ (equivalencia numérica)' if diff < 1e-4 else '⚠ diferencia inesperada'}")

    # ------------------------------------------------------------------
    # 3. Verificación numérica directa
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 3 — Verificación numérica")
    print("="*60)
    dummy_input = torch.randn(4, 1, 28, 28, device=device)
    with torch.no_grad():
        out_bn     = model_bn(dummy_input)
        out_folded = model_folded(dummy_input)
    max_diff = (out_bn - out_folded).abs().max().item()
    print(f"Diferencia máxima en logits: {max_diff:.2e}  (esperado < 1e-4)")

    # ------------------------------------------------------------------
    # 4. Comparación de complejidad
    # ------------------------------------------------------------------
    params_bn     = count_parameters(model_bn)
    params_folded = count_parameters(model_folded)
    macs_bn       = count_macs(model_bn,     (1, 1, 28, 28))
    macs_folded   = count_macs(model_folded, (1, 1, 28, 28))

    print("\n" + "="*60)
    print("COMPARACIÓN DE COMPLEJIDAD")
    print("="*60)
    print(f"{'Modelo':<20} {'Parámetros':>12} {'MACs':>12} {'KB':>8}")
    print("-" * 56)
    print(f"{'Con BatchNorm':<20} {params_bn:>12,} {macs_bn:>12,} {model_size_kb(model_bn):>8.1f}")
    print(f"{'BN Plegado':<20} {params_folded:>12,} {macs_folded:>12,} {model_size_kb(model_folded):>8.1f}")
    print(f"{'Reducción':<20} {(1-params_folded/params_bn):>11.1%}  {'(mismo)':>12}  {(1-model_size_kb(model_folded)/model_size_kb(model_bn)):>7.1%}")

    # ------------------------------------------------------------------
    # 5. Medición de velocidad de inferencia
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("MEDICIÓN DE VELOCIDAD DE INFERENCIA")
    print("="*60)
    t_bn     = measure_inference_time(model_bn,     device)
    t_folded = measure_inference_time(model_folded, device)
    speedup  = t_bn / t_folded
    print(f"  Con BN:    {t_bn:.2f} ms / batch")
    print(f"  Plegado:   {t_folded:.2f} ms / batch")
    print(f"  Speedup:   {speedup:.2f}×")

    # ------------------------------------------------------------------
    # 6. Gráficos
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # (a) Accuracy
    ax = axes[0]
    ax.bar(["Con BN", "BN Plegado"], [acc_bn, acc_folded], color=["#4C72B0", "#55A868"])
    ax.set_ylim(max(0, min(acc_bn, acc_folded) - 0.02), 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy (igual)")
    for i, v in enumerate([acc_bn, acc_folded]):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)

    # (b) Parámetros
    ax = axes[1]
    ax.bar(["Con BN", "BN Plegado"], [params_bn, params_folded], color=["#4C72B0", "#55A868"])
    ax.set_ylabel("Parámetros")
    ax.set_title("Parámetros (BN eliminado)")
    for i, v in enumerate([params_bn, params_folded]):
        ax.text(i, v * 1.01, f"{v:,}", ha="center", fontsize=8)

    # (c) Tiempo de inferencia
    ax = axes[2]
    ax.bar(["Con BN", "BN Plegado"], [t_bn, t_folded], color=["#4C72B0", "#55A868"])
    ax.set_ylabel("Tiempo (ms / batch)")
    ax.set_title(f"Inferencia (speedup {speedup:.2f}×)")
    for i, v in enumerate([t_bn, t_folded]):
        ax.text(i, v * 1.01, f"{v:.2f} ms", ha="center", fontsize=9)

    fig.suptitle("BN Folding: misma accuracy, menos parámetros, más rápido", fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "bn_folding_resultados.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\n[INFO] Gráfico guardado en '{plot_path}'")
    plt.close()

    # Guardar resultados
    save_results(os.path.join(OUTPUT_DIR, "bn_folding_resultados.json"), {
        "acc_bn":             round(acc_bn, 4),
        "acc_folded":         round(acc_folded, 4),
        "params_bn":          params_bn,
        "params_folded":      params_folded,
        "macs_bn":            macs_bn,
        "macs_folded":        macs_folded,
        "inference_ms_bn":    round(t_bn, 3),
        "inference_ms_fold":  round(t_folded, 3),
        "speedup":            round(speedup, 3),
    })

    print("\n[CONCLUSIÓN]")
    print(f"  BN folding absorbe γ, β, μ, σ² en los pesos del Conv.")
    print(f"  Resultado: {1-params_folded/params_bn:.1%} menos parámetros, {speedup:.2f}× más rápido,")
    print(f"  con diferencia de accuracy de {diff:.2e} (numéricamente equivalente).")
    print(f"  TFLite y TorchScript aplican esta optimización automáticamente al exportar.")


if __name__ == "__main__":
    main()
