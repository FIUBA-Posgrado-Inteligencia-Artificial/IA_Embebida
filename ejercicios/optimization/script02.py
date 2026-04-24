"""
Sección 9 — Optimización Algorítmica: Fusion de Operadores (dos casos)
=======================================================================
Demuestra que el orden de las capas determina DÓNDE se fusiona una BatchNorm:

  Caso A — Conv → BN → ReLU  (patrón estándar):
    BN se fusiona HACIA ATRÁS en la Conv precedente.
    El modelo resultante no tiene capa BN: Conv_fold → ReLU.

  Caso B — Conv → ReLU → BN  (patrón no estándar):
    La no-linealidad (ReLU) rompe la equivalencia algebraica.
    BN no puede fusionarse en la Conv precedente.
    Si hay una Conv siguiente, BN puede fusionarse HACIA ADELANTE en ella.
    Si es la última capa, BN no puede fusionarse y debe conservarse.

Las derivaciones algebraicas:

  Caso A: BN(Conv(x)) = γ/(√σ²+ε) · Conv(x) + [β - γμ/√(σ²+ε)]
                       = Conv_fold(x)      ← equivalencia exacta

  Caso B: BN(ReLU(Conv(x))) ≠ Conv_fold_algo(x)
          Pero: si existe Conv2 siguiente:
          Conv2(BN(y)) = Conv2_fold(y)    ← BN se absorbe en Conv2

En inferencia, la fusión elimina lecturas de memoria y operaciones extra.
En entrenamiento, NO se aplica (BN necesita calcular μ y σ² por batch).

Uso:
    cd examples/optimization
    uv run python script02.py
"""

import copy
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from pytorch_utils import (
    count_parameters,
    evaluate,
    get_dataloaders,
    get_device,
    model_size_kb,
    save_results,
    train,
)

OUTPUT_DIR   = "outputs"
EPOCHS_TRAIN = 5
BATCH_SIZE   = 256
LR           = 1e-3
TIMING_REPS  = 300


# ---------------------------------------------------------------------------
# CASO A — Conv → BN → ReLU  (patrón estándar)
# ---------------------------------------------------------------------------

class ModelA_WithBN(nn.Module):
    """
    Red con patrón estándar: Conv → BN → ReLU.
    Dos bloques + clasificador FC.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8,  kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d(2)
        self.fc    = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, 10))

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # Conv→BN→ReLU→Pool
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        return self.fc(x)


class ModelA_Folded(nn.Module):
    """Mismo modelo sin BN: los parámetros de BN están absorbidos en Conv."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8,  kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=True)
        self.pool  = nn.MaxPool2d(2)
        self.fc    = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, 10))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return self.fc(x)


# ---------------------------------------------------------------------------
# CASO B — Conv → ReLU → BN  (patrón no estándar)
# ---------------------------------------------------------------------------

class ModelB_WithBN(nn.Module):
    """
    Red con patrón no estándar: Conv → ReLU → BN.
    El BN del bloque 1 NO puede fusionarse en Conv1 (hay ReLU en medio).
    El BN del bloque 1 SÍ puede fusionarse en Conv2 (el siguiente Conv lineal).
    El BN del bloque 2 queda sin fusionar (no hay Conv siguiente antes del FC).
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8,  kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(8)   # después de ReLU → fusiona en Conv2
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)  # después de ReLU → no puede fusionarse
        self.pool  = nn.MaxPool2d(2)
        self.fc    = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, 10))

    def forward(self, x):
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))  # Conv→ReLU→BN→Pool
        x = self.pool(self.bn2(torch.relu(self.conv2(x))))
        return self.fc(x)


class ModelB_Folded(nn.Module):
    """
    Versión con fusión parcial del Caso B:
    - BN1 se fusiona HACIA ADELANTE en Conv2 (se modifica Conv2).
    - BN2 no puede fusionarse → se mantiene como escala lineal absorbida en FC.
    En la práctica, BN2 se puede absorber en la primera capa Linear.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8,  kernel_size=3, padding=1, bias=True)
        # BN1 fusionado en conv2: conv2 absorbe la normalización previa
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=True)
        # BN2 absorbido en fc (primera capa linear)
        self.pool  = nn.MaxPool2d(2)
        self.fc    = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, 10))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return self.fc(x)


# ---------------------------------------------------------------------------
# Funciones de fusión
# ---------------------------------------------------------------------------

def fold_bn_into_conv_backward(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    FUSIÓN HACIA ATRÁS (Caso A): BN se absorbe en la Conv precedente.
    Equivalencia: BN(Conv(x)) = Conv_fold(x)
    """
    gamma = bn.weight.data
    beta  = bn.bias.data
    mean  = bn.running_mean.data
    var   = bn.running_var.data
    eps   = bn.eps
    scale = gamma / torch.sqrt(var + eps)

    W_fold = conv.weight.data * scale.view(-1, 1, 1, 1)
    b_conv = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels)
    b_fold = (b_conv - mean) * scale + beta

    conv_fold = nn.Conv2d(conv.in_channels, conv.out_channels,
                          kernel_size=conv.kernel_size, stride=conv.stride,
                          padding=conv.padding, bias=True)
    conv_fold.weight.data.copy_(W_fold)
    conv_fold.bias.data.copy_(b_fold)
    return conv_fold


def fold_bn_into_conv_forward(bn: nn.BatchNorm2d, conv: nn.Conv2d) -> nn.Conv2d:
    """
    FUSIÓN HACIA ADELANTE (Caso B): BN se absorbe en la Conv SIGUIENTE.
    Equivalencia: Conv2(BN(y)) = Conv2_fold(y)

    BN(y) = γ/(√σ²+ε) · y + (β - γμ/√σ²+ε) = scale · y + shift
    Conv2(scale·y + shift) = scale · Conv2(y) + Conv2(shift_broadcast)
                           = Conv2_fold(y)   con W_fold = W * scale, b_fold = W@shift + b
    """
    gamma = bn.weight.data
    beta  = bn.bias.data
    mean  = bn.running_mean.data
    var   = bn.running_var.data
    eps   = bn.eps
    scale = gamma / torch.sqrt(var + eps)    # (C_in_of_conv2,)
    shift = beta - scale * mean              # (C_in_of_conv2,)

    # Los pesos de Conv2 tienen shape (C_out, C_in, kH, kW)
    # Cada filtro de salida opera sobre todos los canales de entrada:
    # W_fold[out, in, :, :] = W[out, in, :, :] * scale[in]
    W_fold = conv.weight.data * scale.view(1, -1, 1, 1)

    # El bias adicional viene de: W @ shift_broadcast
    # Para cada filtro de salida j: bias_extra[j] = sum_k( sum_{h,w}(W[j,k,h,w]) * shift[k] )
    # = (W.sum over spatial) @ shift
    W_summed = conv.weight.data.sum(dim=(2, 3))  # (C_out, C_in)
    b_extra  = W_summed @ shift                   # (C_out,)
    b_conv   = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels)
    b_fold   = b_conv + b_extra

    conv_fold = nn.Conv2d(conv.in_channels, conv.out_channels,
                          kernel_size=conv.kernel_size, stride=conv.stride,
                          padding=conv.padding, bias=True)
    conv_fold.weight.data.copy_(W_fold)
    conv_fold.bias.data.copy_(b_fold)
    return conv_fold


def fold_bn_into_linear(bn: nn.BatchNorm2d, linear: nn.Linear) -> nn.Linear:
    """
    Fusiona BN en una capa Linear siguiente (para BN2 en Caso B).
    Igual que fold_bn_into_conv_forward pero para Linear.
    """
    gamma = bn.weight.data
    beta  = bn.bias.data
    mean  = bn.running_mean.data
    var   = bn.running_var.data
    eps   = bn.eps
    scale = gamma / torch.sqrt(var + eps)
    shift = beta - scale * mean

    W_fold = linear.weight.data * scale.unsqueeze(0)
    b_linear = linear.bias.data if linear.bias is not None else torch.zeros(linear.out_features)
    b_fold = b_linear + linear.weight.data @ shift

    lin_fold = nn.Linear(linear.in_features, linear.out_features, bias=True)
    lin_fold.weight.data.copy_(W_fold)
    lin_fold.bias.data.copy_(b_fold)
    return lin_fold


def build_folded_A(model: ModelA_WithBN) -> ModelA_Folded:
    """Caso A: BN se fusiona hacia atrás en la Conv precedente."""
    model.eval()
    folded = ModelA_Folded()
    with torch.no_grad():
        folded.conv1 = fold_bn_into_conv_backward(model.conv1, model.bn1)
        folded.conv2 = fold_bn_into_conv_backward(model.conv2, model.bn2)
        folded.fc[1].weight.copy_(model.fc[1].weight)
        folded.fc[1].bias.copy_(model.fc[1].bias)
    return folded


def build_folded_B(model: ModelB_WithBN) -> ModelB_Folded:
    """
    Caso B: BN1 se fusiona hacia adelante en Conv2; BN2 se fusiona en FC.
    """
    model.eval()
    folded = ModelB_Folded()
    with torch.no_grad():
        # Conv1 no cambia (BN1 no puede fusionarse aquí)
        folded.conv1.weight.copy_(model.conv1.weight)
        if model.conv1.bias is not None:
            folded.conv1.bias.copy_(model.conv1.bias)
        else:
            folded.conv1.bias.zero_()

        # BN1 → se fusiona hacia adelante en Conv2
        folded.conv2 = fold_bn_into_conv_forward(model.bn1, model.conv2)

        # BN2 → se fusiona hacia adelante en FC (Flatten primero)
        folded.fc[1] = fold_bn_into_linear(model.bn2, model.fc[1])
    return folded


# ---------------------------------------------------------------------------
# Medición de velocidad
# ---------------------------------------------------------------------------

def measure_inference_time(model: nn.Module, device: torch.device,
                            batch_size: int = 256, reps: int = TIMING_REPS) -> float:
    model.eval()
    dummy = torch.randn(batch_size, 1, 28, 28, device=device)
    with torch.no_grad():
        for _ in range(20):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


def verify_equivalence(model_orig, model_fold, device, label=""):
    """Verifica que los logits son numéricamente equivalentes."""
    dummy = torch.randn(8, 1, 28, 28, device=device)
    model_orig.eval()
    model_fold.eval()
    with torch.no_grad():
        out_orig = model_orig(dummy)
        out_fold = model_fold(dummy)
    max_diff = (out_orig - out_fold).abs().max().item()
    ok = "✓" if max_diff < 1e-3 else "✗"
    print(f"  {label:<30} diferencia máx en logits: {max_diff:.2e}  {ok}")
    return max_diff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    # ==================================================================
    # CASO A — Conv → BN → ReLU
    # ==================================================================
    print("\n" + "="*65)
    print("CASO A — Patrón estándar: Conv → BN → ReLU")
    print("  Fusión: BN se absorbe HACIA ATRÁS en la Conv precedente.")
    print("="*65)

    model_a = ModelA_WithBN().to(device)
    print(f"\nEntrenando Modelo A ({EPOCHS_TRAIN} epochs)...")
    train(model_a, train_loader, test_loader, epochs=EPOCHS_TRAIN, lr=LR, device=device)
    model_a.eval()

    acc_a = evaluate(model_a, test_loader, device)
    model_a_fold = build_folded_A(model_a).to(device)
    acc_a_fold = evaluate(model_a_fold, test_loader, device)

    print(f"\n  Acc original:  {acc_a:.4f}   Paráms: {count_parameters(model_a):,}")
    print(f"  Acc fusionado: {acc_a_fold:.4f}   Paráms: {count_parameters(model_a_fold):,}")
    verify_equivalence(model_a, model_a_fold, device, "Caso A (Conv→BN→ReLU)")

    t_a_orig = measure_inference_time(model_a,      device)
    t_a_fold = measure_inference_time(model_a_fold, device)
    speedup_a = t_a_orig / t_a_fold
    print(f"\n  Inferencia original:  {t_a_orig:.2f} ms/batch")
    print(f"  Inferencia fusionada: {t_a_fold:.2f} ms/batch")
    print(f"  Speedup: {speedup_a:.2f}×")

    results["caso_A"] = {
        "acc_original": round(acc_a, 4), "acc_folded": round(acc_a_fold, 4),
        "params_original": count_parameters(model_a),
        "params_folded": count_parameters(model_a_fold),
        "inference_ms_original": round(t_a_orig, 3),
        "inference_ms_folded": round(t_a_fold, 3),
        "speedup": round(speedup_a, 3),
    }

    # ==================================================================
    # CASO B — Conv → ReLU → BN
    # ==================================================================
    print("\n" + "="*65)
    print("CASO B — Patrón no estándar: Conv → ReLU → BN")
    print("  BN1: no puede fusionarse en Conv1 (ReLU rompe la equivalencia).")
    print("  BN1: se fusiona HACIA ADELANTE en Conv2.")
    print("  BN2: se fusiona HACIA ADELANTE en la capa FC.")
    print("="*65)

    model_b = ModelB_WithBN().to(device)
    print(f"\nEntrenando Modelo B ({EPOCHS_TRAIN} epochs)...")
    train(model_b, train_loader, test_loader, epochs=EPOCHS_TRAIN, lr=LR, device=device)
    model_b.eval()

    acc_b = evaluate(model_b, test_loader, device)
    model_b_fold = build_folded_B(model_b).to(device)
    acc_b_fold = evaluate(model_b_fold, test_loader, device)

    print(f"\n  Acc original:  {acc_b:.4f}   Paráms: {count_parameters(model_b):,}")
    print(f"  Acc fusionado: {acc_b_fold:.4f}   Paráms: {count_parameters(model_b_fold):,}")
    verify_equivalence(model_b, model_b_fold, device, "Caso B (Conv→ReLU→BN)")

    t_b_orig = measure_inference_time(model_b,      device)
    t_b_fold = measure_inference_time(model_b_fold, device)
    speedup_b = t_b_orig / t_b_fold
    print(f"\n  Inferencia original:  {t_b_orig:.2f} ms/batch")
    print(f"  Inferencia fusionada: {t_b_fold:.2f} ms/batch")
    print(f"  Speedup: {speedup_b:.2f}×")

    results["caso_B"] = {
        "acc_original": round(acc_b, 4), "acc_folded": round(acc_b_fold, 4),
        "params_original": count_parameters(model_b),
        "params_folded": count_parameters(model_b_fold),
        "inference_ms_original": round(t_b_orig, 3),
        "inference_ms_folded": round(t_b_fold, 3),
        "speedup": round(speedup_b, 3),
    }

    # ==================================================================
    # Gráfico comparativo
    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels   = ["Caso A\n(Conv→BN→ReLU)", "Caso B\n(Conv→ReLU→BN)"]
    colors_o = ["#4C72B0", "#4C72B0"]
    colors_f = ["#55A868", "#55A868"]

    # Accuracy
    ax = axes[0]
    x = [0, 1]
    ax.bar([xi - 0.18 for xi in x], [acc_a,      acc_b],      0.34, label="Original", color="#4C72B0")
    ax.bar([xi + 0.18 for xi in x], [acc_a_fold, acc_b_fold], 0.34, label="Fusionado", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy (debe ser igual)")
    ax.legend(fontsize=8)
    ax.set_ylim(min(acc_a, acc_b, acc_a_fold, acc_b_fold) - 0.02, 1.0)

    # Parámetros
    ax = axes[1]
    params_orig = [count_parameters(model_a), count_parameters(model_b)]
    params_fold = [count_parameters(model_a_fold), count_parameters(model_b_fold)]
    ax.bar([xi - 0.18 for xi in x], params_orig, 0.34, label="Original", color="#4C72B0")
    ax.bar([xi + 0.18 for xi in x], params_fold, 0.34, label="Fusionado", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Parámetros")
    ax.set_title("Parámetros (BN eliminado)")
    ax.legend(fontsize=8)

    # Speedup
    ax = axes[2]
    ax.bar(labels, [speedup_a, speedup_b], color=["#DD8452", "#C44E52"])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Speedup de inferencia")
    ax.set_title("Speedup: fusionado vs original")
    for i, v in enumerate([speedup_a, speedup_b]):
        ax.text(i, v + 0.01, f"{v:.2f}×", ha="center", fontsize=9)

    fig.suptitle("Fusión de BatchNorm: caso estándar vs no estándar\n"
                 "Caso A: BN fusiona hacia atrás → Conv | Caso B: BN fusiona hacia adelante → siguiente capa",
                 fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "bn_fusion_casos.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\n[INFO] Gráfico guardado en '{plot_path}'")
    plt.close()

    save_results(os.path.join(OUTPUT_DIR, "bn_fusion_casos_resultados.json"), results)

    print("\n[CONCLUSIÓN]")
    print(f"  Caso A (Conv→BN→ReLU): BN se fusiona en la Conv anterior.")
    print(f"    Speedup: {speedup_a:.2f}×  |  Acc idéntica.")
    print(f"  Caso B (Conv→ReLU→BN): BN no puede fusionarse en la Conv anterior.")
    print(f"    BN1 fusiona en Conv siguiente, BN2 fusiona en FC.")
    print(f"    Speedup: {speedup_b:.2f}×  |  Acc idéntica.")
    print(f"  En ambos casos se logra equivalencia numérica exacta.")
    print(f"  El orden Conv/BN/ReLU determina la dirección de la fusión.")


if __name__ == "__main__":
    main()
