"""
Sección 8 — Factorización: Tucker Decomposition en capas Conv2d (Avanzado)
===========================================================================
Aplica Tucker-2 decomposition a capas convolucionales, descomponiendo el
tensor de pesos W ∈ ℝ^(C_out × C_in × kH × kW) en tres operaciones:

  W ≈ G ×₁ U_out ×₂ U_in

  donde:
    U_out ∈ ℝ^(C_out × R_out)   : factores de salida (top-R_out singular vectors)
    U_in  ∈ ℝ^(C_in  × R_in)    : factores de entrada
    G     ∈ ℝ^(R_out × R_in × kH × kW) : núcleo comprimido

  Implementado como tres Conv2d encadenadas:
    1. Conv2d(C_in,  R_in,  1×1, bias=False)   ← reduce canales de entrada
    2. Conv2d(R_in,  R_out, kH×kW, bias=False) ← convolución espacial comprimida
    3. Conv2d(R_out, C_out, 1×1,  bias=True)   ← restaura canales de salida

  El rango (R_out, R_in) controla el tradeoff precisión/compresión.

La descomposición se calcula con HOSVD (Higher-Order SVD):
  - SVD del unfolding modo-0 → U_out
  - SVD del unfolding modo-1 → U_in
  - Núcleo G = W ×₁ U_out^T ×₂ U_in^T

Flujo:
  1. Entrenar CNN con conv2d estándar.
  2. Para distintos rangos (fracción de canales conservados):
     a. Aplicar Tucker-2 a las dos capas Conv2d.
     b. Evaluar accuracy PTQ (sin fine-tuning).
     c. Fine-tune breve (3 epochs) y evaluar.
     d. Reportar reducción de MACs.

Relación con script01.py:
  - SVD (08a) descompone capas Dense en dos matrices delgadas.
  - Tucker-2 (08b) extiende la misma idea a tensores convolucionales.
  - Depthwise separable (08a) es un caso especial de Tucker-2 con R_in=C_in.

Uso:
    cd examples/factorization
    uv run python script02.py
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

OUTPUT_DIR   = "outputs"
EPOCHS_TRAIN = 10
EPOCHS_FT    = 3
BATCH_SIZE   = 256
LR           = 1e-3

# Rangos a explorar: fracción de canales conservados (R/C)
RANK_RATIOS = [0.25, 0.50, 0.75]


# ---------------------------------------------------------------------------
# Arquitectura CNN con capas Conv2d de interés
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    """
    CNN estándar. Usamos padding=2 en conv1 para tener mapas más grandes y
    poder observar mejor la compresión en los canales.
    Conv(1→16, 3×3) → Pool → Conv(16→32, 3×3) → Pool → FC → FC → salida
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  16, kernel_size=3, padding=1)  # 28→28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 14→14
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # → (B, 16, 14, 14)
        x = self.pool(torch.relu(self.conv2(x)))   # → (B, 32, 7, 7)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Tucker-2 Decomposition
# ---------------------------------------------------------------------------

def tucker2_decompose(conv: nn.Conv2d, rank_out: int, rank_in: int
                      ) -> nn.Sequential:
    """
    Descompone una capa Conv2d en tres capas más pequeñas usando Tucker-2 (HOSVD).

    Args:
        conv:     capa Conv2d original
        rank_out: número de componentes de salida a conservar (R_out ≤ C_out)
        rank_in:  número de componentes de entrada a conservar (R_in ≤ C_in)

    Retorna:
        nn.Sequential con tres Conv2d:
          [0] pointwise_in : (C_in  → R_in,  1×1, no bias)
          [1] spatial      : (R_in  → R_out, kH×kW, no bias)
          [2] pointwise_out: (R_out → C_out, 1×1,  bias=True)
    """
    W = conv.weight.data  # (C_out, C_in, kH, kW)
    C_out, C_in, kH, kW = W.shape
    rank_out = min(rank_out, C_out, C_in * kH * kW)
    rank_in  = min(rank_in,  C_in, C_out * kH * kW)

    # --- HOSVD: modo 0 (output channels) ---
    W0 = W.reshape(C_out, -1)             # (C_out, C_in*kH*kW)
    U_out, _, _ = torch.linalg.svd(W0, full_matrices=False)
    U_out = U_out[:, :rank_out]           # (C_out, R_out)

    # --- HOSVD: modo 1 (input channels) ---
    W1 = W.permute(1, 0, 2, 3).reshape(C_in, -1)  # (C_in, C_out*kH*kW)
    U_in, _, _ = torch.linalg.svd(W1, full_matrices=False)
    U_in = U_in[:, :rank_in]             # (C_in, R_in)

    # --- Núcleo: G = W ×₁ U_out^T ×₂ U_in^T ---
    # Paso 1: contraer modo 0: (R_out, C_in, kH, kW)
    G = torch.einsum("ri,ijhw->rjhw", U_out.T, W)
    # Paso 2: contraer modo 1: (R_out, R_in, kH, kW)
    G = torch.einsum("rjhw,sj->rshw", G, U_in.T)

    # --- Construir las tres capas ---
    pointwise_in = nn.Conv2d(C_in, rank_in, kernel_size=1,
                              stride=conv.stride, bias=False)
    spatial      = nn.Conv2d(rank_in, rank_out, kernel_size=conv.kernel_size,
                              stride=(1, 1), padding=conv.padding, bias=False)
    pointwise_out = nn.Conv2d(rank_out, C_out, kernel_size=1, bias=conv.bias is not None)

    with torch.no_grad():
        # pointwise_in: weight = U_in^T → shape (R_in, C_in, 1, 1)
        pointwise_in.weight.copy_(U_in.T.reshape(rank_in, C_in, 1, 1))
        # spatial: weight = G → shape (R_out, R_in, kH, kW)
        spatial.weight.copy_(G)
        # pointwise_out: weight = U_out → shape (C_out, R_out, 1, 1)
        pointwise_out.weight.copy_(U_out.reshape(C_out, rank_out, 1, 1))
        if conv.bias is not None:
            pointwise_out.bias.copy_(conv.bias.data)

    return nn.Sequential(pointwise_in, spatial, pointwise_out)


def apply_tucker2_to_cnn(model: CNN, ratio: float) -> nn.Module:
    """
    Aplica Tucker-2 a conv1 y conv2 conservando `ratio` fracción de canales.
    Retorna un nuevo módulo con las capas reemplazadas.
    """
    decomposed = copy.deepcopy(model)

    r1_out = max(1, int(model.conv1.out_channels * ratio))
    r1_in  = max(1, int(model.conv1.in_channels  * ratio))
    decomposed.conv1 = tucker2_decompose(model.conv1, r1_out, r1_in)

    r2_out = max(1, int(model.conv2.out_channels * ratio))
    r2_in  = max(1, int(model.conv2.in_channels  * ratio))
    decomposed.conv2 = tucker2_decompose(model.conv2, r2_out, r2_in)

    return decomposed


# ---------------------------------------------------------------------------
# Verificación numérica
# ---------------------------------------------------------------------------

def max_output_diff(model_orig: nn.Module, model_tucker: nn.Module,
                    device: torch.device) -> float:
    """Diferencia máxima en los logits entre el modelo original y el descompuesto."""
    dummy = torch.randn(4, 1, 28, 28, device=device)
    model_orig.eval(); model_tucker.eval()
    with torch.no_grad():
        out_orig   = model_orig(dummy)
        out_tucker = model_tucker(dummy)
    return (out_orig - out_tucker).abs().max().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    # ------------------------------------------------------------------
    # 1. Baseline
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 1 — Entrenando CNN baseline")
    print("="*60)
    baseline = CNN().to(device)
    train(baseline, train_loader, test_loader, epochs=EPOCHS_TRAIN, lr=LR, device=device)
    base_acc = evaluate(baseline, test_loader, device)
    base_macs = count_macs(baseline)
    print(f"\nBaseline  acc={base_acc:.4f}")
    print_model_stats(baseline, "Baseline")

    results["baseline"] = {
        "acc": round(base_acc, 4),
        "params": count_parameters(baseline),
        "macs": base_macs,
    }

    # ------------------------------------------------------------------
    # 2. Experimentos Tucker-2 para distintos rangos
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 2 — Tucker-2 Decomposition para distintos rangos")
    print("  Rank ratio = fracción de canales conservados (R/C)")
    print("="*60)

    print(f"\n  {'Ratio':>6} | {'R_out':>6} {'R_in':>6} | {'Acc PTQ':>8} | {'Acc FT':>8} | {'MACs':>12} | {'Reducción':>10} | {'Diff logits':>12}")
    print("  " + "-"*85)

    accs_ptq, accs_ft, mac_reductions, ratios_done = [], [], [], []

    for ratio in RANK_RATIOS:
        decomp = apply_tucker2_to_cnn(baseline, ratio).to(device)

        diff = max_output_diff(baseline, decomp, device)
        acc_ptq = evaluate(decomp, test_loader, device)

        train(decomp, train_loader, test_loader, epochs=EPOCHS_FT, lr=LR/5,
              device=device, verbose=False)
        acc_ft = evaluate(decomp, test_loader, device)

        d_macs = count_macs(decomp)
        mac_red = 1 - d_macs / base_macs
        # Obtener rangos reales (pueden haber sido limitados por el rango máximo teórico)
        r1_out = decomp.conv1[1].out_channels
        r1_in  = decomp.conv1[1].in_channels

        print(f"  {ratio:>6.2f} | {r1_out:>6} {r1_in:>6} | {acc_ptq:>8.4f} | {acc_ft:>8.4f} | {d_macs:>12,} | {mac_red:>10.1%} | {diff:>12.2e}")

        accs_ptq.append(acc_ptq)
        accs_ft.append(acc_ft)
        mac_reductions.append(mac_red)
        ratios_done.append(ratio)

        results[f"tucker_ratio_{int(ratio*100)}"] = {
            "ratio": ratio, "rank_out": r1_out, "rank_in": r1_in,
            "acc_ptq": round(acc_ptq, 4), "acc_ft": round(acc_ft, 4),
            "macs": d_macs, "mac_reduction": round(mac_red, 4),
            "max_diff_logits": round(diff, 6),
        }

    # ------------------------------------------------------------------
    # 3. Gráficos
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ratios_pct = [r * 100 for r in ratios_done]

    ax = axes[0]
    ax.plot(ratios_pct, accs_ptq, "o--", color="#DD8452", label="Sin fine-tuning (PTQ)", markersize=7)
    ax.plot(ratios_pct, accs_ft,  "s-",  color="#55A868", label=f"Con fine-tuning ({EPOCHS_FT} epochs)", markersize=7)
    ax.axhline(base_acc, color="gray", linestyle=":", label=f"Baseline ({base_acc:.4f})")
    ax.set_xlabel("Rank ratio (%)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Rango Tucker-2")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_xticks(ratios_pct)

    ax = axes[1]
    ax.plot(ratios_pct, [r * 100 for r in mac_reductions], "o-", color="#4C72B0", markersize=7)
    ax.set_xlabel("Rank ratio (%)")
    ax.set_ylabel("Reducción de MACs (%)")
    ax.set_title("Reducción de MACs vs Rango")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_xticks(ratios_pct)
    for rp, mr in zip(ratios_pct, mac_reductions):
        ax.text(rp, mr * 100 + 0.5, f"{mr:.1%}", ha="center", fontsize=8)

    fig.suptitle("Tucker-2 Decomposition en Conv2d — Fashion MNIST CNN\n"
                 "W ≈ G ×₁ U_out ×₂ U_in  →  3 Conv2d más pequeñas", fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "tucker_resultados.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\n[INFO] Gráfico guardado en '{plot_path}'")
    plt.close()

    save_results(os.path.join(OUTPUT_DIR, "tucker_resultados.json"), results)

    print("\n[CONCLUSIÓN]")
    print(f"  Tucker-2 descompone W^(C_out×C_in×kH×kW) en tres Conv2d:")
    print(f"    (C_in→R_in, 1×1) → (R_in→R_out, kH×kW) → (R_out→C_out, 1×1)")
    print(f"  Relación con script01.py:")
    print(f"    - SVD Dense: descompone matrices 2D")
    print(f"    - Tucker-2:  extiende SVD a tensores 4D preservando la estructura espacial")
    print(f"    - Depthwise: caso extremo de Tucker con R_in=C_in, R_out=1")
    print(f"  Con ratio=0.5: {mac_reductions[1]:.1%} menos MACs con fine-tuning de {EPOCHS_FT} epochs.")


if __name__ == "__main__":
    main()
