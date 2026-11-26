"""
Sección 11 — Factorización: convoluciones
==========================================
Compara cuatro formas de armar las convoluciones de una misma CNN, con dos preguntas:
¿qué dimensión del kernel se factoriza? y ¿se factoriza después de entrenar o al diseñar?

  - Estándar: Conv2d(C_in→C_out, k×k). Es el baseline.
  - Tucker-2 (después de entrenar): descompone el kernel ya entrenado W (C_out × C_in × k × k)
    en los canales de entrada y de salida, con HOSVD:
        W ≈ G ×₁ U_out ×₂ U_in
    y lo ejecuta como tres convs: 1×1 (C_in→R_in) → k×k (R_in→R_out) → 1×1 (R_out→C_out).
    Necesita un fine-tuning corto para recuperar la accuracy.
  - Depthwise separable (al diseñar, como MobileNet): una conv k×k por canal (groups=C_in)
    seguida de una 1×1 que mezcla canales. El kernel equivalente es
        W[o, i, :, :] = P[o, i] · D[i, :, :]
    es decir, rango 1 por canal de entrada. Equivale a un Tucker-2 con U_in = I, R_out = C_in y un
    núcleo diagonal en los canales (no a uno con R_out = 1), pero no sale de truncar la HOSVD de una
    conv entrenada: se entrena así desde el principio.
    MACs: H·W·C_in·(k² + C_out) en lugar de H·W·C_out·C_in·k², una fracción 1/C_out + 1/k².
  - Espacial k×1 + 1×k (al diseñar, como Inception-v3): factoriza el kernel en el espacio.
    Con C_in canales intermedios, la fracción de MACs es (C_in + C_out)/(C_out·k).

En depthwise y espacial no hay ReLU entre las dos convs, para que el bloque sea exactamente una conv
factorizada (MobileNet e Inception ponen BatchNorm + ReLU ahí). La primera conv queda estándar en
todas las variantes: con 1 canal de entrada no hay nada que separar ni reducir.

Comparación pareja: todos los modelos se entrenan EPOCHS_TRAIN épocas con LR y después
EPOCHS_FT épocas con LR/5. Tucker-2 parte del baseline ya entrenado y hace solo el fine-tuning;
el baseline hace el mismo fine-tuning ("Estándar + FT").

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
    save_results,
    set_reproducibility,
    train,
)

OUTPUT_DIR   = "outputs"
EPOCHS_TRAIN = 10
EPOCHS_FT    = 3
BATCH_SIZE   = 256
LR           = 1e-3
RATIOS       = [0.25, 0.50, 0.75]   # fracción de canales que conserva Tucker-2 (R/C)


# ---------------------------------------------------------------------------
# Arquitecturas
# ---------------------------------------------------------------------------

class CNNBase(nn.Module):
    """
    Conv(1→16, 3×3) → pool → Conv(16→32, 3×3) → pool → Conv(32→64, 3×3) → pool → Linear(576→10).
    Las convoluciones tienen el 80 % de los parámetros y casi todos los MACs.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)    # 28×28
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)   # 14×14
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)   # 7×7
        self.pool  = nn.MaxPool2d(2)
        self.fc    = nn.Linear(64 * 3 * 3, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))      # 64×3×3
        return self.fc(x.flatten(1))


def depthwise_separable(c_in: int, c_out: int, k: int = 3) -> nn.Sequential:
    """Conv k×k por canal (groups=c_in) + conv 1×1 que mezcla canales."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_in, k, padding=k // 2, groups=c_in, bias=False),
        nn.Conv2d(c_in, c_out, 1),
    )


def espacial_separable(c_in: int, c_out: int, k: int = 3) -> nn.Sequential:
    """Conv k×1 (c_in→c_in) + conv 1×k (c_in→c_out)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_in, (k, 1), padding=(k // 2, 0), bias=False),
        nn.Conv2d(c_in, c_out, (1, k), padding=(0, k // 2)),
    )


class CNNDepthwise(CNNBase):
    """CNNBase con conv2 y conv3 depthwise separables (MobileNet)."""
    def __init__(self):
        super().__init__()
        self.conv2 = depthwise_separable(16, 32)
        self.conv3 = depthwise_separable(32, 64)


class CNNEspacial(CNNBase):
    """CNNBase con conv2 y conv3 separadas en 3×1 + 1×3 (Inception-v3)."""
    def __init__(self):
        super().__init__()
        self.conv2 = espacial_separable(16, 32)
        self.conv3 = espacial_separable(32, 64)


# ---------------------------------------------------------------------------
# Tucker-2
# ---------------------------------------------------------------------------

def tucker2(conv: nn.Conv2d, rango_out: int, rango_in: int) -> nn.Sequential:
    """
    Descompone una Conv2d con HOSVD en tres convs:
      1×1 (C_in→R_in, U_inᵀ) → k×k (R_in→R_out, núcleo G) → 1×1 (R_out→C_out, U_out).
    La k×k lleva el stride y el padding originales; la última, el bias original.
    """
    W = conv.weight.data                                   # (C_out, C_in, kH, kW)
    c_out, c_in, kh, kw = W.shape
    rango_out = max(1, min(rango_out, c_out, c_in * kh * kw))
    rango_in  = max(1, min(rango_in,  c_in,  c_out * kh * kw))

    # Vectores singulares de cada "desplegado" (unfolding) del tensor
    U_out = torch.linalg.svd(W.reshape(c_out, -1), full_matrices=False)[0][:, :rango_out]
    U_in  = torch.linalg.svd(W.permute(1, 0, 2, 3).reshape(c_in, -1), full_matrices=False)[0][:, :rango_in]

    # Núcleo: G = W ×₁ U_outᵀ ×₂ U_inᵀ → (R_out, R_in, kH, kW)
    G = torch.einsum("oihw,or,is->rshw", W, U_out, U_in)

    entrada = nn.Conv2d(c_in, rango_in, 1, bias=False)
    nucleo  = nn.Conv2d(rango_in, rango_out, (kh, kw), stride=conv.stride, padding=conv.padding,
                        dilation=conv.dilation, bias=False)
    salida  = nn.Conv2d(rango_out, c_out, 1, bias=conv.bias is not None)
    with torch.no_grad():
        entrada.weight.copy_(U_in.T.reshape(rango_in, c_in, 1, 1))
        nucleo.weight.copy_(G)
        salida.weight.copy_(U_out.reshape(c_out, rango_out, 1, 1))
        if conv.bias is not None:
            salida.bias.copy_(conv.bias.data)
    return nn.Sequential(entrada, nucleo, salida).to(W.device)


def aplicar_tucker2(modelo: CNNBase, ratio: float) -> nn.Module:
    """Copia el modelo y aplica Tucker-2 a conv2 y conv3 conservando `ratio` de los canales."""
    copia = copy.deepcopy(modelo)
    for nombre in ("conv2", "conv3"):
        conv = getattr(modelo, nombre)
        r_out = max(1, int(conv.out_channels * ratio))
        r_in  = max(1, int(conv.in_channels * ratio))
        setattr(copia, nombre, tucker2(conv, r_out, r_in))
    return copia


def contar_capas(modelo: nn.Module) -> int:
    """Capas con pesos (Conv2d y Linear): cada una suma metadatos, bias y un buffer de salida."""
    return sum(isinstance(m, (nn.Conv2d, nn.Linear)) for m in modelo.modules())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_reproducibility(42)
    device = get_device()
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    modelos = []

    def fila(nombre, cuando, modelo, acc, **extra):
        modelos.append({"nombre": nombre, "cuando": cuando, "params": count_parameters(modelo),
                        "macs": count_macs(modelo), "capas": contar_capas(modelo),
                        "acc": round(acc, 4), **extra})

    def fine_tuning(modelo):
        train(modelo, train_loader, test_loader, epochs=EPOCHS_FT, lr=LR / 5, device=device, verbose=False)

    # ------------------------------------------------------------------
    # 1. Baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PASO 1 — CNN estándar (baseline)")
    print("=" * 70)
    baseline = CNNBase().to(device)
    train(baseline, train_loader, test_loader, epochs=EPOCHS_TRAIN, lr=LR, device=device)
    fila("Estándar", "—", baseline, evaluate(baseline, test_loader, device))
    baseline_ft = copy.deepcopy(baseline)
    fine_tuning(baseline_ft)
    fila("Estándar + FT", "—", baseline_ft, evaluate(baseline_ft, test_loader, device))

    # ------------------------------------------------------------------
    # 2. Tucker-2 sobre el baseline entrenado
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"PASO 2 — Tucker-2 en conv2 y conv3, sin FT y con {EPOCHS_FT} épocas de FT")
    print("=" * 70)
    for ratio in RATIOS:
        modelo = aplicar_tucker2(baseline, ratio).to(device)
        acc_sin_ft = evaluate(modelo, test_loader, device)
        fine_tuning(modelo)
        rangos = {n: [getattr(modelo, n)[1].in_channels, getattr(modelo, n)[1].out_channels]
                  for n in ("conv2", "conv3")}
        fila(f"Tucker-2 ({ratio:.0%})", "después de entrenar", modelo, evaluate(modelo, test_loader, device),
             acc_sin_ft=round(acc_sin_ft, 4), ratio=ratio, rangos=rangos)
        print(f"  ratio={ratio:.0%}  (R_in, R_out) conv2={tuple(rangos['conv2'])} conv3={tuple(rangos['conv3'])}"
              f"  acc sin FT={acc_sin_ft:.4f}  con FT={modelos[-1]['acc']:.4f}")

    # ------------------------------------------------------------------
    # 3. Factorización al diseñar: se entrena desde cero
    # ------------------------------------------------------------------
    for nombre, clase in (("Depthwise separable", CNNDepthwise), ("Espacial 3×1 + 1×3", CNNEspacial)):
        print("\n" + "=" * 70)
        print(f"PASO 3 — {nombre} (al diseñar): {EPOCHS_TRAIN} épocas + {EPOCHS_FT} de FT")
        print("=" * 70)
        modelo = clase().to(device)
        train(modelo, train_loader, test_loader, epochs=EPOCHS_TRAIN, lr=LR, device=device)
        fine_tuning(modelo)
        fila(nombre, "diseño", modelo, evaluate(modelo, test_loader, device))

    # ------------------------------------------------------------------
    # Tabla
    # ------------------------------------------------------------------
    base_macs = modelos[0]["macs"]
    print("\n" + "=" * 92)
    print("TABLA COMPARATIVA")
    print("=" * 92)
    print(f"{'Modelo':<22} {'Cuándo':<20} {'Paráms':>8} {'MACs':>11} {'% MACs':>7} {'Capas':>6} "
          f"{'Acc sin FT':>10} {'Acc':>7}")
    print("-" * 92)
    for m in modelos:
        sin_ft = f"{m['acc_sin_ft']:.4f}" if "acc_sin_ft" in m else "—"
        print(f"{m['nombre']:<22} {m['cuando']:<20} {m['params']:>8,} {m['macs']:>11,} "
              f"{m['macs'] / base_macs:>7.0%} {m['capas']:>6} {sin_ft:>10} {m['acc']:>7.4f}")

    print("\n[CONCLUSIÓN]")
    print("  Tucker-2 comprime un modelo ya entrenado, pero sin fine-tuning pierde accuracy.")
    print("  Depthwise y espacial se eligen al diseñar la red y se entrenan desde cero.")
    print("  Todas reducen MACs y parámetros, pero agregan capas: más metadatos, bias y buffers")
    print("  intermedios, y una recuantización más cuando el modelo se pasa a int8.")

    # ------------------------------------------------------------------
    # Figura y JSON
    # ------------------------------------------------------------------
    colores = {"—": "gray", "después de entrenar": "#DD8452", "diseño": "#55A868"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in modelos:
        ax.scatter(m["macs"] / 1e6, m["acc"], color=colores[m["cuando"]], s=50)
        ax.annotate(m["nombre"], (m["macs"] / 1e6, m["acc"]), textcoords="offset points", xytext=(5, 4),
                    fontsize=8)
    ax.set_xlabel("MACs (millones)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Convoluciones factorizadas: accuracy vs. MACs")
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    ruta_png = os.path.join(OUTPUT_DIR, "convs_resultados.png")
    fig.savefig(ruta_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[INFO] Gráfico guardado en '{ruta_png}'")

    save_results(os.path.join(OUTPUT_DIR, "convs_resultados.json"), {"modelos": modelos})


if __name__ == "__main__":
    main()
