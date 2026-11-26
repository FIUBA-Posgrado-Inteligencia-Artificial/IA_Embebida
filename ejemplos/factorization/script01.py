"""
Sección 11 — Factorización: SVD en las capas densas de un MLP
==============================================================
Aproxima la matriz de pesos W (m × n) de cada capa densa con la SVD truncada a rango r:

    W ≈ U_r · Σ_r · V_rᵀ

y reemplaza la capa por dos capas más delgadas: Linear(n→r) con V_rᵀ y Linear(r→m) con U_r·Σ_r.

Dos formas de ejecutar una capa factorizada:
  - Dos capas encadenadas (lo que hace este script): se guardan r·(m+n) pesos y se hacen
    r·(m+n) MACs. Ahorra parámetros Y operaciones.
  - Reconstruir W = U_r·Σ_r·V_rᵀ y multiplicar: se guardan los mismos r·(m+n) pesos, pero para
    ejecutar hace falta la matriz entera en RAM y se hacen los m·n MACs del original. Solo ahorra
    almacenamiento (en un microcontrolador, peor: la RAM es más escasa que la flash).

Factorizar conviene solo si r·(m+n) < m·n, es decir, si r < m·n/(m+n) (el rango límite).
Una capa con un rango por encima de su límite queda sin factorizar.

Flujo:
  1. Entrena el MLP baseline.
  2. Muestra el rango límite de cada capa y cuánta energía (Σσ²) junta la SVD de la primera.
  3. Para cada rango: factoriza, evalúa sin fine-tuning y con un fine-tuning corto.
     El baseline recibe el mismo fine-tuning, así la comparación es pareja.
  4. Compara las dos formas de ejecutar para r = 16.

Uso:
    cd examples/factorization
    uv run python script01.py
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

OUTPUT_DIR      = "outputs"
EPOCHS_TRAIN    = 10
EPOCHS_FT       = 3
BATCH_SIZE      = 256
LR              = 1e-3
RANGOS          = [4, 8, 16, 32]
RANGO_EJECUCION = 16
NOMBRES         = ["fc1", "fc2", "fc3", "fc4"]


# ---------------------------------------------------------------------------
# Arquitectura
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """MLP: Flatten → Linear(784→64) → Linear(64→32) → Linear(32→16) → Linear(16→10), con ReLU."""
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


# ---------------------------------------------------------------------------
# SVD
# ---------------------------------------------------------------------------

def rango_limite(m: int, n: int) -> float:
    """Rango a partir del cual factorizar no ahorra nada: r·(m+n) = m·n."""
    return m * n / (m + n)


def factorizar_linear(linear: nn.Linear, rango: int) -> nn.Sequential:
    """
    Reemplaza Linear(n→m) por Linear(n→r, sin bias) + Linear(r→m, con el bias original),
    con los pesos de la SVD truncada: V_rᵀ en la primera y U_r·Σ_r en la segunda.
    """
    W = linear.weight.data                                  # (m, n)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    capa1 = nn.Linear(linear.in_features, rango, bias=False)
    capa2 = nn.Linear(rango, linear.out_features, bias=linear.bias is not None)
    with torch.no_grad():
        capa1.weight.copy_(Vt[:rango, :])                   # (r, n)
        capa2.weight.copy_(U[:, :rango] * S[:rango])        # (m, r)
        if linear.bias is not None:
            capa2.bias.copy_(linear.bias.data)
    return nn.Sequential(capa1, capa2).to(W.device)


def aplicar_svd(mlp: MLP, rango: int) -> nn.Module:
    """Copia el MLP y factoriza cada Linear cuyo rango límite supere `rango`."""
    copia = copy.deepcopy(mlp)
    capas = []
    for capa in copia.net:
        if isinstance(capa, nn.Linear) and rango < rango_limite(capa.out_features, capa.in_features):
            capas.append(factorizar_linear(capa, rango))
        else:
            capas.append(capa)
    copia.net = nn.Sequential(*capas)
    return copia


def lineales(mlp: MLP) -> list[nn.Linear]:
    return [c for c in mlp.net if isinstance(c, nn.Linear)]


def capas_factorizadas(mlp: MLP, rango: int) -> list[str]:
    return [nombre for nombre, c in zip(NOMBRES, lineales(mlp))
            if rango < rango_limite(c.out_features, c.in_features)]


def energia_capturada(S: torch.Tensor, k: int) -> float:
    """Fracción de la energía (Σσ²) que juntan los k valores singulares más grandes."""
    energia = S ** 2
    return float(energia[:k].sum() / energia.sum())


def valores_singulares(linear: nn.Linear) -> torch.Tensor:
    return torch.linalg.svdvals(linear.weight.detach().cpu())


def kb(params: int) -> float:
    """Tamaño de `params` pesos en FP32."""
    return params * 4 / 1024


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_reproducibility(42)
    device = get_device()
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PASO 1 — MLP baseline")
    print("=" * 70)
    mlp = MLP().to(device)
    S_inicial = valores_singulares(mlp.net[1])
    train(mlp, train_loader, test_loader, epochs=EPOCHS_TRAIN, lr=LR, device=device)
    base_acc = evaluate(mlp, test_loader, device)
    base_params = count_parameters(mlp)
    base_macs = count_macs(mlp)
    S_entrenada = valores_singulares(mlp.net[1])

    # El baseline recibe el mismo fine-tuning que los modelos factorizados
    mlp_ft = copy.deepcopy(mlp)
    train(mlp_ft, train_loader, test_loader, epochs=EPOCHS_FT, lr=LR / 5, device=device, verbose=False)
    base_acc_ft = evaluate(mlp_ft, test_loader, device)
    print(f"\n  Baseline: acc={base_acc:.4f}  (+{EPOCHS_FT} épocas de FT: {base_acc_ft:.4f})"
          f"  params={base_params:,}  MACs={base_macs:,}")

    # ------------------------------------------------------------------
    # 2. Rango límite y energía
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PASO 2 — Rango límite de cada capa: factorizar conviene si r < m·n/(m+n)")
    print("=" * 70)
    limites = {}
    for nombre, capa in zip(NOMBRES, lineales(mlp)):
        limites[nombre] = rango_limite(capa.out_features, capa.in_features)
        print(f"  {nombre}  Linear({capa.in_features:>3}→{capa.out_features:<3})  límite = {limites[nombre]:5.1f}")

    print(f"\n  Energía (Σσ²) que juntan los {RANGO_EJECUCION} valores singulares más grandes de fc1 (64 en total):")
    print(f"    pesos iniciales (aleatorios): {energia_capturada(S_inicial, RANGO_EJECUCION):.1%}")
    print(f"    pesos entrenados:             {energia_capturada(S_entrenada, RANGO_EJECUCION):.1%}")

    # ------------------------------------------------------------------
    # 3. SVD para distintos rangos
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"PASO 3 — SVD a rango r, sin fine-tuning y con {EPOCHS_FT} épocas de fine-tuning")
    print("=" * 70)
    resultados_rangos = {}
    for r in RANGOS:
        modelo = aplicar_svd(mlp, r).to(device)
        acc_sin_ft = evaluate(modelo, test_loader, device)
        train(modelo, train_loader, test_loader, epochs=EPOCHS_FT, lr=LR / 5, device=device, verbose=False)
        acc_ft = evaluate(modelo, test_loader, device)
        resultados_rangos[str(r)] = {
            "acc_sin_ft": round(acc_sin_ft, 4),
            "acc_ft": round(acc_ft, 4),
            "params": count_parameters(modelo),
            "macs": count_macs(modelo),
            "capas_factorizadas": capas_factorizadas(mlp, r),
        }
        fact = ", ".join(resultados_rangos[str(r)]["capas_factorizadas"])
        print(f"  r={r:>2}  acc sin FT={acc_sin_ft:.4f}  con FT={acc_ft:.4f}  factoriza: {fact}")

    # ------------------------------------------------------------------
    # 4. Dos formas de ejecutar (r = RANGO_EJECUCION)
    # ------------------------------------------------------------------
    r_ej = resultados_rangos[str(RANGO_EJECUCION)]
    ejecucion = {
        "dos_capas":   {"pesos_kb": round(kb(r_ej["params"]), 1), "ram_pesos_kb": round(kb(r_ej["params"]), 1),
                        "macs": r_ej["macs"]},
        "reconstruir": {"pesos_kb": round(kb(r_ej["params"]), 1), "ram_pesos_kb": round(kb(base_params), 1),
                        "macs": base_macs},
    }

    # ------------------------------------------------------------------
    # Tablas
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TABLA COMPARATIVA")
    print("=" * 70)
    print(f"{'Modelo':<34} {'Acc sin FT':>10} {'Acc':>7} {'Paráms':>8} {'MACs':>8}")
    print("-" * 70)
    print(f"{'MLP baseline':<34} {'—':>10} {base_acc:>7.4f} {base_params:>8,} {base_macs:>8,}")
    print(f"{'MLP baseline + FT':<34} {'—':>10} {base_acc_ft:>7.4f} {base_params:>8,} {base_macs:>8,}")
    for r in RANGOS:
        d = resultados_rangos[str(r)]
        nombre = f"SVD r={r} ({', '.join(d['capas_factorizadas'])}) + FT"
        print(f"{nombre:<34} {d['acc_sin_ft']:>10.4f} {d['acc_ft']:>7.4f} {d['params']:>8,} {d['macs']:>8,}")

    print(f"\nDos formas de ejecutar el modelo con r={RANGO_EJECUCION} (misma accuracy: es la misma función)")
    print(f"{'Forma':<16} {'Pesos guardados':>16} {'Pesos en RAM':>14} {'MACs':>8}")
    print("-" * 57)
    for etiqueta, clave in (("Dos capas", "dos_capas"), ("Reconstruir W", "reconstruir")):
        e = ejecucion[clave]
        print(f"{etiqueta:<16} {e['pesos_kb']:>13.1f} KB {e['ram_pesos_kb']:>11.1f} KB {e['macs']:>8,}")

    print("\n[CONCLUSIÓN]")
    print(f"  Como dos capas, la SVD reduce parámetros Y MACs mientras r < m·n/(m+n):")
    print(f"  con r={RANGO_EJECUCION}, {1 - r_ej['macs'] / base_macs:.0%} menos MACs y "
          f"{r_ej['acc_ft']:.4f} de accuracy (baseline + FT: {base_acc_ft:.4f}).")
    print("  Reconstruir W solo ahorra almacenamiento: la RAM y los MACs son los del original.")

    # ------------------------------------------------------------------
    # Figura y JSON
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    k = torch.arange(1, len(S_entrenada) + 1)
    for S, etiqueta, color in ((S_entrenada, "Pesos entrenados", "#4C72B0"),
                               (S_inicial, "Pesos iniciales", "#999999")):
        acumulada = torch.cumsum(S ** 2, 0) / (S ** 2).sum()
        ax.plot(k, acumulada, label=etiqueta, color=color)
    ax.axvline(RANGO_EJECUCION, color="red", linestyle="--", label=f"r = {RANGO_EJECUCION}")
    ax.axvline(limites["fc1"], color="black", linestyle=":", label=f"límite ≈ {limites['fc1']:.0f}")
    ax.set_xlabel("Cantidad de valores singulares (k)")
    ax.set_ylabel("Energía acumulada (Σσ²)")
    ax.set_title("fc1: Linear(784→64)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhline(base_acc_ft, color="gray", linestyle="--", label="Baseline + FT")
    ax.plot(RANGOS, [resultados_rangos[str(r)]["acc_sin_ft"] for r in RANGOS], "o--", label="Sin FT")
    ax.plot(RANGOS, [resultados_rangos[str(r)]["acc_ft"] for r in RANGOS], "s-", label="Con FT")
    ax.set_xlabel("Rango r")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(RANGOS)
    ax.set_title("Accuracy vs. rango")
    ax.legend(fontsize=8)
    fig.suptitle("SVD en las capas densas del MLP")
    fig.tight_layout()
    ruta_png = os.path.join(OUTPUT_DIR, "svd_resultados.png")
    fig.savefig(ruta_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[INFO] Gráfico guardado en '{ruta_png}'")

    save_results(os.path.join(OUTPUT_DIR, "svd_resultados.json"), {
        "baseline": {"acc": round(base_acc, 4), "acc_ft": round(base_acc_ft, 4),
                     "params": base_params, "macs": base_macs},
        "rango_limite": {n: round(v, 2) for n, v in limites.items()},
        "espectro": {"inicial": [round(float(s), 5) for s in S_inicial],
                     "entrenada": [round(float(s), 5) for s in S_entrenada]},
        "rangos": resultados_rangos,
        f"ejecucion_r{RANGO_EJECUCION}": ejecucion,
    })


if __name__ == "__main__":
    main()
