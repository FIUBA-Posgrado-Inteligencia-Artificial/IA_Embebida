"""
Sección 6 — Cuantización: Visualización de Resultados
======================================================
Lee el JSON producido por script03.py (formato barrido 3 pasos) y genera:
  - Una figura con 3 subplots (uno por paso): curvas PTQ y QAT vs. bits.
  - Una figura resumen con los 3 pasos superpuestos (sólo la mejor métrica).

También es capaz de leer el formato antiguo de script03 (6 estrategias) para
retrocompatibilidad.

Uso:
    cd examples/quantization && uv run python script04.py [path/al/json]
"""

import os
import sys
import json

import matplotlib.pyplot as plt
import numpy as np


COLORS_PTQ = ["#4C72B0", "#DD8452", "#55A868"]
COLORS_QAT = ["#8be9fd", "#ffb86c", "#50fa7b"]


# ---------------------------------------------------------------------------
# Formato nuevo (barrido 3 pasos)
# ---------------------------------------------------------------------------

def plot_new_format(data: dict, output_dir: str):
    model_type  = data.get("model_type", "model")
    float_acc   = data.get("float_acc", 0.0)
    steps_meta  = data.get("steps", [])
    results     = data.get("results", {})
    run_qat     = data.get("run_qat", False)

    # --- 3 subplots: uno por paso ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, step_meta, c_ptq, c_qat in zip(axes, steps_meta, COLORS_PTQ, COLORS_QAT):
        paso_id = step_meta["id"]
        label   = step_meta["label"]
        paso_data = results.get(paso_id, {})

        bits_list, ptq_list, qat_list = [], [], []
        for bits_str, res in paso_data.items():
            if "error" in res:
                continue
            bits_list.append(res["bits"])
            ptq_list.append(res["ptq_acc"])
            qat_list.append(res.get("qat_acc", 0.0))

        idx   = np.argsort(bits_list)
        bits  = np.array(bits_list)[idx]
        ptqs  = np.array(ptq_list)[idx]
        qats  = np.array(qat_list)[idx]

        ax.plot(bits, ptqs, "x--", color=c_ptq, label="PTQ", linewidth=1.5, markersize=7)
        if run_qat:
            ax.plot(bits, qats, "o-",  color=c_qat, label="QAT", linewidth=2, markersize=7)
        ax.axhline(float_acc, color="gray", linestyle=":", linewidth=1.5,
                   label=f"Float32 ({float_acc:.4f})")

        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Bits de cuantización")
        ax.set_xticks(bits)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Accuracy")
    fig.suptitle(f"Cuantización en 3 Pasos vs Bitwidth — Modelo: {model_type.upper()}", fontsize=11)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{model_type}_barrido_3pasos.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"[INFO] Gráfico guardado en '{plot_path}'")
    plt.close()

    # --- Figura resumen: 3 pasos superpuestos ---
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for step_meta, c_ptq in zip(steps_meta, COLORS_PTQ):
        paso_id   = step_meta["id"]
        label     = step_meta["label"]
        paso_data = results.get(paso_id, {})

        bits_list, acc_list = [], []
        for bits_str, res in paso_data.items():
            if "error" in res:
                continue
            bits_list.append(res["bits"])
            best = res.get("qat_acc", 0.0) if run_qat else res.get("ptq_acc", 0.0)
            acc_list.append(best)

        idx  = np.argsort(bits_list)
        bits = np.array(bits_list)[idx]
        accs = np.array(acc_list)[idx]
        ax2.plot(bits, accs, "o-", color=c_ptq, label=label, linewidth=2, markersize=7)

    ax2.axhline(float_acc, color="gray", linestyle=":", linewidth=1.5,
                label=f"Float32 ({float_acc:.4f})")
    ax2.set_xlabel("Bits de cuantización")
    ax2.set_ylabel("Accuracy")
    metric_label = "QAT" if run_qat else "PTQ"
    ax2.set_title(f"Comparación 3 Pasos ({metric_label}) — Modelo: {model_type.upper()}")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    all_bits = sorted(set(b for s in steps_meta for b_str, r in results.get(s["id"], {}).items()
                          if "error" not in r for b in [r["bits"]]))
    if all_bits:
        ax2.set_xticks(all_bits)
    plt.tight_layout()

    plot2_path = os.path.join(output_dir, f"{model_type}_barrido_resumen.png")
    plt.savefig(plot2_path, dpi=120, bbox_inches="tight")
    print(f"[INFO] Gráfico resumen guardado en '{plot2_path}'")
    plt.close()


# ---------------------------------------------------------------------------
# Formato antiguo (6 estrategias, para retrocompatibilidad)
# ---------------------------------------------------------------------------

def plot_legacy_format(data: dict, json_path: str, output_dir: str):
    if "results" in data:
        model_name = data.get("model_name", "model")
        float_acc  = data.get("float_acc", 0.0)
        results    = data.get("results", {})
        strategies = data.get("strategies", [])
    else:
        results    = data
        float_acc  = 0.0
        model_name = os.path.basename(json_path).replace("_resultados_cuantizacion.json", "")
        strat_set  = set()
        for k in results:
            parts = k.split("_")
            if len(parts) >= 3:
                strat_set.add((parts[1], parts[2]))
        strategies = sorted(strat_set)

    print(f"\n[FORMATO ANTIGUO] Modelo: {model_name}  Float32: {float_acc:.4f}")

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors     = prop_cycle.by_key()["color"]

    plt.figure(figsize=(14, 9))
    for i, (s_scope, s_metric) in enumerate(strategies):
        bits_plot, ptq_plot, qat_plot = [], [], []
        for bits_key, res in results.items():
            if res.get("scope") == s_scope and res.get("metric") == s_metric:
                bits_plot.append(res["bits"])
                ptq_plot.append(res.get("ptq_acc", 0.0))
                qat_plot.append(res.get("qat_acc", 0.0))
        if not bits_plot:
            continue
        idx   = np.argsort(bits_plot)
        bp    = np.array(bits_plot)[idx]
        pp    = np.array(ptq_plot)[idx]
        qp    = np.array(qat_plot)[idx]
        color = colors[i % len(colors)]
        plt.plot(bp, qp, "o-",  color=color, label=f"QAT: {s_scope}-{s_metric}")
        plt.plot(bp, pp, "x--", color=color, alpha=0.6, label=f"PTQ: {s_scope}-{s_metric}")

    plt.axhline(float_acc, color="black", linestyle=":", linewidth=2,
                label=f"Baseline Float32 ({float_acc:.4f})")
    plt.xlabel("Bits de Cuantización")
    plt.ylabel("Accuracy")
    plt.title(f"PTQ vs QAT vs Baseline — {model_name}")
    all_bits = sorted(set(res["bits"] for res in results.values() if "bits" in res))
    if all_bits:
        plt.xticks(all_bits)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{model_name}_grafico_accuracy_evolucion.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"[INFO] Gráfico guardado en '{plot_path}'")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        # Intentar el nuevo formato primero, luego el antiguo
        candidates = [
            "outputs/cnn_barrido_resultados.json",
            "outputs/mlp_barrido_resultados.json",
            "outputs/modelo_CNN_imagenes_fashion_resultados_cuantizacion.json",
            "outputs/modelo_MLP_imagenes_fashion_resultados_cuantizacion.json",
        ]
        json_path = next((p for p in candidates if os.path.exists(p)), None)
        if json_path is None:
            print("[ERROR] No se encontró ningún archivo de resultados.")
            print("Ejecute script03.py primero o especifique la ruta al JSON.")
            sys.exit(1)

    if not os.path.exists(json_path):
        print(f"[ERROR] No se encuentra: {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Detectar formato por la presencia de la clave "steps" (nuevo)
    if "steps" in data and "results" in data and isinstance(data["results"], dict):
        first_val = next(iter(data["results"].values()), None)
        if isinstance(first_val, dict) and all(isinstance(v, dict) for v in first_val.values()):
            plot_new_format(data, output_dir)
            return

    plot_legacy_format(data, json_path, output_dir)


if __name__ == "__main__":
    main()
