"""
Sección 6 — Cuantización: 3 Pasos de Precisión Mixta
======================================================
Aplica las tres estrategias de cuantización progresivas sobre un modelo entrenado
(CNN o MLP), a un ancho de bits fijo (por defecto 8 bits):

  Paso 1 — Uniforme:            mismos bits totales Y mismos bits enteros globalmente.
  Paso 2a — Mixta/max:          mismos bits totales, bits enteros per-capa (rango max).
  Paso 2b — Mixta/p95:          mismos bits totales, bits enteros per-capa (rango p95).
  Paso 2c — Mixta/p75:          mismos bits totales, bits enteros per-capa (rango p75).
  Paso 3  — Mixta completa/p95: bits totales distintos por capa (primera/última=8, resto=4),
                                  bits enteros per-capa (rango p95).

Para cada configuración se evalúa PTQ (calibración) y, si run_qat=True, también QAT.
Adicionalmente se genera una visualización de calibración de clipping para una capa.

Uso:
    cd examples/quantization && uv run python script02.py [cnn|mlp]
"""

import os
import sys
import logging
import warnings
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from qkeras.utils import model_quantize

from script06_utils import (
    calculate_network_statistics,
    calculate_complexity,
    load_and_preprocess_data,
    build_uniform_config,
    build_mixed_intbits_config,
    build_full_mixed_config,
    make_default_full_mixed_bits_map,
    plot_clipping_calibration,
    print_complexity_report,
)

original_tf_log_level = tf.get_logger().level
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="keras.initializers.initializers")


# ---------------------------------------------------------------------------
# Parámetros generales
# ---------------------------------------------------------------------------
BITS       = 8       # bits para los pasos 1 y 2
HIGH_BITS  = 8       # bits para primera/última capa en el paso 3
LOW_BITS   = 4       # bits para capas intermedias en el paso 3
RUN_QAT    = False   # poner True para incluir QAT en la comparativa
QAT_EPOCHS = 3
BATCH_SIZE = 4096
OUTPUT_DIR = "outputs"


# ---------------------------------------------------------------------------
# Helper: quantizar, evaluar PTQ, opcionalmente QAT
# ---------------------------------------------------------------------------

def _run_step(tag, original_model, config, bits_for_qkeras,
              x_train, y_train, x_test, y_test):
    """
    Cuantiza el modelo con `config`, evalúa PTQ y opcionalmente QAT.
    Retorna dict con ptq_acc, qat_acc (0 si RUN_QAT=False), mem_bytes, macs.
    """
    tf.keras.backend.clear_session()
    try:
        qmodel = model_quantize(original_model, config, bits_for_qkeras, transfer_weights=True)
        qmodel.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            run_eagerly=True,
        )
        ptq_acc = qmodel.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)[1]

        qat_acc = 0.0
        if RUN_QAT:
            qmodel.fit(x_train, y_train, epochs=QAT_EPOCHS, batch_size=BATCH_SIZE,
                       validation_split=0.1, verbose=0)
            qat_acc = qmodel.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)[1]

        _, totals = calculate_complexity(qmodel)
        mem_bytes = totals["total_spatial_bytes"]
        macs      = totals["total_macs"]

        return {"ptq_acc": ptq_acc, "qat_acc": qat_acc,
                "mem_bytes": mem_bytes, "macs": macs, "error": None}
    except Exception as e:
        print(f"  [ERROR] {tag}: {e}")
        return {"ptq_acc": 0.0, "qat_acc": 0.0, "mem_bytes": 0.0, "macs": 0, "error": str(e)}


def main():
    print("Python :", sys.version.split(" ")[0])
    print("TF     :", tf.__version__)
    print("Keras  :", keras.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[GPU] {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[WARN] Sin GPU, usando CPU.")

    # ---------------------------------------------------------------
    # Cargar modelo y datos
    # ---------------------------------------------------------------
    model_type = "cnn"
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("cnn", "mlp"):
        model_type = sys.argv[1].lower()

    model_name = f"modelo_{model_type.upper()}_imagenes_fashion.h5"
    model_path = os.path.join(os.getcwd(), "models", model_name)
    if not os.path.exists(model_path):
        print(f"[ERROR] No se encuentra: {model_path}")
        print("Ejecute primero script01.py.")
        sys.exit(1)

    print(f"\n[INFO] Cargando modelo: {model_path}")
    original_model = load_model(model_path)

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------------
    # Baseline float32
    # ---------------------------------------------------------------
    float_acc = original_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)[1]
    _, fp32_totals = calculate_complexity(original_model)
    fp32_mem  = fp32_totals["total_spatial_bytes"]
    fp32_macs = fp32_totals["total_macs"]
    print(f"\n[Float32] acc={float_acc:.4f}  mem={fp32_mem/1024:.1f}KB  MACs={fp32_macs:,}")

    # ---------------------------------------------------------------
    # Estadísticas de la red
    # ---------------------------------------------------------------
    print("\n[INFO] Calculando estadísticas de pesos y activaciones...")
    layer_metrics, global_metrics = calculate_network_statistics(original_model)

    # Detectar una capa representativa para calibración de clipping
    # (primera capa con pesos que no sea la capa de salida)
    calib_layer_name = None
    for layer in original_model.layers:
        if layer.get_weights() and layer.name != "output":
            calib_layer_name = layer.name
            break

    # ---------------------------------------------------------------
    # Ejecutar los 5 pasos
    # ---------------------------------------------------------------
    steps = [
        {
            "tag":  "Paso1-Uniforme",
            "step": 1,
            "clip": "max",
            "config": build_uniform_config(original_model, BITS, layer_metrics, global_metrics),
            "bits_for_qkeras": BITS,
            "label": f"Paso 1 — Uniforme ({BITS}b global)",
        },
        {
            "tag":  "Paso2a-Mixta/max",
            "step": 2,
            "clip": "max",
            "config": build_mixed_intbits_config(original_model, BITS, layer_metrics, "max"),
            "bits_for_qkeras": BITS,
            "label": f"Paso 2a — Mixta int-bits / max ({BITS}b)",
        },
        {
            "tag":  "Paso2b-Mixta/p95",
            "step": 2,
            "clip": "p95",
            "config": build_mixed_intbits_config(original_model, BITS, layer_metrics, "p95"),
            "bits_for_qkeras": BITS,
            "label": f"Paso 2b — Mixta int-bits / p95 ({BITS}b)",
        },
        {
            "tag":  "Paso2c-Mixta/p75",
            "step": 2,
            "clip": "p75",
            "config": build_mixed_intbits_config(original_model, BITS, layer_metrics, "p75"),
            "bits_for_qkeras": BITS,
            "label": f"Paso 2c — Mixta int-bits / p75 ({BITS}b)",
        },
        {
            "tag":   "Paso3-Completa/p95",
            "step":  3,
            "clip":  "p95",
            "config": build_full_mixed_config(
                original_model,
                make_default_full_mixed_bits_map(original_model, HIGH_BITS, LOW_BITS),
                layer_metrics,
                "p95",
            ),
            "bits_for_qkeras": HIGH_BITS,
            "label": f"Paso 3 — Mixta completa / p95 ({HIGH_BITS}b/{LOW_BITS}b)",
        },
    ]

    print(f"\n{'='*75}")
    print(f"CUANTIZACIÓN EN 3 PASOS  (baseline float32 acc = {float_acc:.4f})")
    print(f"{'='*75}")
    hdr = f"{'Paso/Clip':<30} {'Acc PTQ':>8}"
    if RUN_QAT:
        hdr += f" {'Acc QAT':>8}"
    hdr += f" {'ΔFLT':>7} {'Mem KB':>8} {'MACs':>12}"
    print(hdr)
    print("-" * 75)

    all_results = []
    for step_cfg in steps:
        res = _run_step(
            step_cfg["tag"], original_model, step_cfg["config"],
            step_cfg["bits_for_qkeras"], x_train, y_train, x_test, y_test,
        )
        res.update({"tag": step_cfg["tag"], "label": step_cfg["label"],
                    "step": step_cfg["step"], "clip": step_cfg["clip"]})
        all_results.append(res)

        if res["error"]:
            print(f"  {step_cfg['tag']:<30}  ERROR: {res['error']}")
            continue

        best_acc = res["qat_acc"] if RUN_QAT else res["ptq_acc"]
        delta    = best_acc - float_acc
        row = f"  {step_cfg['label']:<30} {res['ptq_acc']:>8.4f}"
        if RUN_QAT:
            row += f" {res['qat_acc']:>8.4f}"
        row += f" {delta:>+7.4f} {res['mem_bytes']/1024:>8.1f} {res['macs']:>12,}"
        print(row)

    print("-" * 75)

    # ---------------------------------------------------------------
    # Visualización de calibración de clipping
    # ---------------------------------------------------------------
    if calib_layer_name:
        clip_path = os.path.join(OUTPUT_DIR, "calibracion_clipping.png")
        plot_clipping_calibration(original_model, calib_layer_name, clip_path, bits=BITS)

    # ---------------------------------------------------------------
    # Gráfico de barras comparativo
    # ---------------------------------------------------------------
    labels   = [r["label"].replace(" — ", "\n").replace(" / ", "\n") for r in all_results if not r["error"]]
    ptq_accs = [r["ptq_acc"] for r in all_results if not r["error"]]
    qat_accs = [r["qat_acc"] for r in all_results if not r["error"]] if RUN_QAT else []

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    if RUN_QAT:
        ax.bar(x - width/2, ptq_accs, width, label="PTQ", color="#4C72B0")
        ax.bar(x + width/2, qat_accs, width, label="QAT", color="#55A868")
    else:
        ax.bar(x, ptq_accs, width, label="PTQ", color="#4C72B0")
    ax.axhline(float_acc, color="red", linestyle="--", linewidth=1.2, label=f"Float32 ({float_acc:.4f})")

    all_vals = ptq_accs + (qat_accs if RUN_QAT else []) + [float_acc]
    valid    = [v for v in all_vals if v > 0]
    ax.set_ylim(max(0, min(valid) - 0.03), min(1.02, max(valid) + 0.02))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Cuantización en 3 Pasos — {BITS} bits fijos, Paso 3 = {HIGH_BITS}b/{LOW_BITS}b\n"
                 f"Modelo: {model_type.upper()}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    def _label_bars(rects):
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7, rotation=90)

    for container in ax.containers:
        _label_bars(container)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"{model_type}_3pasos_comparacion.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"[INFO] Gráfico guardado en '{plot_path}'")
    plt.close()

    # ---------------------------------------------------------------
    # Guardar JSON
    # ---------------------------------------------------------------
    json_data = {
        "model_type":  model_type,
        "bits":        BITS,
        "high_bits":   HIGH_BITS,
        "low_bits":    LOW_BITS,
        "float_acc":   float_acc,
        "fp32_mem_kb": round(fp32_mem / 1024, 2),
        "fp32_macs":   fp32_macs,
        "steps":       [
            {k: v for k, v in r.items() if k != "error"}
            for r in all_results if not r["error"]
        ],
    }
    json_path = os.path.join(OUTPUT_DIR, f"{model_type}_3pasos_resultados.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Resultados guardados en '{json_path}'")

    # ---------------------------------------------------------------
    # Reporte de complejidad del modelo float32
    # ---------------------------------------------------------------
    print_complexity_report(original_model)

    tf.get_logger().setLevel(original_tf_log_level)


if __name__ == "__main__":
    main()
