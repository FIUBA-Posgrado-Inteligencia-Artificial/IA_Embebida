"""
Sección 6 — Cuantización: Barrido de Bits en 3 Pasos
======================================================
Barre un rango de bitwidths sobre los tres pasos de cuantización:

  Paso 1 — Uniforme
  Paso 2 — Mixta int-bits (clip p95)
  Paso 3 — Mixta completa (primera/última = bits, interior = bits//2 mínimo 2)

Genera una tabla PTQ+QAT para cada (bits, paso) y guarda los resultados en JSON.
El script04.py los visualiza después.

Uso:
    cd examples/quantization && uv run python script03.py [cnn|mlp] [--full]

    --full : barrido completo bits 1..16 con QAT  (por defecto: [2,4,6,8] sin QAT)
"""

import os
import sys
import json
import logging
import warnings

import numpy as np
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
)

original_tf_log_level = tf.get_logger().level
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="keras.initializers.initializers")

BATCH_SIZE = 512
OUTPUT_DIR = "outputs"


def _quantize_and_eval(original_model, config, bits_for_qkeras,
                       x_train, y_train, x_test, y_test,
                       run_qat: bool, qat_epochs: int):
    tf.keras.backend.clear_session()
    qmodel = model_quantize(original_model, config, bits_for_qkeras, transfer_weights=True)
    qmodel.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True,
    )
    ptq_acc = qmodel.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)[1]
    qat_acc = 0.0
    if run_qat:
        qmodel.fit(x_train, y_train, epochs=qat_epochs, batch_size=BATCH_SIZE,
                   validation_split=0.1, verbose=0)
        qat_acc = qmodel.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)[1]
    _, totals = calculate_complexity(qmodel)
    return float(ptq_acc), float(qat_acc), totals["total_spatial_bytes"], totals["total_macs"]


def main():
    print("Python :", sys.version.split(" ")[0])
    print("TF     :", tf.__version__)
    print("Keras  :", keras.__version__)

    full_simulation = "--full" in sys.argv
    args_clean = [a for a in sys.argv[1:] if not a.startswith("--")]

    model_type = "cnn"
    if args_clean and args_clean[0].lower() in ("cnn", "mlp"):
        model_type = args_clean[0].lower()

    if full_simulation:
        bits_list  = list(range(1, 17))
        qat_epochs = 10
        run_qat    = True
    else:
        bits_list  = [2, 4, 6, 8]
        qat_epochs = 1
        run_qat    = True

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[WARN] Sin GPU, usando CPU.")

    model_name = f"modelo_{model_type.upper()}_imagenes_fashion.h5"
    model_path = os.path.join(os.getcwd(), "models", model_name)
    if not os.path.exists(model_path):
        print(f"[ERROR] No se encuentra: {model_path}")
        print("Ejecute primero script01.py.")
        sys.exit(1)

    original_model = load_model(model_path)
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    float_acc = original_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)[1]
    print(f"\n[Float32] acc={float_acc:.4f}")

    print("[INFO] Calculando estadísticas de pesos y activaciones...")
    layer_metrics, global_metrics = calculate_network_statistics(original_model)

    # ------------------------------------------------------------------
    # Definición de los 3 pasos de cuantización
    # ------------------------------------------------------------------
    STEP_DEFS = [
        {"id": "paso1", "label": "Paso 1 — Uniforme",      "clip": "max"},
        {"id": "paso2", "label": "Paso 2 — Mixta/p95",     "clip": "p95"},
        {"id": "paso3", "label": "Paso 3 — Completa/p95",  "clip": "p95"},
    ]

    results = {s["id"]: {} for s in STEP_DEFS}

    print(f"\n{'='*70}")
    print(f"BARRIDO DE BITS: {bits_list}  |  QAT={run_qat}  |  Modelo={model_type.upper()}")
    print(f"{'='*70}")
    print(f"  {'Bits':>4} | {'Paso':<25} | {'PTQ':>8} | {'QAT':>8} | {'ΔFLT':>7}")
    print("  " + "-" * 60)

    try:
        for bits in bits_list:
            # Paso 3 usa: primera/última = bits, interior = max(2, bits//2)
            low_bits_p3 = max(2, bits // 2)

            paso_configs = [
                (
                    "paso1",
                    build_uniform_config(original_model, bits, layer_metrics, global_metrics),
                    bits,
                ),
                (
                    "paso2",
                    build_mixed_intbits_config(original_model, bits, layer_metrics, "p95"),
                    bits,
                ),
                (
                    "paso3",
                    build_full_mixed_config(
                        original_model,
                        make_default_full_mixed_bits_map(original_model, bits, low_bits_p3),
                        layer_metrics,
                        "p95",
                    ),
                    bits,
                ),
            ]

            for paso_id, config, bfq in paso_configs:
                label = next(s["label"] for s in STEP_DEFS if s["id"] == paso_id)
                try:
                    ptq, qat, mem, macs = _quantize_and_eval(
                        original_model, config, bfq,
                        x_train, y_train, x_test, y_test,
                        run_qat, qat_epochs,
                    )
                    best = qat if run_qat else ptq
                    results[paso_id][str(bits)] = {
                        "bits": bits, "ptq_acc": ptq, "qat_acc": qat,
                        "mem_bytes": mem, "macs": macs,
                    }
                    print(f"  {bits:4d} | {label:<25} | {ptq:8.4f} | {qat:8.4f} | {best-float_acc:+7.4f}")
                except Exception as e:
                    print(f"  {bits:4d} | {label:<25} | ERROR: {e}")
                    results[paso_id][str(bits)] = {"bits": bits, "error": str(e)}

    except KeyboardInterrupt:
        print("\n[INTERRUPCION] Guardando resultados parciales...")

    # ------------------------------------------------------------------
    # Guardar JSON
    # ------------------------------------------------------------------
    json_data = {
        "model_type": model_type,
        "float_acc":  float_acc,
        "bits_list":  bits_list,
        "run_qat":    run_qat,
        "steps":      STEP_DEFS,
        "results":    results,
    }
    json_path = os.path.join(OUTPUT_DIR, f"{model_type}_barrido_resultados.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    print(f"\n[INFO] Resultados guardados en '{json_path}'")
    print("[INFO] Ejecute script04.py para visualizar los gráficos.")

    tf.get_logger().setLevel(original_tf_log_level)


if __name__ == "__main__":
    main()
