import os
import sys
import logging
import warnings
import numpy as np

# TensorFlow y Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
# QKeras para Cuantización
from qkeras.utils import model_quantize

# Import de funciones compartidas
from script06_utils import calculate_network_statistics, get_integer_bits, save_results, load_and_preprocess_data

original_tf_log_level = tf.get_logger().level
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='keras.initializers.initializers')

def main():
    """
    Este script toma un modelo entrenado (por defecto CNN) y aplica 6 estrategias de
    cuantización distintas para un BARRIDO de bits (1 al 16 si full_simulation).
    Evalúa PTQ (Post-Training Quantization) y corre QAT (Quantization-Aware Training).
    """
    
    # Toggle para acortar o hacer el barrido largo
    full_simulation = False
    if full_simulation:
        bits_list = np.arange(1, 17)
        epochs = 15 # 15 epocas QAT para convergencia completa, o reducir a gusto
    else:
        bits_list = [2, 4, 8]
        epochs = 1
        
    batch_size = 512
    results_acc = {}
    
    print("Python version:", sys.version.split(' ')[0])
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", keras.__version__)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[GPU] Detectada(s): {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[WARN] No se detectó GPU, usando CPU.")

    # 2. Cargar el Modelo de Referencia (Base)
    model_type = "cnn"  # Por defecto usará CNN
    if len(sys.argv) > 1:
        arg_type = sys.argv[1].lower()
        if arg_type in ["cnn", "mlp"]:
            model_type = arg_type
        else:
            print("[WARN] Argumento no reconocido. Use 'cnn' o 'mlp'. Usando CNN por defecto.")
            
    model_name = f"modelo_{model_type.upper()}_imagenes_fashion.h5"
    model_path = os.path.join(os.getcwd(), "modelos", model_name)
    model_basename = os.path.splitext(model_name)[0]
    
    if not os.path.exists(model_path):
        print(f"[ERROR] No se encuentra el modelo: {model_path}")
        print("Por favor, corra 'script06a_cnn.py' para entrenar la red primero.")
        sys.exit(1)
        
    print(f"\n[INFO] Cargando modelo base desde: {model_path}")
    original_model = load_model(model_path)
    original_model.summary()
    
    # 3. Carga y Normalización del Conjunto de Datos (Fashion MNIST)
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
        
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Base estadística usando nuestra utilidad externa
    print("\n[INFO] Calculando base estadística para las 6 estrategias...")
    layer_metrics, global_metrics = calculate_network_statistics(original_model)
    
    print("\n" + "="*70)
    print("PTQ & QAT: Explorando las 6 Estrategias (Global/Mixed, Max/Percentiles)")
    print("="*70)
    
    print("\n[Float32] Evaluando precisión del modelo original FP32...")
    float_eval = original_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    float_acc = float_eval[1]
    
    print(f"  {'Bits':>4s} | {'Scope':>6s} | {'Metrica':>7s} | {'Acc PTQ':>8s} | {'Acc QAT':>8s}")
    print("  " + "-"*65)
    
    strategies = [
        ("Global", "max"),
        ("Global", "p95"),
        ("Global", "p75"),
        ("Mixed", "max"),
        ("Mixed", "p95"),
        ("Mixed", "p75")
    ]

    try:
        # Se barren todos los bitwidths especificados (e.g. 1 a 16)
        for bits in bits_list:
            for scope, metric in strategies:
                tf.keras.backend.clear_session()
                
                config_calibrado = {}
                for layer in original_model.layers:
                    if not layer.get_weights():
                        continue
                    
                    if scope == "Global":
                        w_val = global_metrics['w'][metric]
                        b_val = global_metrics['b'][metric]
                        act_val = global_metrics['act'][metric]
                    else:
                        w_val = layer_metrics[layer.name]['w'][metric]
                        b_val = layer_metrics[layer.name].get('b', {}).get(metric, 0)
                        act_val = layer_metrics.get(layer.name, {}).get('act', {}).get(metric, 0)
                    
                    w_int = get_integer_bits(w_val)
                    b_int = get_integer_bits(b_val)
                    
                    layer_config = {
                        "kernel_quantizer": f"quantized_linear({bits}, {w_int}, symmetric=False, keep_negative=True, alpha=1)",
                        "bias_quantizer": f"quantized_linear({bits}, {b_int}, symmetric=False, keep_negative=True, alpha=1)",
                    }
                    
                    # Cuantizar activaciones (excepto logits de salida)
                    if layer.name != "output": 
                        act_int = get_integer_bits(act_val)
                        # Omitimos bug de QKeras limitando relu a int minimo 0
                        act_int_safe = max(0, act_int)
                        layer_config["activation_quantizer"] = f"quantized_relu({bits}, {act_int_safe})"
                        
                    config_calibrado[layer.name] = layer_config
                    
                bits_key = f"{bits}_{scope}_{metric}"
                results_acc[bits_key] = {
                    "bits": int(bits),
                    "scope": scope,
                    "metric": metric,
                    "config": config_calibrado
                }

                tag = f"[{bits}b-{scope}-{metric}]"
                try:
                    # QKeras quantize operation
                    qmodel_calibrado = model_quantize(original_model, config_calibrado, bits, transfer_weights=True)
                    qmodel_calibrado.compile(
                        optimizer=tf.keras.optimizers.Adam(1e-4),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"],
                        run_eagerly=True
                    )

                    # 1. Post-Training Quantization
                    q_acc_ptq = qmodel_calibrado.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)[1]
                    delta_ptq = q_acc_ptq - float_acc
                    results_acc[bits_key]["ptq_acc"] = float(q_acc_ptq)
                    print(f"  {bits:4d} | {scope:6s} | {metric:7s} | PTQ: {q_acc_ptq:8.4f} ({delta_ptq:+8.4f}) | Iniciando QAT...")
                    
                    # 2. Quantization-Aware Training
                    qat_fit_params = {
                        "epochs": epochs, 
                        "batch_size": batch_size, 
                        "validation_split": 0.2, 
                        "verbose": 0  # Silenciado ya que son muchas combinaciones
                    }
                    results_acc[bits_key]["qat_fit_params"] = qat_fit_params
                    
                    history = qmodel_calibrado.fit(x_train, y_train, **qat_fit_params)
                    results_acc[bits_key]["history"] = {k: [float(v) for v in l] for k, l in history.history.items()}

                    # 3. Evaluar QAT
                    q_acc_qat = qmodel_calibrado.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)[1]
                    delta_qat = q_acc_qat - float_acc
                    results_acc[bits_key]["qat_acc"] = float(q_acc_qat)
                    print(f"  {bits:4d} | {scope:6s} | {metric:7s} | QAT: {q_acc_qat:8.4f} ({delta_qat:+8.4f}) | Listo.")
                    
                except Exception as e:
                    results_acc[bits_key]["error"] = str(e)
                    print(f"  {bits:4d} | {scope:6s} | {metric:7s} | ERROR detectado: {e}")
                    
    except KeyboardInterrupt:
        print("\n\n" + "!"*80)
        print("INTERRUPCIÓN DETECTADA (Ctrl+C). Guardando resultados parciales...")
        print("!"*80)

    # Wrapup: guardar .json y plot final en consola
    save_results(output_dir, model_basename, float_acc, results_acc, strategies)
    
    print("\n" + "="*100)
    print(f"TABLA DE RESULTADOS FINALES (Baseline Float32 Acc: {float_acc:8.4f})")
    print("="*100)
    print(f"{'Bits':>5s} | {'Scope':>6s} | {'Metrica':>7s} | {'Acc PTQ':>8s} | {'Acc QAT':>8s} | {'Mejora QAT':>10s} | {'Delta Orig':>10s}")
    print("-" * 100)
    
    for bits_key, res in results_acc.items():
        if "error" in res:
            print(f"{res.get('bits', ''):>5} | {res.get('scope', ''):>6s} | {res.get('metric', ''):>7s} | ERROR: {res['error']}")
        else:
            b = res["bits"]
            scope = res["scope"]
            metric = res["metric"]
            ptq = res["ptq_acc"]
            qat = res["qat_acc"]
            mejora = qat - ptq
            delta_orig = qat - float_acc
            
            print(f"{b:5d} | {scope:6s} | {metric:7s} | {ptq:8.4f} | {qat:8.4f} | {mejora:+10.4f} | {delta_orig:+10.4f}")
    
    print("-" * 100)
    tf.get_logger().setLevel(original_tf_log_level)

if __name__ == "__main__":
    main()
