import os
import sys
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow y Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
# QKeras para Cuantización
from qkeras.utils import model_quantize

# Import de funciones compartidas (reutilización de código)
from script06_utils import calculate_network_statistics, get_integer_bits, save_results, load_and_preprocess_data, print_complexity_report

# Desactiva logs molestos y warnings de inicializadores al crear sub-modelos de QKeras
original_tf_log_level = tf.get_logger().level
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='keras.initializers.initializers')

def main():
    """
    Este script toma un modelo entrenado (por defecto CNN) y aplica 6 estrategias de
    cuantización distintas para un UNICO ancho de bits (por defecto 8 bits).
    Evalúa PTQ (Post-Training Quantization), corre QAT (Quantization-Aware Training),
    y grafica los resultados.
    """
    # 1. Parámetros Generales de Cuantización
    bits = 8                  # Se fija el número de bits para la cuantización.
    run_qat = False           # Habilitar o deshabilitar ejecución de QAT (ahorra tiempo si solo se quiere ver PTQ)
    epochs = 1                # Para QAT, si se desea un experimento corto. Se puede extender.
    batch_size = 8192         # Tamaño batch (menor que en el script original para no fallar de RAM)
    use_manual_config = False # Controla si se usa la config manual (True) o calibración automática (False)
    
    results_acc = {}
    
    # Logs iniciales de versiones
    print("Python version:", sys.version.split(' ')[0])
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", keras.__version__)

    # Configuración de GPUs
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
    
    # 4. Estadísticas del Modelo Previo a Cuantización
    # Utilizamos el módulo externo "script06_utils.py" para no duplicar código
    print("\n[INFO] Calculando base estadística para las 6 estrategias...")
    layer_metrics, global_metrics = calculate_network_statistics(original_model)
    
    # =============================================================================
    # 5. ESTRATEGIAS DE CUANTIZACIÓN (PTQ + QAT)
    # =============================================================================
    
    print("\n" + "="*70)
    print("PTQ & QAT: Explorando las 6 Estrategias (Global/Mixed, Max/Percentiles)")
    print("="*70)
    
    # Exactitud "Gold Standard" o Float32 del modelo original para ser tomada como base
    print(f"\n[Float32] Evaluando precisión del modelo original FP32...")
    float_eval = original_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    float_acc = float_eval[1]
    
    if run_qat:
        print(f"  {'Bits':>4s} | {'Scope':>6s} | {'Metrica':>7s} | {'Acc PTQ':>8s} | {'Acc QAT':>8s}")
    else:
        print(f"  {'Bits':>4s} | {'Scope':>6s} | {'Metrica':>7s} | {'Acc PTQ':>8s}")
    print("  " + "-"*65)
    
    # Se definen las 6 estrategias a probar en un array de tuplas
    strategies = [
        ("Global", "max"),
        ("Global", "p95"),
        ("Global", "p75"),
        ("Mixed", "max"),
        ("Mixed", "p95"),
        ("Mixed", "p75")
    ]

    last_qmodel = None  # Guardar el último modelo cuantizado exitoso para el reporte de complejidad
    try:
        # Iteramos en las 6 estrategias usando el nivel fijo de cuantización (ej: 8 bits)
        for scope, metric in strategies:
            tf.keras.backend.clear_session()
            
            if use_manual_config:
                # --- CONFIGURACIÓN MANUAL ---
                # Permite elegir a mano los parámetros de cuantización por capa.
                # (Suficiente con listar las capas deseadas, el resto tomará defaults si existen)
                config_calibrado = {
                    "conv2d_1": {
                        "kernel_quantizer": f"quantized_linear({bits}, 0, symmetric=False, keep_negative=True, alpha=1)",
                        "bias_quantizer": f"quantized_linear({bits}, -2, symmetric=False, keep_negative=True, alpha=1)",
                        "activation_quantizer": f"quantized_relu({bits}, 2)"
                    },
                    "conv2d_2": {
                        "kernel_quantizer": f"quantized_linear({bits}, 0, symmetric=False, keep_negative=True, alpha=1)",
                        "bias_quantizer": f"quantized_linear({bits}, -2, symmetric=False, keep_negative=True, alpha=1)",
                        "activation_quantizer": f"quantized_relu({bits}, 2)"
                    },
                    "dense_1": {
                        "kernel_quantizer": f"quantized_linear({bits}, 0, symmetric=False, keep_negative=True, alpha=1)",
                        "bias_quantizer": f"quantized_linear({bits}, -2, symmetric=False, keep_negative=True, alpha=1)",
                        "activation_quantizer": f"quantized_relu({bits}, 2)"
                    },
                    "dense_2": {
                        "kernel_quantizer": f"quantized_linear({bits}, 0, symmetric=False, keep_negative=True, alpha=1)",
                        "bias_quantizer": f"quantized_linear({bits}, -2, symmetric=False, keep_negative=True, alpha=1)",
                        "activation_quantizer": f"quantized_relu({bits}, 2)"
                    },
                    "output": {
                        "kernel_quantizer": f"quantized_linear({bits}, 0, symmetric=False, keep_negative=True, alpha=1)",
                        "bias_quantizer": f"quantized_linear({bits}, -2, symmetric=False, keep_negative=True, alpha=1)"
                    }
                }
            else:
                # --- CONFIGURACIÓN CALIBRADA (ESTADÍSTICAS) ---
                config_calibrado = {}
                for layer in original_model.layers:
                    if not layer.get_weights(): # Ignorar capas sin parámetros (e.g. Flatten, MaxPooling)
                        continue
                    
                    # 5.1 Selección de la Métrica por Alcance
                    if scope == "Global":
                        w_val = global_metrics['w'][metric]
                        b_val = global_metrics['b'][metric]
                        act_val = global_metrics['act'][metric]
                    else: # Mixed
                        w_val = layer_metrics[layer.name]['w'][metric]
                        b_val = layer_metrics[layer.name].get('b', {}).get(metric, 0)
                        act_val = layer_metrics.get(layer.name, {}).get('act', {}).get(metric, 0)
                    
                    # 5.2 Conversión de métrica en Bits Enteros
                    w_int = get_integer_bits(w_val)
                    b_int = get_integer_bits(b_val)
                    
                    # 5.3 Asignación de Quantizers (Pesos y Biases)
                    layer_config = {
                        "kernel_quantizer": f"quantized_linear({bits}, {w_int}, symmetric=False, keep_negative=True, alpha=1)",
                        "bias_quantizer": f"quantized_linear({bits}, {b_int}, symmetric=False, keep_negative=True, alpha=1)",
                    }
                    
                    # 5.4 Cuantizadores de Activación
                    # La salida general (e.g., Softmax) no requiere ReLU cuantizado.
                    if layer.name != "output": 
                        act_int = get_integer_bits(act_val)
                        # FIX: quantized_relu de QKeras falla cuando recibe act_int < 0 debido a un bug con tf.pow()
                        # A diferencia de quantized_linear donde los bits negativos funcionan perfecto.
                        # Por ende, solo clampeamos las activaciones a 0 como mínimo.
                        act_int_safe = max(0, act_int)
                        layer_config["activation_quantizer"] = f"quantized_relu({bits}, {act_int_safe})"
                        
                    config_calibrado[layer.name] = layer_config
            
            _scope = "Manual" if use_manual_config else scope
            _metric = "custom" if use_manual_config else metric
            bits_key = f"{bits}_{_scope}_{_metric}"
            
            if use_manual_config and bits_key in results_acc:
                continue

            results_acc[bits_key] = {
                "bits": bits,
                "scope": _scope,
                "metric": _metric,
                "config": config_calibrado
            }

            tag = f"[{bits}b-{scope}-{metric}]"
            try:
                # Aplicar QKeras para generar un Sub-Modelo Cuantizado (PTQ)
                print(f"\n{tag} Cuantizando modelo QKeras...")
                qmodel_calibrado = model_quantize(original_model, config_calibrado, bits, transfer_weights=True)
                qmodel_calibrado.compile(
                    optimizer=tf.keras.optimizers.Adam(1e-4),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                    run_eagerly=True
                )
                last_qmodel = qmodel_calibrado  # Guardar referencia al último modelo cuantizado

                # 6. Evaluación PTQ (Post-Training Quantization)
                print(f"{tag} PTQ - Evaluando modelo directamente...")
                q_acc_ptq = qmodel_calibrado.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)[1]
                delta_ptq = q_acc_ptq - float_acc
                results_acc[bits_key]["ptq_acc"] = float(q_acc_ptq)
                print(f"  --> Resultado PTQ: {q_acc_ptq:8.4f} (Delta ref: {delta_ptq:+8.4f})")
                
                # 7. Ejecución de QAT (Quantization-Aware Training)
                if run_qat:
                    print(f"{tag} QAT - Reentrenando modelo cuantizado ({epochs} época(s))...")
                    qat_fit_params = {
                        "epochs": epochs, 
                        "batch_size": batch_size, 
                        "validation_split": 0.2, 
                        "verbose": 1
                    }
                    results_acc[bits_key]["qat_fit_params"] = qat_fit_params
                    # Hacemos finetuning a través de un corto Fit
                    history = qmodel_calibrado.fit(x_train, y_train, **qat_fit_params)

                    results_acc[bits_key]["history"] = {k: [float(v) for v in l] for k, l in history.history.items()}

                    # 8. Evaluación QAT Resultante
                    print(f"{tag} QAT - Evaluando nuevo modelo entrenado...")
                    q_acc_qat = qmodel_calibrado.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)[1]
                    delta_qat = q_acc_qat - float_acc
                    results_acc[bits_key]["qat_acc"] = float(q_acc_qat)
                    print(f"  --> Resultado QAT: {q_acc_qat:8.4f} (Delta ref: {delta_qat:+8.4f})")
                else:
                    results_acc[bits_key]["qat_acc"] = 0.0
                
            except Exception as e:
                results_acc[bits_key]["error"] = str(e)
                print(f"[{tag}] ERROR detectado en la estrategia: {e}")
                
    except KeyboardInterrupt:
        print("\n\n" + "!"*80)
        print("INTERRUPCIÓN DETECTADA (Ctrl+C). Guardando resultados parciales...")
        print("!"*80)

    # 9. Guardar a métricas y tabla
    save_results(output_dir, model_basename, float_acc, results_acc, strategies)
    
    print("\n" + "="*100)
    print(f"TABLA DE RESULTADOS FINALES (Baseline Float32 Acc: {float_acc:8.4f})")
    print("="*100)
    if run_qat:
        print(f"{'Bits':>5s} | {'Scope':>6s} | {'Metrica':>7s} | {'Acc PTQ':>8s} | {'Acc QAT':>8s} | {'Mejora QAT':>10s} | {'Delta Orig':>10s}")
    else:
        print(f"{'Bits':>5s} | {'Scope':>6s} | {'Metrica':>7s} | {'Acc PTQ':>8s} | {'Delta Orig':>10s}")
    print("-" * 100)
    
    for _, res in results_acc.items():
        if "error" in res:
            print(f"{res.get('bits', ''):>5} | {res.get('scope', ''):>6s} | {res.get('metric', ''):>7s} | ERROR: {res['error']}")
        else:
            b = res["bits"]
            scope = res["scope"]
            metric = res["metric"]
            ptq = res["ptq_acc"]
            qat = res.get("qat_acc", 0.0)
            
            if run_qat:
                mejora = qat - ptq
                delta_orig = qat - float_acc
                print(f"{b:5d} | {scope:6s} | {metric:7s} | {ptq:8.4f} | {qat:8.4f} | {mejora:+10.4f} | {delta_orig:+10.4f}")
            else:
                delta_orig = ptq - float_acc
                print(f"{b:5d} | {scope:6s} | {metric:7s} | {ptq:8.4f} | {delta_orig:+10.4f}")
    
    print("-" * 100)
    
    # 10. Gráfico Comparativo
    print("\n[INFO] Generando gráfico de comparación de estrategias...")
    
    labels = []
    ptq_accs = []
    qat_accs = []
    
    for scope, metric in strategies:
        bits_key = f"{bits}_{scope}_{metric}"
        res = results_acc.get(bits_key, {})
        labels.append(f"{scope}\n{metric}")
        ptq_accs.append(res.get("ptq_acc", 0.0))
        qat_accs.append(res.get("qat_acc", 0.0))
        
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if run_qat:
        rects1 = ax.bar(x - width/2, ptq_accs, width, label='PTQ', color='skyblue')
        rects2 = ax.bar(x + width/2, qat_accs, width, label='QAT', color='orange')
    else:
        rects1 = ax.bar(x, ptq_accs, width, label='PTQ', color='skyblue')
    
    # Línea horizontal para Float32 base
    ax.axhline(y=float_acc, color='r', linestyle='--', label=f'Original FP32 ({float_acc:.4f})')
    
    ax.set_ylabel('Accuracy')
    title_suffix = " (Solo PTQ)" if not run_qat else ""
    ax.set_title(f'Comparación de Estrategias{title_suffix} - Modelo {bits} Bits\n{model_basename}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Zoom en el gráfico dinámico adaptándose a los valores reales
    all_accs = ptq_accs + [float_acc]
    if run_qat:
        all_accs += qat_accs
    valid_accs = [a for a in all_accs if a > 0]
    min_acc = min(valid_accs) - 0.02 if valid_accs else 0.0
    max_acc = max(valid_accs) + 0.02 if valid_accs else 1.0
    ax.set_ylim(max(0.0, min_acc), min(1.05, max_acc))
    
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    
    # Añadir valores sobre las barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=90, fontsize=8)
                
    autolabel(rects1)
    if run_qat:
        autolabel(rects2)
    
    fig.tight_layout()
    
    # Guardar la imagen
    plot_path = os.path.join(output_dir, f"{model_basename}_{bits}bits_comparacion.png")
    plt.savefig(plot_path)
    print(f"[INFO] Gráfico guardado exitosamente en '{plot_path}'")
    
    # Mostrar el gráfico en pantalla (bloqueante)
    plt.show()

    # 11. Reporte de Complejidad del Modelo
    # Los bits se extraen automáticamente de los cuantizadores del modelo:
    #   - original_model: capas Keras estándar → 32 bits (float32)
    #   - last_qmodel: capas QKeras con quantizers → bits se leen de kernel_quantizer/activation_quantizer
    print("\n[INFO] Calculando complejidad del modelo...")
    print_complexity_report(original_model)
    if last_qmodel is not None:
        print_complexity_report(last_qmodel)
    else:
        print("[WARN] No se pudo generar reporte de complejidad cuantizado (ningún modelo cuantizado exitoso).")


    # Restaurar logs
    tf.get_logger().setLevel(original_tf_log_level)

if __name__ == "__main__":
    main()
