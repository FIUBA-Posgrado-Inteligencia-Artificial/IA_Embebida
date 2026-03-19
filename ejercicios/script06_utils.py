import os
import math
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel
from scipy.stats import skew, kurtosis
from tensorflow.keras.datasets import fashion_mnist

def load_and_preprocess_data():
    """
    Carga el dataset Fashion MNIST y lo escala a [0, 1].
    Retorna arrays con forma (batch_size, 28, 28).
    """
    print("\n[INFO] Cargando dataset Fashion MNIST...")
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    X_train_norm = X_train.astype('float32') / 255.0
    X_test_norm = X_test.astype('float32') / 255.0
    
    # Añadimos canal=1 universalmente para compatibilidad estricta con Conv2D.
    # MLP no se ve afectado porque Flatten resuelve exitosamente (28, 28, 1) a 784.
    X_train_norm = np.expand_dims(X_train_norm, axis=-1)
    X_test_norm = np.expand_dims(X_test_norm, axis=-1)
    
    print(f"X_train shape: {X_train_norm.shape}")
        
    return (X_train_norm, y_train), (X_test_norm, y_test)

def save_results(output_dir, model_basename, float_acc, results_acc, strategies):
    """
    Guarda los resultados del experimento de cuantización en un diccionario estructurado
    dentro de un archivo JSON. Retiene el accuracy original (float) y los resultados
    para cada métrica estudiada.
    """
    json_path = os.path.join(output_dir, f"{model_basename}_resultados_cuantizacion.json")
    final_output = {
        "model_name": model_basename,
        "float_acc": float_acc,
        "results": results_acc,
        "strategies": strategies
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    print(f"\n[INFO] Resultados guardados exitosamente en '{json_path}'")

def get_integer_bits(value):
    """
    Retorna la cantidad de bits enteros requeridos para representar un valor dado
    utilizando logaritmo en base 2.
    Permitimos bits negativos ya que proveen precision suplementaria para valores
    muy pequeños, y quantized_linear de QKeras es compatible.
    """
    if value <= 0:
        return 0
    return math.ceil(math.log2(value))

def calculate_network_statistics(model, n_samples=1000, batch_size=32768, seed=42):
    """
    Calcula estadísticas detalladas (min, max, media, desvío, percentiles, skewness, kurtosis)
    de los pesos (kernels), biases y activaciones para cada capa del modelo.
    
    Estos parámetros permiten comprender cuán "dispersos" están los valores, facilitando
    luego la elección de la escala en una estrategia de cuantización.
    
    Retorna:
        - layer_metrics: estadísticas separadas capa por capa.
        - global_metrics: métricas consolidadas de manera global a lo largo de toda la red.
    """
    print("\n" + "="*106)
    print("Estadísticas de Pesos, Biases y Activaciones de la Red")
    print("="*106)
    
    input_shape = model.input_shape[1:]
    
    # Generamos datos aleatorios como "estímulo" para calcular el estado de las activaciones
    np.random.seed(seed)
    random_data = np.random.rand(n_samples, *input_shape).astype('float32')
    
    print(f"{'Capa':<15s} | {'Tipo':<10s} | {'Min':>9s} | {'Max':>9s} | {'Media':>9s} | {'Std':>9s} | {'P5':>9s} | {'P95':>9s} | {'P25':>9s} | {'P75':>9s} | {'Skew':>9s} | {'Kurt':>9s}")
    print("-" * 155)
    
    layer_metrics = {}
    global_metrics = {
        'w': {'max': 0, 'p95': 0, 'p75': 0},
        'b': {'max': 0, 'p95': 0, 'p75': 0},
        'act': {'max': 0, 'p95': 0, 'p75': 0}
    }
    
    for layer in model.layers:
        layer_metrics[layer.name] = {}
        pesos_capa = layer.get_weights()
        if pesos_capa:
            # 1. Estadísticas del Kernel (Pesos de la capa)
            w = pesos_capa[0].flatten()
            p5, p95 = np.percentile(w, [5, 95])
            p25, p75 = np.percentile(w, [25, 75])
            
            c_max = max(abs(w.min()), abs(w.max()))
            c_p95 = max(abs(p5), abs(p95))
            c_p75 = max(abs(p25), abs(p75))
            
            layer_metrics[layer.name]['w'] = {'max': c_max, 'p95': c_p95, 'p75': c_p75}
            
            global_metrics['w']['max'] = max(global_metrics['w']['max'], c_max)
            global_metrics['w']['p95'] = max(global_metrics['w']['p95'], c_p95)
            global_metrics['w']['p75'] = max(global_metrics['w']['p75'], c_p75)
            
            print(f"{layer.name:<15s} | {'Kernel':<10s} | {w.min():9.4f} | {w.max():9.4f} | {np.mean(w):9.4f} | {np.std(w):9.4f} | {p5:9.4f} | {p95:9.4f} | {p25:9.4f} | {p75:9.4f} | {skew(w):9.4f} | {kurtosis(w):9.4f}")
            
            # 2. Estadísticas de los Biases
            if len(pesos_capa) > 1:
                b = pesos_capa[1].flatten()
                bp5, bp95 = np.percentile(b, [5, 95])
                bp25, bp75 = np.percentile(b, [25, 75])
                
                cb_max = max(abs(b.min()), abs(b.max()))
                cb_p95 = max(abs(bp5), abs(bp95))
                cb_p75 = max(abs(bp25), abs(bp75))
                
                layer_metrics[layer.name]['b'] = {'max': cb_max, 'p95': cb_p95, 'p75': cb_p75}
                
                global_metrics['b']['max'] = max(global_metrics['b']['max'], cb_max)
                global_metrics['b']['p95'] = max(global_metrics['b']['p95'], cb_p95)
                global_metrics['b']['p75'] = max(global_metrics['b']['p75'], cb_p75)
                
                print(f"{layer.name:<15s} | {'Bias':<10s} | {b.min():9.4f} | {b.max():9.4f} | {np.mean(b):9.4f} | {np.std(b):9.4f} | {bp5:9.4f} | {bp95:9.4f} | {bp25:9.4f} | {bp75:9.4f} | {skew(b):9.4f} | {kurtosis(b):9.4f}")
        
        # 3. Estadísticas de las Activaciones de salida
        try:
            # Modelo truncado hasta la capa actual para inferir
            intermediate_model = KerasModel(inputs=model.input, outputs=layer.output)
            activations = intermediate_model.predict(random_data, batch_size=batch_size, verbose=0).flatten()
            
            act_min = float(np.min(activations))
            act_max = float(np.max(activations))
            
            ap5, ap95 = np.percentile(activations, [5, 95])
            ap25, ap75 = np.percentile(activations, [25, 75])
            
            ca_max = max(abs(act_min), abs(act_max))
            ca_p95 = max(abs(ap5), abs(ap95))
            ca_p75 = max(abs(ap25), abs(ap75))
            
            layer_metrics[layer.name]['act'] = {'max': ca_max, 'p95': ca_p95, 'p75': ca_p75}
            
            global_metrics['act']['max'] = max(global_metrics['act']['max'], ca_max)
            global_metrics['act']['p95'] = max(global_metrics['act']['p95'], ca_p95)
            global_metrics['act']['p75'] = max(global_metrics['act']['p75'], ca_p75)
            
            print(f"{layer.name:<15s} | {'Activación':<10s} | {act_min:9.4f} | {act_max:9.4f} | {np.mean(activations):9.4f} | {np.std(activations):9.4f} | {ap5:9.4f} | {ap95:9.4f} | {ap25:9.4f} | {ap75:9.4f} | {skew(activations):9.4f} | {kurtosis(activations):9.4f}")
        except Exception:
            # Algunas capas (input, dropout) no procesan de igual forma las activaciones numéricamente.
            print(f"{layer.name:<15s} | {'Activación':<10s} | (no calculable)")
        
        print("-" * 155)
        
    return layer_metrics, global_metrics

def verify_quantized_weights(qmodel, bits, w_integer):
    """
    Verifica que los pesos pasen las asunciones teóricas de la cuantización
    seleccionada, retornando posibles errores como diccionario.
    """
    try:
        qlayer = None
        for layer in qmodel.layers:
            if len(layer.get_weights()) > 0:
                qlayer = layer
                break
                
        if qlayer is None:
            return {"status": "ERROR", "errors": ["No cuantizable layers found"]}
            
        qweights_raw = qlayer.get_weights()[0]
        
        # Extracción de pesos cuantizados usando API de QKeras interna
        if hasattr(qlayer, 'kernel_quantizer_internal') and qlayer.kernel_quantizer_internal is not None:
            qweights = qlayer.kernel_quantizer_internal(qweights_raw).numpy()
        elif hasattr(qlayer, 'kernel_quantizer') and qlayer.kernel_quantizer is not None:
            qweights = qlayer.kernel_quantizer(qweights_raw).numpy()
        else:
            qweights = qweights_raw
        
        errors = []
        unique_vals = np.unique(qweights)
        
        if len(unique_vals) > 2**bits:
            errors.append(f"{len(unique_vals)} vals unicos > 2^{bits}")
        
        fractional_bits = bits - w_integer - 1 
        step_size = 2.0 ** -fractional_bits
        
        max_teorico = (2**(bits-1) - 1) * step_size
        min_teorico = -(2**(bits-1)) * step_size
        
        if qweights.max() > max_teorico + 1e-5:
            errors.append(f"max {qweights.max():.4f} > teo {max_teorico:.4f}")
        if qweights.min() < min_teorico - 1e-5:
            errors.append(f"min {qweights.min():.4f} < teo {min_teorico:.4f}")
        
        grid_pos = qweights / step_size
        if not np.allclose(grid_pos, np.round(grid_pos), atol=1e-3):
            errors.append("No es multiplo step_size")
            
        result = {
            "status": "PASS" if not errors else "FAIL", 
            "unique_values": int(len(unique_vals)), 
            "min_val": float(qweights.min()), 
            "max_val": float(qweights.max()), 
            "step_size": float(step_size)
        }
        
        if errors:
            result["errors"] = errors
            
        return result
    except Exception as e:
        return {"status": "ERROR", "errors": [str(e)]}


def calculate_ops_and_params(model):
    """
    Calcula la cantidad de operaciones (MACs) y parámetros por cada capa del modelo,
    así como los totales.

    Para Conv2D: MACs = K_h * K_w * C_in * C_out * H_out * W_out
                 Params = K_h * K_w * C_in * C_out + C_out (bias)
    Para Dense:  MACs = input_units * output_units
                 Params = input_units * output_units + output_units (bias)

    Retorna:
        layer_info: lista de dicts con info por capa
        totals: dict con totales de MACs y parámetros
    """
    layer_info = []
    total_macs = 0
    total_params = 0

    for layer in model.layers:
        info = {
            "name": layer.name,
            "type": layer.__class__.__name__,
            "macs": 0,
            "params": 0,
            "output_shape": layer.output_shape,
        }

        if isinstance(layer, tf.keras.layers.Conv2D):
            # Kernel shape: (K_h, K_w, C_in, C_out)
            kernel_shape = layer.kernel.shape
            k_h, k_w, c_in, c_out = kernel_shape
            # Output spatial dimensions
            out_shape = layer.output_shape  # (batch, H_out, W_out, C_out)
            h_out, w_out = out_shape[1], out_shape[2]
            macs = int(k_h * k_w * c_in * c_out * h_out * w_out)
            params = int(k_h * k_w * c_in * c_out)
            if layer.use_bias:
                params += int(c_out)
            info["macs"] = macs
            info["params"] = params
            info["details"] = {
                "kernel": (int(k_h), int(k_w)),
                "channels_in": int(c_in),
                "channels_out": int(c_out),
                "output_spatial": (int(h_out), int(w_out)),
            }

        elif isinstance(layer, tf.keras.layers.Dense):
            # Kernel shape: (input_units, output_units)
            kernel_shape = layer.kernel.shape
            n_in, n_out = kernel_shape
            macs = int(n_in * n_out)
            params = int(n_in * n_out)
            if layer.use_bias:
                params += int(n_out)
            info["macs"] = macs
            info["params"] = params
            info["details"] = {
                "units_in": int(n_in),
                "units_out": int(n_out),
            }

        else:
            # Capas sin operaciones aritméticas (Flatten, MaxPooling, Input, etc.)
            info["params"] = layer.count_params()

        total_macs += info["macs"]
        total_params += info["params"]
        layer_info.append(info)

    totals = {
        "total_macs": total_macs,
        "total_params": total_params,
    }

    return layer_info, totals


def _get_layer_bits(layer):
    """
    Extrae los bits de cuantización de los pesos y activaciones de una capa.
    
    Para modelos QKeras, busca los atributos kernel_quantizer y activation_quantizer
    que contienen la propiedad 'bits'. Para modelos Keras estándar (float32), retorna 32.
    
    Retorna:
        bits_w: bits del kernel (pesos)
        bits_act_out: bits de la activación de salida
    """
    bits_w = 32       # default: float32
    bits_act_out = 32  # default: float32
    
    # Buscar kernel quantizer (QKeras usa kernel_quantizer_internal y/o kernel_quantizer)
    for attr in ('kernel_quantizer_internal', 'kernel_quantizer'):
        q = getattr(layer, attr, None)
        if q is not None and hasattr(q, 'bits'):
            bits_w = int(q.bits)
            break
    
    # Buscar activation quantizer (QKeras)
    for attr in ('activation_quantizer_internal', 'activation_quantizer'):
        q = getattr(layer, attr, None)
        if q is not None and hasattr(q, 'bits'):
            bits_act_out = int(q.bits)
            break
    
    return bits_w, bits_act_out


def calculate_complexity(model):
    """
    Calcula la complejidad espacial y computacional de cada capa, extrayendo
    automáticamente el número de bits desde los cuantizadores del modelo.
    
    - Para modelos QKeras: lee bits desde kernel_quantizer y activation_quantizer de cada capa.
    - Para modelos Keras estándar (float32): asume 32 bits para pesos y activaciones.
    
    Los bits de activación de entrada de una capa corresponden a los bits de activación
    de salida de la capa anterior (propagación a lo largo de la red).

    Complejidad espacial (por capa):
        Tamaño en memoria de los pesos = params * bits_weights (en bits)

    Complejidad computacional (por capa):
        Cada MAC (multiply-accumulate) consiste en:
            - 1 multiplicación: costo = bits_act_entrada * bits_weights
            - 1 suma (acumulación): costo = max(bits_act_entrada, bits_weights)
        Complejidad total por capa = MACs * (b_act * b_w + max(b_act, b_w))

    Retorna:
        layer_complexity: lista de dicts con complejidad por capa
        totals: dict con totales de complejidad
    """
    layer_info, ops_totals = calculate_ops_and_params(model)

    # Detectar bits de activación inicial: buscar la primera capa cuantizada
    # para inferir los bits de entrada al modelo (en un pipeline cuantizado,
    # la entrada también se cuantiza al mismo nivel).
    initial_act_bits = 32
    for layer in model.layers:
        b_w, _ = _get_layer_bits(layer)
        if b_w != 32:
            initial_act_bits = b_w
            break

    prev_act_bits = initial_act_bits

    layer_complexity = []
    total_spatial_bits = 0
    total_computational = 0

    for info in layer_info:
        layer = model.get_layer(info["name"])
        b_w, b_act_out = _get_layer_bits(layer)

        # Bits de activación de entrada = activación de salida de la capa anterior
        b_a = prev_act_bits

        cost_mul = b_a * b_w          # complejidad de una multiplicación
        cost_add = max(b_a, b_w)      # complejidad de una suma

        spatial_bits = info["params"] * b_w           # bits totales en memoria
        spatial_bytes = spatial_bits / 8               # bytes totales en memoria
        computational = info["macs"] * (cost_mul + cost_add)  # complejidad computacional

        complexity = {
            "name": info["name"],
            "type": info["type"],
            "params": info["params"],
            "macs": info["macs"],
            "bits_weights": b_w,
            "bits_activations": b_a,
            "spatial_bits": spatial_bits,
            "spatial_bytes": spatial_bytes,
            "cost_mul": cost_mul,
            "cost_add": cost_add,
            "computational": computational,
        }

        # Actualizar bits de activación para la siguiente capa.
        # Solo las capas de cómputo (con MACs) transforman las activaciones;
        # las capas pass-through (Flatten, MaxPooling) propagan los bits sin cambio.
        if info["macs"] > 0:
            prev_act_bits = b_act_out

        total_spatial_bits += spatial_bits
        total_computational += computational
        layer_complexity.append(complexity)

    totals = {
        "total_params": ops_totals["total_params"],
        "total_macs": ops_totals["total_macs"],
        "total_spatial_bits": total_spatial_bits,
        "total_spatial_bytes": total_spatial_bits / 8,
        "total_computational": total_computational,
    }

    return layer_complexity, totals


def print_complexity_report(model):
    """
    Calcula e imprime un reporte completo de complejidad del modelo,
    incluyendo operaciones, parámetros, complejidad espacial y computacional.
    
    Los bits se extraen automáticamente de los cuantizadores QKeras del modelo.
    Para modelos Keras estándar (float32), se reportan 32 bits.

    Args:
        model: modelo Keras o QKeras
    """
    layer_complexity, totals = calculate_complexity(model)

    # Determinar si el modelo está cuantizado
    is_quantized = any(lc["bits_weights"] != 32 for lc in layer_complexity if lc["macs"] > 0)
    model_label = "Cuantizado" if is_quantized else "Float32"

    print("\n" + "=" * 145)
    print(f"REPORTE DE COMPLEJIDAD DEL MODELO ({model_label})")
    print("=" * 145)
    print()
    print(f"  {'Capa':<20s} | {'Tipo':<12s} | {'b_w':>4s} | {'b_act':>5s} | {'Params':>10s} | {'MACs':>12s} | "
          f"{'Mem [bits]':>12s} | {'Mem [bytes]':>12s} | {'Comp. Total':>14s}")
    print("  " + "-" * 135)

    for lc in layer_complexity:
        if lc["params"] == 0 and lc["macs"] == 0:
            print(f"  {lc['name']:<20s} | {lc['type']:<12s} | {'—':>4s} | {'—':>5s} | {'—':>10s} | {'—':>12s} | "
                  f"{'—':>12s} | {'—':>12s} | {'—':>14s}")
        else:
            print(f"  {lc['name']:<20s} | {lc['type']:<12s} | {lc['bits_weights']:>4d} | {lc['bits_activations']:>5d} | "
                  f"{lc['params']:>10,d} | {lc['macs']:>12,d} | "
                  f"{lc['spatial_bits']:>12,d} | {lc['spatial_bytes']:>12,.1f} | {lc['computational']:>14,d}")

    print("  " + "-" * 135)
    print(f"  {'TOTAL':<20s} | {'':<12s} | {'':>4s} | {'':>5s} | {totals['total_params']:>10,d} | {totals['total_macs']:>12,d} | "
          f"{totals['total_spatial_bits']:>12,d} | {totals['total_spatial_bytes']:>12,.1f} | {totals['total_computational']:>14,d}")
    print("=" * 145)

    # Comparación FP32 como referencia (solo si el modelo está cuantizado)
    if is_quantized:
        fp32_spatial_bits = totals["total_params"] * 32
        fp32_spatial_bytes = fp32_spatial_bits / 8
        fp32_cost_mul = 32 * 32
        fp32_cost_add = 32
        fp32_computational = totals["total_macs"] * (fp32_cost_mul + fp32_cost_add)

        reduction_spatial = (1 - totals["total_spatial_bits"] / fp32_spatial_bits) * 100 if fp32_spatial_bits > 0 else 0
        reduction_compute = (1 - totals["total_computational"] / fp32_computational) * 100 if fp32_computational > 0 else 0

        print(f"\n  Comparación con FP32 (32 bits pesos, 32 bits activaciones):")
        print(f"    Memoria FP32:  {fp32_spatial_bytes:>12,.1f} bytes  |  Memoria cuantizada:  {totals['total_spatial_bytes']:>12,.1f} bytes  |  Reducción: {reduction_spatial:.1f}%")
        print(f"    Comp. FP32:    {fp32_computational:>14,d}  |  Comp. cuantizada:    {totals['total_computational']:>14,d}  |  Reducción: {reduction_compute:.1f}%")
        print()

    return layer_complexity, totals
