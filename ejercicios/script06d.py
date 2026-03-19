import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_results(json_path):
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Detectar formato (viejo vs nuevo)
    if "results" in data:
        # Formato Nuevo
        model_name = data.get("model_name", "model")
        float_acc = data.get("float_acc", 0.0)
        results = data.get("results", {})
        strategies = data.get("strategies", [])
    else:
        # Formato Viejo (solo dict de resultados)
        results = data
        float_acc = 0.0 # No estaba guardado antes
        strategies = [] # Habrá que inferirlas de los resultados
        # Intentar sacar el nombre del modelo del path del archivo
        model_name = os.path.basename(json_path).replace("_resultados_cuantizacion.json", "")
        
        # Inferir estrategias de las keys
        strat_set = set()
        for k in results.keys():
            parts = k.split('_')
            if len(parts) >= 3:
                strat_set.add((parts[1], parts[2]))
        strategies = sorted(list(strat_set))

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n" + "="*80)
    print(f"CARGANDO DATOS DESDE: {json_path}")
    print(f"MODELO: {model_name} ({'Formato Nuevo' if 'results' in data else 'Formato Viejo'})")
    print(f"FLOAT32 ACCURACY: {float_acc:.4f}")
    print("="*80)

    # --- Mostrar Tabla de Resultados ---
    print("\n" + "="*100)
    print(f"TABLA DE RESULTADOS (Cargada de JSON)")
    print("="*100)
    print(f"{'Bits':>5s} | {'Scope':>6s} | {'Metrica':>7s} | {'Acc PTQ':>8s} | {'Acc QAT':>8s} | {'Delta Orig':>10s}")
    print("-" * 100)

    # Ordenar las keys para que la tabla sea legible
    sorted_keys = sorted(results.keys(), key=lambda x: (int(x.split('_')[0]), x))

    for key in sorted_keys:
        res = results[key]
        if "error" in res:
            print(f"{res.get('bits', ''):>5} | {res.get('scope', ''):>6s} | {res.get('metric', ''):>7s} | ERROR: {res['error']}")
        else:
            b = res["bits"]
            scope = res["scope"]
            metric = res["metric"]
            ptq = res["ptq_acc"]
            qat = res["qat_acc"]
            delta_orig = qat - float_acc
            print(f"{b:5d} | {scope:6s} | {metric:7s} | {ptq:8.4f} | {qat:8.4f} | {delta_orig:+10.4f}")
    print("-" * 100 + "\n")

    # --- Generar Plot ---
    plt.figure(figsize=(14, 9))
    
    # Obtener el ciclo de colores por defecto para repetir colores entre PTQ y QAT
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i, (s_scope, s_metric) in enumerate(strategies):
        bits_plot = []
        ptq_plot = []
        qat_plot = []
        
        # Filtrar resultados para esta estrategia
        for bits_key, res in results.items():
            if res.get("scope") == s_scope and res.get("metric") == s_metric:
                bits_plot.append(res["bits"])
                ptq_plot.append(res.get("ptq_acc"))
                qat_plot.append(res.get("qat_acc"))
        
        if bits_plot:
            # Ordenar por bits para que la linea se vea bien
            sorted_indices = np.argsort(bits_plot)
            bp = np.array(bits_plot)[sorted_indices]
            pp = np.array(ptq_plot)[sorted_indices]
            qp = np.array(qat_plot)[sorted_indices]
            
            color = colors[i % len(colors)]
            
            # Plot QAT (Solid)
            plt.plot(bp, qp, marker='o', linestyle='-', color=color, 
                     label=f'QAT: {s_scope}-{s_metric}')
            
            # Plot PTQ (Dashed) - mismos colores
            plt.plot(bp, pp, marker='x', linestyle='--', color=color, alpha=0.6,
                     label=f'PTQ: {s_scope}-{s_metric}')

    plt.axhline(y=float_acc, color='black', linestyle=':', linewidth=2, label=f'Baseline Float32 ({float_acc:.4f})')
    
    plt.xlabel('Bits de Cuantización')
    plt.ylabel('Accuracy')
    plt.title(f'Comparación de Accuracy: PTQ vs QAT vs Baseline\nModelo: {model_name}')
    
    # Configurar xticks basados en los bits encontrados
    all_bits = sorted(list(set([res["bits"] for res in results.values()])))
    plt.xticks(all_bits)
    
    # Mover la leyenda afuera si son muchas
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'{model_name}_grafico_accuracy_evolucion.png')
    plt.savefig(plot_path)
    print(f"\n[INFO] Gráfico guardado en '{plot_path}'")
    plt.show()
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Intentar buscar el archivo por defecto de script02
        path = "outputs/modelo_MLP_imagenes_fashion_resultados_cuantizacion.json"
        
    plot_results(path)
