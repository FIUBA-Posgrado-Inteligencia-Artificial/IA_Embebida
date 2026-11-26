"""
evaluar.py — evaluador oficial del TP final (competencia CIFAR-100, ESP32-S3).

Uso:
    uv run python evaluar.py modelo.onnx                 # evaluación completa
    uv run python evaluar.py modelo.onnx --rapido        # sólo métricas estáticas (sin cuantizar)
    uv run python evaluar.py modelo.onnx --json res.json

Qué hace, en orden:
  1. Verifica la interfaz: una entrada float [1,3,32,32] (RGB normalizado, ver
     datos.py) y una salida de 100 logits.
  2. Simplifica el grafo (onnxsim) y verifica que todos los ops estén en la
     lista OPS_PERMITIDOS (ops que ESP-DL soporta).
  3. Calcula MACs y pico de activaciones sobre el grafo (ver más abajo).
  4. Cuantiza a int8 con esp-ppq para el ESP32-S3 (PTQ, configuración fija,
     calibración con las primeras 512 imágenes de val) y exporta el .espdl.
     Tamaño de pesos = tamaño del archivo .espdl (lo que va a la flash).
  5. Mide (en CPU: en GPU la simulación int8 da décimas distintas) la accuracy
     top-1 del modelo int8 SIMULADO sobre el test oficial
     (10 000 imágenes) y, como referencia, la del modelo float (onnxruntime).
  6. Informa el nivel alcanzado (los niveles son independientes: puntos de la
     frontera de Pareto), si compite por el 10 (cumple la fila de Bronce) y el score.

Definiciones (son las reglas del juego; no pretenden ser idénticas al
consumo real de ESP-DL, sino una medida simple y reproducible):

  MACs: sólo Conv, Gemm y MatMul. Conv = elementos de salida × (Cin/grupos) × kh × kw.

  Pico de activaciones: se recorre el grafo en orden; en cada nodo se suman los
  bytes (int8, 1 byte por valor, batch 1) de todos los tensores "vivos" (ya
  producidos y que algún nodo posterior todavía consume, incluida la entrada
  del modelo) más los que produce ese nodo. Las activaciones elementwise
  (Relu, Clip, Sigmoid, ...), BatchNorm y los reshapes se cuentan en el lugar
  (no ocupan un buffer nuevo). El pico es el máximo sobre todos los nodos.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

# ---------------------------------------------------------------------------
# Reglas del TP
# ---------------------------------------------------------------------------

# Las reglas numéricas (baseline, niveles) viven en reglas.json.
REGLAS_JSON = Path(__file__).resolve().parent / "reglas.json"
_reglas = json.loads(REGLAS_JSON.read_text()) if REGLAS_JSON.exists() else {}
BASE_BYTES = _reglas.get("base_bytes")
BASE_MACS = _reglas.get("base_macs")
# Niveles: (nombre, pesos_max_bytes, activ_max_bytes, macs_max, acc_min)
NIVELES: list[tuple[str, int, int, int, float]] = [
    (n["nombre"], n["pesos_max"], n["activ_max"], n["macs_max"], n["acc_min"])
    for n in _reglas.get("niveles", [])]

OPS_PERMITIDOS = {
    "Conv", "Gemm", "MatMul", "Add", "Sub", "Mul",
    "Relu", "Clip", "LeakyRelu", "PRelu", "Sigmoid", "HardSigmoid", "HardSwish", "Tanh",
    "MaxPool", "AveragePool", "GlobalAveragePool",
    "Concat", "Flatten", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Pad",
    "Softmax", "BatchNormalization", "Identity",
}

# Ops que no ocupan buffer nuevo (se cuentan "en el lugar").
OPS_EN_LUGAR = {
    "Relu", "Clip", "LeakyRelu", "PRelu", "Sigmoid", "HardSigmoid", "HardSwish", "Tanh",
    "BatchNormalization", "Identity", "Flatten", "Reshape", "Squeeze", "Unsqueeze", "Dropout",
}

N_CALIB = 512


# ---------------------------------------------------------------------------
# Análisis estático del grafo
# ---------------------------------------------------------------------------

def simplificar(modelo: onnx.ModelProto) -> onnx.ModelProto:
    try:
        from onnxsim import simplify
        m, ok = simplify(modelo)
        if ok:
            return m
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] onnxsim falló ({e}); sigo con el grafo original")
    return modelo


def _shapes(modelo: onnx.ModelProto) -> dict[str, list[int]]:
    m = shape_inference.infer_shapes(modelo)
    out = {}
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
        out[vi.name] = dims
    return out


def verificar_interfaz(modelo: onnx.ModelProto) -> list[str]:
    errores = []
    inits = {i.name for i in modelo.graph.initializer}
    entradas = [i for i in modelo.graph.input if i.name not in inits]
    if len(entradas) != 1:
        errores.append(f"se espera 1 entrada, hay {len(entradas)}")
    else:
        dims = [d.dim_value for d in entradas[0].type.tensor_type.shape.dim]
        if dims[1:] != [3, 32, 32] or dims[0] not in (0, 1):
            errores.append(f"la entrada debe ser [1,3,32,32], es {dims}")
    if len(modelo.graph.output) != 1:
        errores.append(f"se espera 1 salida, hay {len(modelo.graph.output)}")
    else:
        dims = [d.dim_value for d in modelo.graph.output[0].type.tensor_type.shape.dim]
        if dims[-1] != 100:
            errores.append(f"la salida debe tener 100 logits, es {dims}")
    return errores


def ops_no_permitidos(modelo: onnx.ModelProto) -> set[str]:
    return {n.op_type for n in modelo.graph.node} - OPS_PERMITIDOS


def contar_macs(modelo: onnx.ModelProto, shapes: dict) -> int:
    inits = {i.name: numpy_helper.to_array(i) for i in modelo.graph.initializer}
    total = 0
    for n in modelo.graph.node:
        if n.op_type == "Conv":
            w = inits.get(n.input[1])
            wshape = list(w.shape) if w is not None else shapes[n.input[1]]
            out = shapes[n.output[0]]
            total += int(np.prod(out[1:])) * int(np.prod(wshape[1:]))
        elif n.op_type in ("Gemm", "MatMul"):
            a = shapes[n.input[0]]
            out = shapes[n.output[0]]
            k = a[-1]
            if n.op_type == "Gemm":
                trans_a = next((at.i for at in n.attribute if at.name == "transA"), 0)
                if trans_a:
                    k = a[0]
            total += int(np.prod(out)) * int(k)
    return total


def pico_activaciones(modelo: onnx.ModelProto, shapes: dict) -> tuple[int, str]:
    """Pico de bytes vivos (int8, batch 1). Devuelve (bytes, nodo donde ocurre)."""
    g = modelo.graph
    inits = {i.name for i in g.initializer} | {
        o for n in g.node if n.op_type == "Constant" for o in n.output}
    nodos = [n for n in g.node if n.op_type != "Constant"]

    def tam(t):
        return int(np.prod(shapes.get(t, [0])))

    # Raíz de buffer: los ops "en el lugar" reutilizan el buffer de su 1.ª entrada.
    raiz: dict[str, str] = {}
    for inp in g.input:
        if inp.name not in inits:
            raiz[inp.name] = inp.name
    for n in nodos:
        for o in n.output:
            if n.op_type in OPS_EN_LUGAR and n.input and n.input[0] in raiz:
                raiz[o] = raiz[n.input[0]]
            else:
                raiz[o] = o

    # Último uso de cada buffer raíz.
    ultimo: dict[str, int] = {}
    for i, n in enumerate(nodos):
        for t in n.input:
            if t in raiz:
                ultimo[raiz[t]] = max(ultimo.get(raiz[t], -1), i)
    for o in g.output:
        ultimo[raiz[o.name]] = len(nodos)

    definido: dict[str, int] = {r: -1 for r in raiz if raiz[r] == r and r not in {
        o for n in nodos for o in n.output}}
    for i, n in enumerate(nodos):
        for o in n.output:
            if raiz[o] == o:
                definido[o] = i

    pico, donde = 0, ""
    for i, n in enumerate(nodos):
        if n.op_type in OPS_EN_LUGAR:
            continue
        vivos = sum(tam(b) for b, d in definido.items() if d < i and ultimo.get(b, -1) >= i)
        nuevos = sum(tam(o) for o in n.output if raiz[o] == o)
        if vivos + nuevos > pico:
            pico, donde = vivos + nuevos, f"{n.op_type} {n.name}"
    return pico, donde


def contar_parametros(modelo: onnx.ModelProto) -> int:
    return int(sum(np.prod(i.dims) for i in modelo.graph.initializer if len(i.dims) > 0))


# ---------------------------------------------------------------------------
# Cuantización y accuracy
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def silencio(activo: bool = True):
    """Manda a un archivo temporal todo lo que se imprima (también lo que
    escriben esp-ppq, tqdm y las bibliotecas en C). Si algo falla, muestra el
    log capturado para poder ver el error."""
    if not activo:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    with tempfile.TemporaryFile(mode="w+b") as log:
        guardados = [os.dup(1), os.dup(2)]
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            yield
        except BaseException:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(guardados[0], 1)
            os.dup2(guardados[1], 2)
            log.seek(0)
            sys.stderr.write(log.read().decode(errors="replace"))
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(guardados[0], 1)
            os.dup2(guardados[1], 2)
            for fd in guardados:
                os.close(fd)


def cuantizar(ruta_onnx: Path, ruta_espdl: Path, calib: list, device: str):
    from esp_ppq.api import espdl_quantize_onnx
    graph = espdl_quantize_onnx(
        onnx_import_file=str(ruta_onnx),
        espdl_export_file=str(ruta_espdl),
        calib_dataloader=calib,
        calib_steps=len(calib),
        input_shape=[1, 3, 32, 32],
        target="esp32s3",
        num_of_bits=8,
        collate_fn=lambda x: x.to(device),
        device=device,
        error_report=False,
        export_test_values=False,
        verbose=0,
    )
    return graph


def accuracy_ppq(graph, x_test, y_test, device: str, bs: int = 250) -> float:
    import torch
    from esp_ppq.executor import TorchExecutor
    ex = TorchExecutor(graph=graph, device=device)
    correctos = 0
    try:
        for i in range(0, len(x_test), bs):
            xb = x_test[i:i + bs].to(device)
            out = ex.forward(inputs=xb)[0]
            correctos += (out.argmax(1).cpu() == y_test[i:i + bs]).sum().item()
    except Exception:  # noqa: BLE001 — grafo con batch fijo: se cae a batch 1
        correctos = 0
        with torch.no_grad():
            for i in range(len(x_test)):
                out = ex.forward(inputs=x_test[i:i + 1].to(device))[0]
                correctos += int(out.argmax(1).item() == y_test[i].item())
    return correctos / len(x_test)


def accuracy_float(ruta_onnx: Path, x_test, y_test) -> float:
    import onnxruntime as ort
    s = ort.InferenceSession(str(ruta_onnx), providers=["CPUExecutionProvider"])
    nombre = s.get_inputs()[0].name
    x = x_test.numpy()
    correctos = 0
    for i in range(len(x)):
        out = s.run(None, {nombre: x[i:i + 1]})[0]
        correctos += int(out.argmax() == int(y_test[i]))
    return correctos / len(x)


# ---------------------------------------------------------------------------
# Reglas: requisitos, nivel y score
# ---------------------------------------------------------------------------

def cumple(r: dict, nivel: tuple) -> bool:
    _, pmax, amax, mmax, accmin = nivel
    return (r["pesos_bytes"] <= pmax and r["pico_activaciones_bytes"] <= amax
            and r["macs"] <= mmax and r["acc_int8"] >= accmin)


def nivel_alcanzado(r: dict) -> str | None:
    """Niveles independientes (puntos de la frontera de Pareto): devuelve el
    más alto cuya fila completa se cumple, aunque no se cumplan los anteriores."""
    alcanzado = None
    for nivel in NIVELES:
        if cumple(r, nivel):
            alcanzado = nivel[0]
    return alcanzado


def compite(r: dict) -> bool:
    """Compiten por el 10 sólo las entregas que cumplen la fila de Bronce
    (incluida su accuracy mínima)."""
    return bool(NIVELES) and cumple(r, NIVELES[0])


def score(r: dict) -> float | None:
    if BASE_BYTES is None or BASE_MACS is None:
        return None
    return 0.5 * r["pesos_bytes"] / BASE_BYTES + 0.5 * r["macs"] / BASE_MACS


def kb(b):
    return f"{b / 1024:.1f} KB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--rapido", action="store_true", help="sólo métricas estáticas, sin cuantizar")
    ap.add_argument("--json", type=Path, help="guardar resultados en JSON")
    ap.add_argument("--espdl", type=Path, help="dónde guardar el .espdl (default: junto al .onnx)")
    ap.add_argument("--n-test", type=int, default=10000, help="imágenes de test (default: todas)")
    ap.add_argument("--sin-float", action="store_true", help="no medir la accuracy float")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default, el oficial) o cuda: en GPU es más rápido pero la accuracy int8 "
                         "difiere en décimas; el resultado oficial es el de CPU")
    ap.add_argument("--verbose", action="store_true", help="mostrar el log completo de esp-ppq")
    args = ap.parse_args()

    modelo = onnx.load(str(args.onnx))
    errores = verificar_interfaz(modelo)
    modelo = simplificar(modelo)
    malos = ops_no_permitidos(modelo)
    if malos:
        errores.append(f"ops no permitidos: {sorted(malos)}")
    shapes = _shapes(modelo)

    r: dict = {"modelo": args.onnx.name}
    r["parametros"] = contar_parametros(modelo)
    r["macs"] = contar_macs(modelo, shapes)
    r["pico_activaciones_bytes"], r["pico_en"] = pico_activaciones(modelo, shapes)

    print(f"\n=== {args.onnx.name} ===")
    print(f"Parámetros:            {r['parametros']:,}  (~{kb(r['parametros'])} en int8)")
    print(f"MACs:                  {r['macs'] / 1e6:.2f} M")
    print(f"Pico de activaciones:  {kb(r['pico_activaciones_bytes'])}  (en {r['pico_en']})")

    if errores:
        r["errores"] = errores
        for e in errores:
            print(f"[ERROR] {e}")

    if not args.rapido and not errores:
        import torch
        from datos import cargar_splits, normalizar
        device = args.device
        d = cargar_splits()
        xv = normalizar(d["val"][0][:N_CALIB])
        calib = [xv[i:i + 32] for i in range(0, N_CALIB, 32)]
        xt, yt = normalizar(d["test"][0][:args.n_test]), d["test"][1][:args.n_test]

        ruta_espdl = args.espdl or args.onnx.with_suffix(".espdl")
        with tempfile.TemporaryDirectory() as tmp:
            ruta_simpl = Path(tmp) / "simpl.onnx"
            onnx.save(modelo, str(ruta_simpl))
            print(f"Cuantizando a int8 con esp-ppq y midiendo accuracy en {len(xt)} imágenes de test...",
                  flush=True)
            with silencio(not args.verbose):
                graph = cuantizar(ruta_simpl, ruta_espdl, calib, device)
                r["pesos_bytes"] = ruta_espdl.stat().st_size
                r["acc_int8"] = accuracy_ppq(graph, xt, yt, device)
                if not args.sin_float:
                    r["acc_float"] = accuracy_float(ruta_simpl, xt, yt)

        print(f"Pesos (.espdl):        {kb(r['pesos_bytes'])}  ({ruta_espdl})")
        if "acc_float" in r:
            print(f"Accuracy float:        {100 * r['acc_float']:.2f} %")
        print(f"Accuracy int8 (sim):   {100 * r['acc_int8']:.2f} %"
              + ("" if device == "cpu" else f"  (en {device}: NO oficial, el oficial es en CPU)"))

        if NIVELES:
            r["nivel"] = nivel_alcanzado(r)
            r["compite"] = compite(r)
            print(f"Nivel:                 {r['nivel'] or 'ninguno'}")
            print(f"Compite por el 10:     {'sí' if r['compite'] else 'no (hay que cumplir la fila de Bronce)'}")
        s = score(r)
        if s is not None:
            r["score"] = s
            print(f"Score:                 {s:.4f}  (menor es mejor)")

    if args.json:
        args.json.write_text(json.dumps(r, indent=2, ensure_ascii=False))
    return 1 if errores else 0


if __name__ == "__main__":
    sys.exit(main())
