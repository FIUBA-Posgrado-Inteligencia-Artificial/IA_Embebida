"""
export_onnx2c.py — Export de un checkpoint .pth a C plano float via onnx2c.

Pipeline: .pth (torch) → ONNX float (opset 13, batch 1) → onnxsim →
`onnx2c model.onnx > model.c`. La referencia de logits se calcula en PC con
onnxruntime sobre el MISMO model.onnx simplificado.

Corre con el uv env PRINCIPAL del ejemplo (ya tiene torch/onnx/onnxsim/
onnxruntime), NO tiene pyproject propio:

  uv run python tools/rt_export/onnx2c/export_onnx2c.py \
      --source depthwise_rgb32_pruned_activation_30_ft2

Requiere el binario onnx2c compilado (correr build.sh una vez).

Outputs en out/<source>/:
  - model.onnx      (simplificado, batch 1)
  - model.c         (generado por onnx2c; entrada float NCHW normalizada)
  - golden_data.h   (10 golden STL-10: float NCHW ya normalizados + logits
                     float de referencia de onnxruntime + class names)
  - model_meta.json (runtime, accuracy test set con onnxruntime, tamaños)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLE_DIR = HERE.parents[2]  # examples/esp32-cam
sys.path.insert(0, str(EXAMPLE_DIR / "pipelib"))

import torch  # noqa: E402

from esp32cam_utils import (  # noqa: E402
    MODELS_DIR, STL10_CLASSES, get_dataloaders, load_checkpoint,
)

OUT_ROOT = HERE / "out"
ONNX2C_BIN = HERE / "onnx2c-src" / "build" / "onnx2c"
MARGIN = 0.15


def export_onnx_simplified(model, meta, onnx_path: Path) -> None:
    import onnx
    from onnxsim import simplify

    in_ch = 1 if meta["input_mode"] == "gray" else 3
    res = meta["resolution"]
    dummy = torch.zeros(1, in_ch, res, res)
    model.eval()
    # dynamo=False: el exporter legacy respeta opset 13 y no emite atributos
    # que onnx2c no soporta (p.ej. MaxPool storage_order del exporter nuevo).
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        opset_version=13, do_constant_folding=True, dynamo=False,
    )
    m = onnx.load(str(onnx_path))
    m_simp, ok = simplify(m)
    if not ok:
        raise RuntimeError("onnxsim: simplify falló la validación")
    # cinturón: remover atributos que onnx2c no implementa
    for node in m_simp.graph.node:
        if node.op_type == "MaxPool":
            keep = [a for a in node.attribute if a.name != "storage_order"]
            del node.attribute[:]
            node.attribute.extend(keep)
    onnx.save(m_simp, str(onnx_path))


def run_onnx2c(onnx_path: Path, c_path: Path) -> None:
    if not ONNX2C_BIN.exists():
        raise SystemExit(f"No existe {ONNX2C_BIN} — correr build.sh primero")
    r = subprocess.run([str(ONNX2C_BIN), str(onnx_path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr + "\n")
        raise RuntimeError(f"onnx2c falló (rc={r.returncode})")
    c_path.write_text(r.stdout, encoding="utf-8")
    # sanity: la entry function que espera el firmware
    if "void entry(" not in r.stdout:
        raise RuntimeError("model.c no contiene 'void entry(' — revisar onnx2c")


def make_ort_session(onnx_path: Path):
    import onnxruntime as ort
    return ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])


def ort_logits(sess, x_nchw: np.ndarray) -> np.ndarray:
    """El modelo tiene batch fijo 1: itera de a una imagen."""
    x = x_nchw.astype(np.float32)
    out = [sess.run(None, {"input": x[i:i + 1]})[0][0]
           for i in range(x.shape[0])]
    return np.stack(out, axis=0)


def evaluate_ort(sess, meta) -> float:
    _, test_loader = get_dataloaders(meta["input_mode"], meta["resolution"],
                                     batch_size=256)
    correct = total = 0
    for xb, yb in test_loader:
        logits = ort_logits(sess, xb.numpy())
        correct += int((np.argmax(logits, axis=1) == yb.numpy()).sum())
        total += len(yb)
    return correct / total


def select_golden(meta, sess):
    """Una golden por clase: primera del test set (orden determinista, sin
    shuffle) que onnxruntime clasifica bien con margen top1-top2 >= 0.15.
    Devuelve [(clase, x_nchw_norm float32, logits float32)]."""
    _, test_loader = get_dataloaders(meta["input_mode"], meta["resolution"],
                                     batch_size=128)
    chosen: dict[int, tuple] = {}
    fallback: dict[int, tuple] = {}
    for xb, yb in test_loader:
        logits = ort_logits(sess, xb.numpy())
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        for i in range(len(yb)):
            c = int(yb[i])
            if c in chosen:
                continue
            order = np.argsort(-probs[i])
            top1 = int(order[0])
            margin = float(probs[i][order[0]] - probs[i][order[1]])
            cand = (c, xb[i].numpy().copy(), logits[i].copy())
            if top1 == c:
                fallback.setdefault(c, cand)
                if margin >= MARGIN:
                    chosen[c] = cand
        if len(chosen) == len(STL10_CLASSES):
            break
    entries = []
    for c in range(len(STL10_CLASSES)):
        cand = chosen.get(c) or fallback.get(c)
        if cand is None:
            raise RuntimeError(f"Sin golden para clase {STL10_CLASSES[c]}")
        note = "" if c in chosen else "  [SIN MARGEN]"
        print(f"  {STL10_CLASSES[c]:<9} logit_top1="
              f"{cand[2].max():.3f}{note}")
        entries.append(cand)
    return entries


def emit_golden_data_h(path: Path, source: str, entries, meta) -> None:
    res = meta["resolution"]
    in_ch = 1 if meta["input_mode"] == "gray" else 3
    n = len(entries)
    nc = len(STL10_CLASSES)
    lines = [
        f"/* Generado por export_onnx2c.py — no editar. Fuente: {source}.pth",
        " * Golden images STL-10 float32 NCHW YA normalizadas (layout que",
        " * espera la entry() de onnx2c) + logits float de onnxruntime. */",
        "#pragma once",
        "",
        f"#define GOLDEN_COUNT {n}",
        f"#define GOLDEN_W {res}",
        f"#define GOLDEN_H {res}",
        f"#define GOLDEN_CH {in_ch}",
        f"#define GOLDEN_NUM_CLASSES {nc}",
        "",
        "static const char* const GOLDEN_CLASSES[%d] = {%s};"
        % (nc, ", ".join('"%s"' % c for c in STL10_CLASSES)),
        "",
    ]
    sz = in_ch * res * res
    for k, (c, x, logits) in enumerate(entries):
        flat = x.reshape(-1)
        assert flat.size == sz
        vals = ",".join(f"{v:.8g}f" for v in flat)
        lines.append(f"static const float GOLDEN_IMG_{k}[{sz}] = {{{vals}}};")
        lv = ",".join(f"{v:.8g}f" for v in logits)
        lines.append(f"static const float GOLDEN_LOGITS_{k}[{nc}] = {{{lv}}};")
    lines += [
        "",
        "typedef struct {",
        "    const float *data;        /* NCHW, CH*H*W floats, normalizado */",
        "    const float *ref_logits;  /* logits float de onnxruntime (PC) */",
        "    unsigned char expected;   /* clase verdadera */",
        "} golden_image_f32_t;",
        "",
        "static const golden_image_f32_t GOLDEN_IMAGES[GOLDEN_COUNT] = {",
    ]
    for k, (c, _x, _l) in enumerate(entries):
        lines.append(f"    {{ GOLDEN_IMG_{k}, GOLDEN_LOGITS_{k}, {c} }},"
                     f"  /* {STL10_CLASSES[c]} */")
    lines += ["};", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True,
                    help="Nombre del checkpoint en models/ (sin .pth)")
    ap.add_argument("--skip-eval", action="store_true",
                    help="No evaluar accuracy en el test set completo")
    args = ap.parse_args()

    src = MODELS_DIR / f"{args.source}.pth"
    if not src.exists():
        raise FileNotFoundError(src)
    model, meta = load_checkpoint(src, torch.device("cpu"))
    if meta["input_mode"] != "rgb":
        raise SystemExit("Solo input_mode=rgb soportado (golden images).")

    out_dir = OUT_ROOT / args.source
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = out_dir / "model.onnx"
    export_onnx_simplified(model, meta, onnx_path)
    print(f"[INFO] ONNX simplificado: {onnx_path} "
          f"({onnx_path.stat().st_size} bytes)")

    c_path = out_dir / "model.c"
    run_onnx2c(onnx_path, c_path)
    print(f"[INFO] model.c: {c_path} ({c_path.stat().st_size} bytes)")

    sess = make_ort_session(onnx_path)
    print("[INFO] Seleccionando golden images (margen >= 0.15)...")
    entries = select_golden(meta, sess)
    emit_golden_data_h(out_dir / "golden_data.h", args.source, entries, meta)

    accuracy = None
    if not args.skip_eval:
        print("[INFO] Evaluando accuracy con onnxruntime...")
        accuracy = evaluate_ort(sess, meta)
        print(f"[INFO] Accuracy onnxruntime float: {accuracy*100:.2f}%")

    meta_json = {
        "source": args.source,
        "runtime": "onnx2c",
        "input_shape": [1, 1 if meta["input_mode"] == "gray" else 3,
                        meta["resolution"], meta["resolution"]],
        "layout": "NCHW",
        "dtype": "float32",
        "classes": STL10_CLASSES,
        "accuracy": accuracy,
        "onnx_size_bytes": onnx_path.stat().st_size,
        "model_c_size_bytes": c_path.stat().st_size,
        "golden_count": len(entries),
    }
    (out_dir / "model_meta.json").write_text(
        json.dumps(meta_json, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Outputs en {out_dir}/")


if __name__ == "__main__":
    main()
