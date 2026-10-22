"""
export_tflite.py — Export de un checkpoint .pth a TFLite full-integer int8.

Pipeline: .pth (torch) → ONNX → onnx2tf (NCHW→NHWC, saved_model) →
TFLiteConverter full-integer PTQ con representative dataset = mismas
imágenes de calibración STL-10 (y misma normalización) que script04a.

Outputs en out/<source>/:
  - <source>.tflite
  - model_data.h    (array C uint8 del .tflite + defines de input/output q)
  - golden_data.h   (10 golden images int8 NHWC + logits int8 de referencia)
  - model_meta.json (runtime, shapes, scale/zp, accuracy test set, tamaño)

Uso:
  uv run --project tools/rt_export/tflite python tools/rt_export/tflite/export_tflite.py \
      --source depthwise_rgb32_pruned_activation_30_ft2
"""

import argparse
import json
import subprocess
import sys
import tempfile
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
MARGIN = 0.15


# ---------------------------------------------------------------------------
# torch → ONNX → saved_model (onnx2tf) → tflite int8
# ---------------------------------------------------------------------------

def export_onnx(model, meta, onnx_path: Path) -> None:
    in_ch = 1 if meta["input_mode"] == "gray" else 3
    res = meta["resolution"]
    dummy = torch.zeros(1, in_ch, res, res)
    model.eval()
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        opset_version=13, do_constant_folding=True,
        dynamo=False,  # exporter legacy: opset 13 estable para onnx2tf
    )


def run_onnx2tf(onnx_path: Path, out_dir: Path) -> Path:
    """onnx2tf: convierte NCHW→NHWC y emite un saved_model."""
    # onnx2tf intenta descargar datos dummy para su inferencia de prueba;
    # si no hay red, usa este archivo si existe en el cwd. Contenido irrelevante.
    dummy = onnx_path.parent / \
        "calibration_image_sample_data_20x128x128x3_float32.npy"
    if not dummy.exists():
        np.save(dummy, np.random.rand(20, 128, 128, 3).astype(np.float32))
    cmd = [sys.executable, "-m", "onnx2tf",
           "-i", str(onnx_path), "-o", str(out_dir),
           "-osd",              # emitir saved_model (para TFLiteConverter)
           "-b", "1",           # batch fijo 1
           "--non_verbose"]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(onnx_path.parent))
    if r.returncode != 0:
        sys.stderr.write(r.stdout + "\n" + r.stderr + "\n")
        raise RuntimeError(f"onnx2tf falló (rc={r.returncode})")
    sm = out_dir / "saved_model"
    if not sm.exists():
        # algunas versiones escriben directo en out_dir
        sm = out_dir
    if not (sm / "saved_model.pb").exists():
        raise RuntimeError(f"saved_model no encontrado bajo {out_dir}")
    return sm


def get_calibration_images(meta, n_batches: int) -> np.ndarray:
    """Mismas imágenes de calibración que script04a: primeros n_batches del
    train_loader (batch 64, misma normalización _NORM_STATS). NHWC float32."""
    train_loader, _ = get_dataloaders(meta["input_mode"], meta["resolution"],
                                      batch_size=64)
    chunks = []
    for i, (x, _) in enumerate(train_loader):
        if i >= n_batches:
            break
        chunks.append(x.permute(0, 2, 3, 1).numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def convert_tflite_int8(saved_model_dir: Path, calib: np.ndarray) -> bytes:
    import tensorflow as tf

    def rep_dataset():
        for i in range(calib.shape[0]):
            yield [calib[i:i + 1]]

    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


# ---------------------------------------------------------------------------
# Interpreter helpers
# ---------------------------------------------------------------------------

def make_interpreter(tflite_bytes: bytes, num_threads: int = 8):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    it = Interpreter(model_content=tflite_bytes, num_threads=num_threads)
    it.allocate_tensors()
    return it


def quantize_input(x_nhwc_f32: np.ndarray, scale: float, zp: int) -> np.ndarray:
    q = np.round(x_nhwc_f32 / scale) + zp
    return np.clip(q, -128, 127).astype(np.int8)


def run_int8(it, x_int8: np.ndarray) -> np.ndarray:
    """x_int8: (1,H,W,C) int8 → logits int8 (n_classes,)"""
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    it.set_tensor(inp["index"], x_int8)
    it.invoke()
    return it.get_tensor(out["index"])[0].copy()


def evaluate_tflite(it, meta) -> float:
    inp = it.get_input_details()[0]
    scale, zp = float(inp["quantization"][0]), int(inp["quantization"][1])
    _, test_loader = get_dataloaders(meta["input_mode"], meta["resolution"],
                                     batch_size=256)
    correct = total = 0
    for xb, yb in test_loader:
        xb = xb.permute(0, 2, 3, 1).numpy().astype(np.float32)
        xq = quantize_input(xb, scale, zp)
        for i in range(xq.shape[0]):
            logits = run_int8(it, xq[i:i + 1])
            correct += int(np.argmax(logits) == int(yb[i]))
            total += 1
    return correct / total


# ---------------------------------------------------------------------------
# Golden images (mismo criterio que gen_test_images.py: candidatas por clase
# del modelo float, elegida final la primera que el modelo CUANTIZADO también
# clasifica bien con margen top1-top2 >= 0.15)
# ---------------------------------------------------------------------------

def select_candidates(model, meta, device, candidates_per_class: int = 32):
    """{class_idx: [(conf, img01_chw), ...]} orden det. (test_loader sin shuffle)."""
    mean_t = torch.tensor(meta["normalize_mean"]).view(3, 1, 1)
    std_t = torch.tensor(meta["normalize_std"]).view(3, 1, 1)
    _, test_loader = get_dataloaders("rgb", meta["resolution"], batch_size=128)
    best = {}
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            probs = torch.softmax(model(xb.to(device)), dim=1).cpu()
            conf, pred = probs.max(dim=1)
            for i in range(len(yb)):
                c = int(yb[i])
                if int(pred[i]) != c:
                    continue
                img01 = (xb[i] * std_t + mean_t).clamp(0, 1)
                lst = best.setdefault(c, [])
                lst.append((float(conf[i]), img01))
                lst.sort(key=lambda t: -t[0])
                del lst[candidates_per_class:]
    return best


def select_golden(model, meta, device, it):
    inp = it.get_input_details()[0]
    in_scale, in_zp = float(inp["quantization"][0]), int(inp["quantization"][1])
    out = it.get_output_details()[0]
    out_scale, out_zp = float(out["quantization"][0]), int(out["quantization"][1])

    mean = np.asarray(meta["normalize_mean"], dtype=np.float32)
    std = np.asarray(meta["normalize_std"], dtype=np.float32)

    best = select_candidates(model, meta, device)
    entries = []
    for c in sorted(best):
        chosen = fallback = None
        for conf, img01 in best[c]:
            nhwc = img01.permute(1, 2, 0).numpy().astype(np.float32)
            norm = (nhwc - mean) / std
            xq = quantize_input(norm[None, ...], in_scale, in_zp)
            logits_i8 = run_int8(it, xq)
            deq = (logits_i8.astype(np.float32) - out_zp) * out_scale
            p = np.exp(deq - deq.max()); p /= p.sum()
            order = np.argsort(-p)
            top1, margin = int(order[0]), float(p[order[0]] - p[order[1]])
            cand = (c, conf, top1, float(p[order[0]]), margin,
                    xq[0].flatten(), logits_i8)
            if fallback is None:
                fallback = cand
            if top1 == c and margin >= MARGIN:
                chosen = cand
                break
        cand = chosen or fallback
        entries.append(cand)
        c, conf, top1, qconf, margin = cand[:5]
        note = "" if chosen else "  [SIN MARGEN]"
        mark = "" if top1 == c else f"  (tflite predice {STL10_CLASSES[top1]})"
        print(f"  {STL10_CLASSES[c]:<9} conf_float={conf:.3f}  "
              f"conf_tfl={qconf:.3f}  margen={margin:.2f}{mark}{note}")
    return entries


# ---------------------------------------------------------------------------
# Emisión de headers / meta
# ---------------------------------------------------------------------------

def emit_model_data_h(path: Path, source: str, tfl: bytes, meta, it) -> None:
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    in_scale, in_zp = float(inp["quantization"][0]), int(inp["quantization"][1])
    out_scale, out_zp = float(out["quantization"][0]), int(out["quantization"][1])
    in_ch = 1 if meta["input_mode"] == "gray" else 3
    res = meta["resolution"]
    hexb = ", ".join(f"0x{b:02X}" for b in tfl)
    classes_str = ", ".join(f'"{c}"' for c in STL10_CLASSES)
    path.write_text(
        f"/* Generado por export_tflite.py — no editar. Fuente: {source}.pth */\n"
        f"#pragma once\n#include <stddef.h>\n#include <stdint.h>\n\n"
        f"#define MODEL_INPUT_W          {res}\n"
        f"#define MODEL_INPUT_H          {res}\n"
        f"#define MODEL_INPUT_CHANNELS   {in_ch}\n"
        f"/* input int8 NHWC: q = round(norm/scale) + zp */\n"
        f"#define MODEL_INPUT_SCALE      {in_scale:.10g}f\n"
        f"#define MODEL_INPUT_ZERO_POINT {in_zp}\n"
        f"#define MODEL_OUTPUT_SCALE     {out_scale:.10g}f\n"
        f"#define MODEL_OUTPUT_ZERO_POINT {out_zp}\n\n"
        f"#define MODEL_NUM_CLASSES      {len(STL10_CLASSES)}\n"
        f"static const char* const MODEL_CLASSES[{len(STL10_CLASSES)}] = "
        f"{{{classes_str}}};\n\n"
        f"static const uint8_t model_tflite[] __attribute__((aligned(16))) = "
        f"{{{hexb}}};\n"
        f"static const size_t model_tflite_len = {len(tfl)};\n",
        encoding="utf-8",
    )


def emit_golden_data_h(path: Path, source: str, entries, meta) -> None:
    res = meta["resolution"]
    lines = [
        f"/* Generado por export_tflite.py — no editar. Fuente: {source}.pth",
        " * Golden images STL-10 int8 NHWC (ya cuantizadas al input del modelo",
        " * TFLite) + logits int8 de referencia del interpreter de PC. */",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define GOLDEN_COUNT {len(entries)}",
        f"#define GOLDEN_W {res}",
        f"#define GOLDEN_H {res}",
        "#define GOLDEN_CH 3",
        "",
        "typedef struct {",
        "    const int8_t *data;        /* NHWC, W*H*CH bytes, ya cuantizado */",
        "    const int8_t *ref_logits;  /* logits int8 del interpreter PC */",
        "    uint8_t       expected;    /* clase verdadera */",
        "    uint8_t       ref_top1;    /* top-1 de referencia en PC */",
        "} golden_image_t;",
        "",
        f"static const char* const GOLDEN_CLASSES[{len(STL10_CLASSES)}] = "
        f"{{{', '.join(f'\"{c}\"' for c in STL10_CLASSES)}}};",
        "",
    ]
    for k, (c, _conf, top1, _qc, _m, flat, logits) in enumerate(entries):
        lines.append(f"static const int8_t GOLDEN_IMG_{k}[{len(flat)}] = "
                     f"{{{','.join(str(int(v)) for v in flat)}}};")
        lines.append(f"static const int8_t GOLDEN_LOGITS_{k}[{len(logits)}] = "
                     f"{{{','.join(str(int(v)) for v in logits)}}};")
    lines.append("")
    lines.append(f"static const golden_image_t GOLDEN_IMAGES[GOLDEN_COUNT] = {{")
    for k, (c, _conf, top1, *_rest) in enumerate(entries):
        lines.append(f"    {{ GOLDEN_IMG_{k}, GOLDEN_LOGITS_{k}, {c}, {top1} }},"
                     f"  /* {STL10_CLASSES[c]} */")
    lines.append("};")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True,
                    help="Nombre del checkpoint en models/ (sin .pth)")
    ap.add_argument("--calib-batches", type=int, default=32,
                    help="Batches (de 64) de calibración, como script04a (default: 32)")
    ap.add_argument("--skip-eval", action="store_true",
                    help="No evaluar accuracy en el test set completo")
    args = ap.parse_args()

    src = MODELS_DIR / f"{args.source}.pth"
    if not src.exists():
        raise FileNotFoundError(src)

    device = torch.device("cpu")
    model, meta = load_checkpoint(src, device)
    if meta["input_mode"] != "rgb":
        raise SystemExit("Solo input_mode=rgb soportado (golden images).")

    out_dir = OUT_ROOT / args.source
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        onnx_path = Path(td) / "model.onnx"
        export_onnx(model, meta, onnx_path)
        print(f"[INFO] ONNX exportado: {onnx_path}")
        sm = run_onnx2tf(onnx_path, Path(td) / "tf")
        print(f"[INFO] saved_model: {sm}")
        calib = get_calibration_images(meta, args.calib_batches)
        print(f"[INFO] Calibración: {calib.shape[0]} imágenes")
        tfl = convert_tflite_int8(sm, calib)

    tfl_path = out_dir / f"{args.source}.tflite"
    tfl_path.write_bytes(tfl)
    print(f"[INFO] TFLite int8: {tfl_path}  ({len(tfl)} bytes)")

    it = make_interpreter(tfl)
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    print(f"[INFO] input: shape={inp['shape'].tolist()} dtype={inp['dtype'].__name__} "
          f"q={inp['quantization']}")
    print(f"[INFO] output: q={out['quantization']}")

    print("[INFO] Seleccionando golden images (margen >= 0.15)...")
    entries = select_golden(model, meta, device, it)

    accuracy = None
    if not args.skip_eval:
        print("[INFO] Evaluando accuracy en test set completo...")
        accuracy = evaluate_tflite(it, meta)
        print(f"[INFO] Accuracy TFLite int8: {accuracy*100:.2f}%")

    emit_model_data_h(out_dir / "model_data.h", args.source, tfl, meta, it)
    emit_golden_data_h(out_dir / "golden_data.h", args.source, entries, meta)

    meta_json = {
        "source": args.source,
        "runtime": "tflm",
        "input_shape": inp["shape"].tolist(),
        "input_scale": float(inp["quantization"][0]),
        "input_zero_point": int(inp["quantization"][1]),
        "output_scale": float(out["quantization"][0]),
        "output_zero_point": int(out["quantization"][1]),
        "classes": STL10_CLASSES,
        "accuracy": accuracy,
        "tflite_size_bytes": len(tfl),
        "calib_batches": args.calib_batches,
        "golden_count": len(entries),
    }
    (out_dir / "model_meta.json").write_text(
        json.dumps(meta_json, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Outputs en {out_dir}/")


if __name__ == "__main__":
    main()
