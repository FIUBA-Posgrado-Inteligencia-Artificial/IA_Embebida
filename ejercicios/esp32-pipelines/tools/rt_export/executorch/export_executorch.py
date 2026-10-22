"""
export_executorch.py — Export de un checkpoint .pth a ExecuTorch .pte portable.

Pipeline: .pth (torch) → torch.export.export → to_edge → to_executorch
(float, kernels portables, SIN delegates/XNNPACK) → out/<source>/:

  - model.pte
  - model_pte.h    (array C uint8 del .pte)
  - golden_data.h  (10 golden images STL-10 float32 NCHW normalizadas +
                    logits float de referencia del runtime ET en PC)
  - model_meta.json (runtime, shapes, accuracy test set con el runtime ET,
                     tamaño .pte)

La referencia de logits se calcula con el MISMO runtime (pybindings de
executorch.runtime, kernels portables) para que la comparación en device
sea runtime-vs-runtime, no runtime-vs-eager.

Uso:
  uv run --project tools/rt_export/executorch python \
      tools/rt_export/executorch/export_executorch.py \
      --source depthwise_rgb32_pruned_activation_30_ft2
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE_DIR = HERE.parents[2]  # examples/esp32-cam
sys.path.insert(0, str(EXAMPLE_DIR / "pipelib"))


def _ensure_libpython() -> None:
    """Las pybindings de executorch linkean libpython3.12.so dinámica; el
    cpython de uv la tiene en <python>/lib pero no en el search path default.
    Si falta, re-ejecutamos con LD_LIBRARY_PATH apuntando ahí."""
    try:
        import executorch.extension.pybindings._portable_lib  # noqa: F401
        return
    except ImportError as e:
        if "libpython" not in str(e):
            raise
    libdir = Path(sys.executable).resolve().parents[1] / "lib"
    if os.environ.get("_ET_REEXEC"):
        raise RuntimeError(f"No pude cargar libpython desde {libdir}")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"
    env["_ET_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


_ensure_libpython()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from executorch.exir import to_edge  # noqa: E402
from executorch.runtime import Runtime  # noqa: E402

from esp32cam_utils import (  # noqa: E402
    MODELS_DIR, STL10_CLASSES, get_dataloaders, load_checkpoint,
)

OUT_ROOT = HERE / "out"
MARGIN = 0.15


# ---------------------------------------------------------------------------
# Export a .pte (float, portable)
# ---------------------------------------------------------------------------

def export_pte(model, meta) -> bytes:
    in_ch = 1 if meta["input_mode"] == "gray" else 3
    res = meta["resolution"]
    example = (torch.zeros(1, in_ch, res, res),)
    model.eval()
    with torch.no_grad():
        ep = torch.export.export(model, example)
    prog = to_edge(ep).to_executorch()
    return prog.buffer


def load_et_method(pte_bytes: bytes):
    rt = Runtime.get()
    program = rt.load_program(pte_bytes)
    return program.load_method("forward")


def et_logits(method, x_nchw_f32: np.ndarray) -> np.ndarray:
    out = method.execute([torch.from_numpy(x_nchw_f32)])
    return out[0].numpy()[0].copy()


# ---------------------------------------------------------------------------
# Accuracy en test set con el runtime ET
# ---------------------------------------------------------------------------

def evaluate_et(method, meta, limit: int | None) -> float:
    _, test_loader = get_dataloaders(meta["input_mode"], meta["resolution"],
                                     batch_size=256)
    correct = total = 0
    for xb, yb in test_loader:
        x = xb.numpy().astype(np.float32)
        for i in range(x.shape[0]):
            logits = et_logits(method, x[i:i + 1])
            correct += int(np.argmax(logits) == int(yb[i]))
            total += 1
            if limit and total >= limit:
                return correct / total
    return correct / total


# ---------------------------------------------------------------------------
# Golden images (mismo criterio que export_tflite.py, pero float: la
# referencia y el margen se evalúan con el runtime ET)
# ---------------------------------------------------------------------------

def select_candidates(model, meta, candidates_per_class: int = 32):
    mean_t = torch.tensor(meta["normalize_mean"]).view(3, 1, 1)
    std_t = torch.tensor(meta["normalize_std"]).view(3, 1, 1)
    _, test_loader = get_dataloaders("rgb", meta["resolution"], batch_size=128)
    best = {}
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            probs = torch.softmax(model(xb), dim=1)
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


def select_golden(model, meta, method):
    mean = np.asarray(meta["normalize_mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(meta["normalize_std"], dtype=np.float32).reshape(3, 1, 1)

    best = select_candidates(model, meta)
    entries = []
    for c in sorted(best):
        chosen = fallback = None
        for conf, img01 in best[c]:
            norm = ((img01.numpy().astype(np.float32) - mean) / std)
            logits = et_logits(method, norm[None, ...])
            p = np.exp(logits - logits.max()); p /= p.sum()
            order = np.argsort(-p)
            top1, margin = int(order[0]), float(p[order[0]] - p[order[1]])
            cand = (c, conf, top1, float(p[order[0]]), margin,
                    norm.flatten(), logits)
            if fallback is None:
                fallback = cand
            if top1 == c and margin >= MARGIN:
                chosen = cand
                break
        cand = chosen or fallback
        entries.append(cand)
        c, conf, top1, qconf, margin = cand[:5]
        note = "" if chosen else "  [SIN MARGEN]"
        mark = "" if top1 == c else f"  (ET predice {STL10_CLASSES[top1]})"
        print(f"  {STL10_CLASSES[c]:<9} conf_float={conf:.3f}  "
              f"conf_et={qconf:.3f}  margen={margin:.2f}{mark}{note}")
    return entries


# ---------------------------------------------------------------------------
# Emisión de headers / meta
# ---------------------------------------------------------------------------

def emit_model_pte_h(path: Path, source: str, pte: bytes, meta) -> None:
    in_ch = 1 if meta["input_mode"] == "gray" else 3
    res = meta["resolution"]
    hexb = ", ".join(f"0x{b:02X}" for b in pte)
    classes_str = ", ".join(f'"{c}"' for c in STL10_CLASSES)
    path.write_text(
        f"/* Generado por export_executorch.py — no editar. Fuente: {source}.pth */\n"
        f"#pragma once\n#include <stddef.h>\n#include <stdint.h>\n\n"
        f"#define MODEL_INPUT_W        {res}\n"
        f"#define MODEL_INPUT_H        {res}\n"
        f"#define MODEL_INPUT_CHANNELS {in_ch}\n"
        f"/* input float32 NCHW ya normalizado ((x-mean)/std) */\n"
        f"#define MODEL_NUM_CLASSES    {len(STL10_CLASSES)}\n"
        f"static const char* const MODEL_CLASSES[{len(STL10_CLASSES)}] = "
        f"{{{classes_str}}};\n\n"
        f"static const uint8_t model_pte[] __attribute__((aligned(16))) = "
        f"{{{hexb}}};\n"
        f"static const size_t model_pte_len = {len(pte)};\n",
        encoding="utf-8",
    )


def _fmt_f(v: float) -> str:
    return f"{v:.8e}f"


def emit_golden_data_h(path: Path, source: str, entries, meta) -> None:
    res = meta["resolution"]
    n_cls = len(STL10_CLASSES)
    lines = [
        f"/* Generado por export_executorch.py — no editar. Fuente: {source}.pth",
        " * Golden images STL-10 float32 NCHW normalizadas + logits float de",
        " * referencia del runtime ExecuTorch (portable) en PC. */",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define GOLDEN_COUNT {len(entries)}",
        f"#define GOLDEN_W {res}",
        f"#define GOLDEN_H {res}",
        "#define GOLDEN_CH 3",
        "",
        "typedef struct {",
        "    const float *data;        /* NCHW, CH*H*W floats, normalizado */",
        "    const float *ref_logits;  /* logits float del runtime ET en PC */",
        "    uint8_t      expected;    /* clase verdadera */",
        "    uint8_t      ref_top1;    /* top-1 de referencia en PC */",
        "} golden_image_t;",
        "",
        f"static const char* const GOLDEN_CLASSES[{n_cls}] = "
        f"{{{', '.join(f'\"{c}\"' for c in STL10_CLASSES)}}};",
        "",
    ]
    for k, (c, _conf, top1, _qc, _m, flat, logits) in enumerate(entries):
        lines.append(
            f"static const float GOLDEN_IMG_{k}[{len(flat)}] = "
            f"{{{','.join(_fmt_f(float(v)) for v in flat)}}};")
        lines.append(
            f"static const float GOLDEN_LOGITS_{k}[{len(logits)}] = "
            f"{{{','.join(_fmt_f(float(v)) for v in logits)}}};")
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
    ap.add_argument("--skip-eval", action="store_true",
                    help="No evaluar accuracy en el test set completo")
    ap.add_argument("--eval-limit", type=int, default=0,
                    help="Evaluar sólo N imágenes del test set (0 = todas)")
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

    pte = export_pte(model, meta)
    pte_path = out_dir / "model.pte"
    pte_path.write_bytes(pte)
    print(f"[INFO] .pte portable (float): {pte_path}  ({len(pte)} bytes)")

    method = load_et_method(pte)

    print("[INFO] Seleccionando golden images (margen >= 0.15, runtime ET)...")
    entries = select_golden(model, meta, method)

    accuracy = None
    if not args.skip_eval:
        print("[INFO] Evaluando accuracy con el runtime ET...")
        accuracy = evaluate_et(method, meta,
                               args.eval_limit if args.eval_limit > 0 else None)
        print(f"[INFO] Accuracy ExecuTorch portable: {accuracy*100:.2f}%")

    emit_model_pte_h(out_dir / "model_pte.h", args.source, pte, meta)
    emit_golden_data_h(out_dir / "golden_data.h", args.source, entries, meta)

    meta_json = {
        "source": args.source,
        "runtime": "executorch",
        "input_shape": [1, 3, meta["resolution"], meta["resolution"]],
        "dtype": "float32",
        "classes": STL10_CLASSES,
        "accuracy": accuracy,
        "eval_limit": args.eval_limit or None,
        "pte_size_bytes": len(pte),
        "golden_count": len(entries),
        "tolerance": 0.01,
    }
    (out_dir / "model_meta.json").write_text(
        json.dumps(meta_json, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Outputs en {out_dir}/")


if __name__ == "__main__":
    main()
