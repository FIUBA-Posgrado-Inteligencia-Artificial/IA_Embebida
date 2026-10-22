"""
ESP32-CAM — gen_test_images: golden images para self-test en hardware
=====================================================================
Genera un header `test_images.h` con N imágenes del test set de STL-10 ya
preprocesadas al formato exacto que consume `inference_run()` en el firmware
(int8 HWC, misma cuantización que `preprocess_normalize_u8`:
`clamp((px - mean_u8) * 128 / std_u8)`), junto con dos referencias por
imagen: la predicción del modelo float y la del modelo CUANTIZADO en PC.

El camino normal es NO correr este script a mano: `script04a-quantize.py`
llama a `generate_test_images_header()` con el MISMO grafo cuantizado que
exporta al `.espdl`, así la referencia `pc_q` es bit-consistente con lo que
corre el device y el self-test (tecla `t`) debe dar N/N PASS exacto.

El CLI standalone queda para experimentos: corre su PROPIO PTQ, cuya
calibración usa otros batches → la referencia puede diferir levemente del
.espdl deployado (no esperar N/N exacto por este camino).

Uso standalone:
    uv run python scripts/gen_test_images.py --source standard_rgb96_pruned_activation_20_ft10
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "pipelib"))
from esp32cam_utils import (
    MODELS_DIR, STL10_CLASSES, get_dataloaders, get_device, load_checkpoint,
)


def device_int8_from_float01(img01: torch.Tensor, mean_u8, std_u8,
                             quant_factor: int = 128) -> np.ndarray:
    """Replica preprocess_normalize_u8 del firmware sobre una imagen CHW [0,1].

    quant_factor = 2^-MODEL_INPUT_EXPONENT (escala de entrada del .espdl).
    Devuelve el buffer HWC int8 tal cual lo vería `inference_run()`."""
    px = (img01 * 255.0).round().clamp(0, 255).to(torch.int32)  # CHW uint8
    px = px.permute(1, 2, 0).numpy()                            # HWC
    mean = np.asarray(mean_u8, dtype=np.int32)
    std = np.asarray(std_u8, dtype=np.int32)
    std = np.where(std == 0, 1, std)
    # División C (truncada hacia cero), no floor division de Python.
    x = (px - mean) * quant_factor
    x = np.trunc(x / std).astype(np.int32)
    return np.clip(x, -128, 127).astype(np.int8)


def select_golden_images(model, test_loader, meta, device,
                         candidates_per_class: int = 32) -> dict:
    """Mejores imágenes (correctas y más confiadas) por clase según el
    modelo float.

    Devuelve {class_idx: [(confidence, img01_chw), ...]} ordenadas por
    confianza descendente (hasta `candidates_per_class` por clase). Se
    guardan varias porque la elegida final debe además sobrevivir a la
    cuantización con margen (lo filtra generate_test_images_header)."""
    best = {}  # class_idx -> list[(conf, img01)]
    mean_t = torch.tensor(meta["normalize_mean"]).view(3, 1, 1)
    std_t = torch.tensor(meta["normalize_std"]).view(3, 1, 1)
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


def generate_test_images_header(qgraph, model, meta, output_path: Path,
                                device, source_name: str) -> Path | None:
    """Emite test_images.h usando el grafo cuantizado `qgraph` como referencia.

    Llamar con el MISMO grafo que se exporta al .espdl para que la referencia
    pc_q sea exacta. Devuelve el path escrito, o None si input_mode != rgb."""
    if meta["input_mode"] != "rgb":
        print("[WARN] test_images: sólo input_mode=rgb soportado — se omite.")
        return None

    from esp_ppq.executor import TorchExecutor
    qexec = TorchExecutor(graph=qgraph, device=str(device))

    # Escala de cuantización de la salida (power-of-two en espdl): permite
    # expresar los logits del sim como int8, igual que los lee el firmware.
    out_var = list(qgraph.outputs.values())[0]
    src_op = out_var.source_op
    out_idx = src_op.outputs.index(out_var)
    out_scale = float(src_op.config.output_quantization_config[out_idx].scale)

    # Escala de la ENTRADA: el buffer int8 embebido se cuantiza con el mismo
    # factor que usa el preprocess del firmware (MODEL_INPUT_EXPONENT).
    from quantize_utils import get_input_exponent
    in_exp = get_input_exponent(qgraph)
    quant_factor = 1 << -in_exp
    in_scale = 2.0 ** in_exp

    resolution = meta["resolution"]
    mean_u8 = [int(m * 255) for m in meta["normalize_mean"]]
    std_u8 = [int(s * 255) for s in meta["normalize_std"]]

    _, test_loader = get_dataloaders("rgb", resolution, batch_size=128)
    best = select_golden_images(model, test_loader, meta, device)
    if not best:
        print("[WARN] test_images: el modelo no clasificó bien ninguna imagen.")
        return None
    missing = [STL10_CLASSES[c] for c in range(len(STL10_CLASSES)) if c not in best]
    if missing:
        print(f"[WARN] test_images: sin imagen correcta para "
              f"{', '.join(missing)} — se omiten.")

    # Referencia cuantizada con la entrada DEScuantizada de nuestro buffer
    # int8 (int8/128): incluye también el truncado del preprocess del firmware.
    # Se elige por clase la primera candidata que el modelo CUANTIZADO también
    # clasifica bien y con margen top1-top2 >= MARGIN: el simulador de esp-ppq
    # y los kernels de ESP-DL difieren en redondeos (~pocos %), así que una
    # imagen al borde de la decisión flipearía en el device y el self-test
    # daría un FAIL espurio.
    MARGIN = 0.15
    entries = []
    with torch.no_grad():
        for c in sorted(best):
            chosen = None
            fallback = None
            for conf, img01 in best[c]:
                buf = device_int8_from_float01(img01, mean_u8, std_u8,
                                               quant_factor)  # HWC
                deq = torch.from_numpy(buf.astype(np.float32) * in_scale) \
                           .permute(2, 0, 1).unsqueeze(0).to(device)
                qout = qexec.forward(deq)[0][0].cpu()
                # Logits int8 del sim (misma grilla que out_data del device).
                qlogits = np.clip(np.round(qout.numpy() / out_scale),
                                  -128, 127).astype(np.int8)
                qprobs = torch.softmax(qout, dim=0)
                top2 = torch.topk(qprobs, 2)
                q_top1 = int(top2.indices[0])
                q_conf = float(top2.values[0])
                margin = q_conf - float(top2.values[1])
                cand = (c, conf, q_top1, q_conf, buf.flatten(), qlogits)
                if fallback is None:
                    fallback = (cand, margin)
                if q_top1 == c and margin >= MARGIN:
                    chosen = (cand, margin)
                    break
            cand, margin = chosen if chosen else fallback
            entries.append(cand)
            _, conf, q_top1, q_conf, _, _ = cand
            note = "" if chosen else "  [SIN MARGEN — puede flipear en device]"
            mark = "" if q_top1 == c else \
                f"  (cuantizado predice {STL10_CLASSES[q_top1]})"
            print(f"  {STL10_CLASSES[c]:<9} conf_float={conf:.3f}  "
                  f"conf_quant={q_conf:.3f}  margen={margin:.2f}{mark}{note}")

    lines = [
        "// Generado por gen_test_images.py — NO editar a mano.",
        f"// Fuente: {source_name}  (imágenes del test set de STL-10,",
        "// preprocesadas al formato int8 HWC que consume inference_run()).",
        "// pc_q_* proviene del MISMO grafo cuantizado exportado al .espdl.",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define TEST_IMAGES_COUNT {len(entries)}",
        f"#define TEST_IMAGES_W {resolution}",
        f"#define TEST_IMAGES_H {resolution}",
        "#define TEST_IMAGES_CH 3",
        "",
        "typedef struct {",
        "    const int8_t *data;         /* HWC, W*H*CH bytes */",
        "    const int8_t *pc_q_logits;  /* logits int8 del sim (misma grilla que el device) */",
        "    uint8_t       expected;     /* clase verdadera (STL-10) */",
        "    uint8_t       pc_q_top1;    /* top-1 del modelo CUANTIZADO en PC */",
        "    float         pc_conf;      /* prob top-1 del modelo float en PC */",
        "    float         pc_q_conf;    /* prob top-1 del modelo cuantizado en PC */",
        "} test_image_t;",
        "",
    ]
    for k, (c, conf, q_top1, q_conf, flat, qlogits) in enumerate(entries):
        vals = ",".join(str(int(v)) for v in flat)
        lines.append(f"static const int8_t TEST_IMG_{k}[{len(flat)}] = {{{vals}}};")
        lvals = ",".join(str(int(v)) for v in qlogits)
        lines.append(f"static const int8_t TEST_LOGITS_{k}[{len(qlogits)}] = {{{lvals}}};")
    lines.append("")
    lines.append("static const test_image_t TEST_IMAGES[TEST_IMAGES_COUNT] = {")
    for k, (c, conf, q_top1, q_conf, _, _) in enumerate(entries):
        lines.append(f"    {{ TEST_IMG_{k}, TEST_LOGITS_{k}, {c}, {q_top1}, "
                     f"{conf:.4f}f, {q_conf:.4f}f }},  /* {STL10_CLASSES[c]} */")
    lines.append("};")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    size_kb = output_path.stat().st_size / 1024
    print(f"[INFO] Generado: {output_path}  "
          f"({len(entries)} imágenes, {size_kb:.0f} KB de header)")
    return output_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Golden images para self-test (camino standalone, "
                    "referencia con PTQ propio — ver docstring).")
    p.add_argument("--source", required=True,
                   help="[required] Checkpoint float a usar (sin .pth)")
    p.add_argument("--output", type=Path,
                   default=Path("firmware/main/test_images.h"),
                   help="Header de salida (default: firmware/main/test_images.h)")
    p.add_argument("--target", default="c", choices=["c", "esp32s3"],
                   help="Target esp-ppq para la referencia cuantizada (default: c)")
    p.add_argument("--calib-batches", type=int, default=32,
                   help="Batches de calibración para el PTQ de referencia (default: 32)")
    return p.parse_args()


def main() -> None:
    from quantize_utils import get_calibration_data, run_esppq_ptq

    args = parse_args()
    device = get_device()

    source_path = MODELS_DIR / f"{args.source}.pth"
    if not source_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {source_path}")

    model, meta = load_checkpoint(source_path, device)

    print("[INFO] Generando referencia cuantizada (PTQ propio — puede diferir "
          "levemente del .espdl deployado; para referencia exacta usar "
          "script04a-quantize.py, que emite test_images junto al modelo).")
    train_loader, _ = get_dataloaders(meta["input_mode"], meta["resolution"],
                                      batch_size=64)
    calib = get_calibration_data(train_loader, args.calib_batches, device)
    qgraph, ptq_ok = run_esppq_ptq(model, calib, meta, target=args.target)
    if not ptq_ok:
        raise SystemExit("PTQ no disponible — no se puede generar la "
                         "referencia cuantizada.")

    out = generate_test_images_header(qgraph, model, meta, args.output,
                                      device, f"{args.source}.pth")
    if out:
        print("[INFO] Rebuildear el firmware y usar la tecla 't' para el self-test.")


if __name__ == "__main__":
    main()
