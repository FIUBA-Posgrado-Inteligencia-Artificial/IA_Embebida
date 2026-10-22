"""
ESP32-CAM — script04a-quantize: Quantization with ESP-PPQ (PTQ/QAT-finetune)
===========================================================================
Cuantización directa vía esp-ppq (PTQ) y, opcionalmente, fine-tune sobre el
modelo ya cuantizado (etiquetado aquí como "qat", aunque **no** es QAT con
fake-quant nodes en el grafo — para eso ver script04b (torchao) y script04c
(brevitas)).

Exporta:
  - Un archivo .espdl listo para ESP-DL.
  - firmware/main/model_data.h con el modelo como array C uint8_t.

Uso:
    uv run python scripts/script04a-quantize.py --source depthwise_gray32_pruned_activation_30 --mode ptq
    uv run python scripts/script04a-quantize.py --source depthwise_gray32_pruned_activation_30 --mode qat
    uv run python scripts/script04a-quantize.py --source depthwise_gray32_pruned_weight_50     --mode ptq
    uv run python scripts/script04a-quantize.py --source depthwise_gray32                      --mode ptq
    uv run python scripts/script04a-quantize.py --source standard_rgb32                        --mode ptq

Flags:
    --source         [requerido] nombre del checkpoint (sin .pth)
    --mode           modo de cuantización: ptq | qat              (default: ptq)
    --calib-batches  batches de calibración para PTQ              (default: 32)
    --qat-epochs     epochs de fine-tune sobre el modelo cuantizado (default: 5)
    --firmware-dir   destino del model_data.h                     (default: MODELS_DIR)
    --target         c | esp32s3 — kernels del .espdl          (default: c)
    --force          re-cuantiza aunque el cache esté al día

Cache: si .espdl + model_data.h + sidecar existen, son más nuevos que el
.pth fuente y el sidecar declara el mismo --target, el stage se saltea.
"""

import argparse
import warnings
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "pipelib"))
from esp32cam_utils import (
    MODELS_DIR, STL10_CLASSES,
    collect_available_results, get_dataloaders, get_device,
    load_checkpoint, print_comparison_table, train,
    write_model_data_meta,
)
from quantize_utils import (
    export_espdl,
    generate_model_data_h,
    get_calibration_data,
    run_esppq_ptq,
)

def parse_args():
    p = argparse.ArgumentParser(
        description="PTQ/QAT-finetune con ESP-PPQ y generación de model_data.h.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python scripts/script04a-quantize.py --source depthwise_gray32_pruned_activation_30 --mode ptq\n"
            "  uv run python scripts/script04a-quantize.py --source depthwise_gray32_pruned_weight_50     --mode ptq\n"
            "  uv run python scripts/script04a-quantize.py --source depthwise_gray32                      --mode ptq\n"
            "  uv run python scripts/script04a-quantize.py --source depthwise_gray32_pruned_activation_30 --mode qat\n"
            "  uv run python scripts/script04a-quantize.py --source standard_rgb32                        --mode ptq\n"
        ),
    )
    p.add_argument("--source",        required=True,
                   help="[required] Checkpoint name to quantize (without .pth extension)")
    p.add_argument("--mode",          default="ptq", choices=["ptq", "qat"],
                   help="Quantization mode: ptq (post-training) | qat (finetune over PTQ model) (default: ptq)")
    p.add_argument("--calib-batches", type=int, default=32,
                   help="Number of calibration batches for PTQ (default: 32)")
    p.add_argument("--qat-epochs",    type=int, default=5,
                   help="Fine-tuning epochs over the quantized model (default: 5)")
    p.add_argument("--firmware-dir",  type=Path, default=MODELS_DIR,
                   help="Destination directory for model_data.h (default: models/)")
    p.add_argument("--show-table",    action="store_true",
                   help="Imprime la tabla comparativa de modelos al finalizar")
    p.add_argument("--force",         action="store_true",
                   help="Re-cuantiza aunque el .espdl cacheado esté al día")
    p.add_argument("--test-images",   action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Emite <source>_test_images.h con golden images y la "
                        "referencia del MISMO grafo cuantizado (default: on)")
    p.add_argument("--target",        default="c", choices=["c", "esp32s3"],
                   help="Target de esp-ppq: c (genérico, ESP32 clásico) | "
                        "esp32s3 (kernels SIMD del S3). El .espdl no es "
                        "intercambiable entre targets. (default: c)")
    return p.parse_args()


def cache_is_fresh(source_path: Path, espdl_path: Path,
                   header_path: Path, meta_path: Path, target: str) -> bool:
    """True si los tres artefactos existen, son más nuevos que el .pth fuente
    y el sidecar declara el mismo target (el nombre del .espdl NO codifica el
    target, así que la existencia sola no alcanza)."""
    import json
    if not (espdl_path.exists() and header_path.exists() and meta_path.exists()):
        return False
    src_mtime = source_path.stat().st_mtime
    if espdl_path.stat().st_mtime < src_mtime or header_path.stat().st_mtime < src_mtime:
        return False
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")).get("target") == target
    except (ValueError, OSError):
        return False


def print_next_steps(header_path: Path, output_name: str) -> None:
    print(f"\n[SIGUIENTE PASO]")
    print(f"  Header generado: {header_path}")
    print(f"  Para buildear y flashear:")
    print(f"    uv run python scripts/script05-build.py --source {output_name} --action build\n"
          f"    uv run python scripts/script05-build.py --action flash --port /dev/ttyUSBN\n"
          f"    uv run python scripts/script05-build.py --action monitor --port /dev/ttyUSBN\n")


def main() -> None:
    args = parse_args()

    source_path = MODELS_DIR / f"{args.source}.pth"
    if not source_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {source_path}")

    output_name = f"{args.source}_q{args.mode}"
    espdl_path  = MODELS_DIR / f"{output_name}.espdl"
    header_path = args.firmware_dir / f"{output_name}_model_data.h"
    meta_path   = args.firmware_dir / f"{output_name}_model_data.meta.json"
    test_images_path = args.firmware_dir / f"{output_name}_test_images.h"

    cache_ok = cache_is_fresh(source_path, espdl_path, header_path,
                              meta_path, args.target)
    if cache_ok and args.test_images and not test_images_path.exists():
        cache_ok = False  # faltan las golden images: regenerar todo junto
    if not args.force and cache_ok:
        print(f"[INFO] Quantize cacheado — {espdl_path.name} está al día con "
              f"{source_path.name} (target={args.target}). Usar --force para re-cuantizar.")
        print_next_steps(header_path, output_name)
        return

    device = get_device()

    model, meta = load_checkpoint(source_path, device)
    train_loader, test_loader = get_dataloaders(
        meta["input_mode"], meta["resolution"], batch_size=64
    )

    print(f"\n{'='*60}")
    print(f"Cuantización {args.mode.upper()}: {args.source}")
    print("=" * 60)

    calib_data = get_calibration_data(train_loader, args.calib_batches, device)
    quantized_model, ptq_ok = run_esppq_ptq(model, calib_data, meta, target=args.target)

    if args.mode == "qat" and ptq_ok:
        print(f"  QAT fine-tuning {args.qat_epochs} epochs sobre modelo cuantizado...")
        try:
            train(quantized_model, train_loader, test_loader,
                  args.qat_epochs, lr=1e-4, device=device)
        except Exception as e:
            warnings.warn(f"[ADVERTENCIA] QAT fine-tuning falló: {e}. Usando resultado PTQ.")
    elif args.mode == "qat" and not ptq_ok:
        warnings.warn("[ADVERTENCIA] QAT omitido porque PTQ no estuvo disponible.")

    export_espdl(quantized_model, espdl_path, meta, target=args.target)

    # Exponente de la entrada calibrado por esp-ppq: el preprocess del
    # firmware cuantiza con este factor (asumir -7 satura si calibró -6).
    from quantize_utils import get_input_exponent
    input_exponent = get_input_exponent(quantized_model) if ptq_ok else -7

    if args.test_images and ptq_ok:
        # Golden images con la referencia del MISMO grafo que se exporta:
        # el self-test del firmware (tecla 't') debe dar N/N PASS exacto.
        from gen_test_images import generate_test_images_header
        print("\n[INFO] Generando golden images para el self-test...")
        generate_test_images_header(quantized_model, model, meta,
                                    test_images_path, device,
                                    f"{args.source}.pth")

    generate_model_data_h(
        espdl_path, args.firmware_dir, meta,
        meta["normalize_mean"], meta["normalize_std"],
        classes=STL10_CLASSES,
        source_script="script04a-quantize.py",
        input_exponent=input_exponent,
    )

    # Sidecar metadata for build preflight (matches model_data.h constants).
    input_channels = 1 if meta["input_mode"] == "gray" else 3
    write_model_data_meta(
        args.firmware_dir / f"{output_name}_model_data.meta.json",
        {
            "num_classes": len(STL10_CLASSES),
            "input_w": meta["resolution"],
            "input_h": meta["resolution"],
            "input_channels": input_channels,
            "mean": [int(m * 255) for m in meta["normalize_mean"]],
            "std":  [int(s * 255) for s in meta["normalize_std"]],
            "classes": list(STL10_CLASSES),
            "target": args.target,
            "input_exponent": input_exponent,
        },
    )

    print_next_steps(header_path, output_name)

    if args.show_table:
        print_comparison_table(collect_available_results())


if __name__ == "__main__":
    main()
