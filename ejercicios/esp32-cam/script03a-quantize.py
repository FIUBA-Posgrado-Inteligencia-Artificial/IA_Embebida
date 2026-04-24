"""
ESP32-CAM — script03a-quantize: Quantization with ESP-PPQ (PTQ/QAT-finetune)
===========================================================================
Cuantización directa vía esp-ppq (PTQ) y, opcionalmente, fine-tune sobre el
modelo ya cuantizado (etiquetado aquí como "qat", aunque **no** es QAT con
fake-quant nodes en el grafo — para eso ver script03b (torchao) y script03c
(brevitas)).

Exporta:
  - Un archivo .espdl listo para ESP-DL.
  - firmware/main/model_data.h con el modelo como array C uint8_t.

Uso:
    uv run python script03a-quantize.py --source depthwise_gray32_pruned_activation_30 --mode ptq
    uv run python script03a-quantize.py --source depthwise_gray32_pruned_activation_30 --mode qat
    uv run python script03a-quantize.py --source depthwise_gray32_pruned_weight_50     --mode ptq
    uv run python script03a-quantize.py --source depthwise_gray32                      --mode ptq
    uv run python script03a-quantize.py --source standard_rgb32                        --mode ptq

Flags:
    --source         [requerido] nombre del checkpoint (sin .pth)
    --mode           modo de cuantización: ptq | qat              (default: ptq)
    --calib-batches  batches de calibración para PTQ              (default: 32)
    --qat-epochs     epochs de fine-tune sobre el modelo cuantizado (default: 5)
    --firmware-dir   destino del model_data.h                     (default: firmware/main/)
"""

import argparse
import warnings
from pathlib import Path

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
            "  uv run python script03a-quantize.py --source depthwise_gray32_pruned_activation_30 --mode ptq\n"
            "  uv run python script03a-quantize.py --source depthwise_gray32_pruned_weight_50     --mode ptq\n"
            "  uv run python script03a-quantize.py --source depthwise_gray32                      --mode ptq\n"
            "  uv run python script03a-quantize.py --source depthwise_gray32_pruned_activation_30 --mode qat\n"
            "  uv run python script03a-quantize.py --source standard_rgb32                        --mode ptq\n"
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    source_path = MODELS_DIR / f"{args.source}.pth"
    if not source_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {source_path}")

    output_name = f"{args.source}_quantized_{args.mode}"
    espdl_path  = MODELS_DIR / f"{output_name}.espdl"

    model, meta = load_checkpoint(source_path, device)
    train_loader, test_loader = get_dataloaders(
        meta["input_mode"], meta["resolution"], batch_size=64
    )

    print(f"\n{'='*60}")
    print(f"Cuantización {args.mode.upper()}: {args.source}")
    print("=" * 60)

    calib_data = get_calibration_data(train_loader, args.calib_batches, device)
    quantized_model, ptq_ok = run_esppq_ptq(model, calib_data, meta)

    if args.mode == "qat" and ptq_ok:
        print(f"  QAT fine-tuning {args.qat_epochs} epochs sobre modelo cuantizado...")
        try:
            train(quantized_model, train_loader, test_loader,
                  args.qat_epochs, lr=1e-4, device=device)
        except Exception as e:
            warnings.warn(f"[ADVERTENCIA] QAT fine-tuning falló: {e}. Usando resultado PTQ.")
    elif args.mode == "qat" and not ptq_ok:
        warnings.warn("[ADVERTENCIA] QAT omitido porque PTQ no estuvo disponible.")

    export_espdl(quantized_model, espdl_path, meta)

    generate_model_data_h(
        espdl_path, args.firmware_dir, meta,
        meta["normalize_mean"], meta["normalize_std"],
        classes=STL10_CLASSES,
        source_script="script03a-quantize.py",
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
        },
    )

    print(f"\n[SIGUIENTE PASO]")
    print(f"  Header generado: {args.firmware_dir / f'{output_name}_model_data.h'}")
    print(f"  Para buildear y flashear:")
    print(f"    uv run python script04-build.py --source {output_name} [--port /dev/ttyUSBN] [--monitor]\n")

    if args.show_table:
        print_comparison_table(collect_available_results())


if __name__ == "__main__":
    main()
