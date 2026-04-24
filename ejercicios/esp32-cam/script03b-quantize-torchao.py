"""
ESP32-CAM — script03b-quantize-torchao: QAT real con torchao + PTQ final con esp-ppq
====================================================================================
Diferencia clave frente a script03a:
  - 03a usa solo esp-ppq (PTQ sin fake-quant durante training).
  - 03b inserta **fake-quant nodes reales** durante training usando la API
    **PT2E** de torchao (`torchao.quantization.pt2e`), entrena el modelo
    contra esos nodos (QAT), luego "strip"-ea los fake-quant y pasa el .pth
    resultante por el mismo flujo PTQ de esp-ppq que usa 03a.

Por qué PT2E y no eager-mode: el path eager (`torch.ao.quantization.prepare_qat`)
quedó deprecado en torch 2.11 ("will be removed in a future release"). La API
PT2E de torchao es el reemplazo oficial y, en esta versión, preserva las
claves del state_dict del modelo original (sin fusionar Conv+BN antes del
strip), lo que permite un strip trivial: cargar los pesos entrenados en una
instancia fresca de StandardCNN/DepthwiseCNN.

Por qué `X86InductorQuantizer` y no XNNPACK (el canónico de los tutoriales
de PyTorch): XNNPACK no viene más como Quantizer en torchao 0.17. Para
nuestro flujo preconditioned el esquema exacto del QAT no importa — esp-ppq
re-calibra todo al esquema de ESP-DL (int8 simétrico per-tensor + PoT scales).
Solo necesitamos que los pesos se entrenen con fake-quant activo; la
configuración int8/per_channel del x86 quantizer es una muy buena
pre-condicionamiento.

Las escalas aprendidas en QAT se descartan (las re-calibra esp-ppq), pero
la robustez al ruido de cuantización sí se transfiere. Este "Puente A"
(preconditioned) no es QAT end-to-end; para eso ver script03c con --bridge qdq.

Uso:
    uv run python script03b-quantize-torchao.py --source depthwise_gray32_pruned_activation_30
    uv run python script03b-quantize-torchao.py --source standard_rgb32 --qat-epochs 10

Flags:
    --source         [requerido] nombre del checkpoint FP32 (sin .pth)
    --qat-epochs     epochs de QAT con fake-quant                   (default: 5)
    --qat-lr         learning rate del QAT                          (default: 1e-4)
    --calib-batches  batches de calibración para el PTQ post-strip  (default: 32)
    --firmware-dir   destino del model_data.h                       (default: firmware/main/)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.export import Dim
from torchao.quantization.pt2e import allow_exported_model_train_eval
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)

from esp32cam_utils import (
    MODELS_DIR, STL10_CLASSES,
    build_model, collect_available_results, evaluate, get_dataloaders, get_device,
    load_checkpoint, print_comparison_table, save_checkpoint, train,
    write_model_data_meta,
)
from quantize_utils import (
    export_espdl,
    generate_model_data_h,
    get_calibration_data,
    infer_input_shape,
    run_esppq_ptq,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QAT con torchao (PT2E) + PTQ final con esp-ppq.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python script03b-quantize-torchao.py --source depthwise_gray32_pruned_activation_30\n"
            "  uv run python script03b-quantize-torchao.py --source standard_rgb32 --qat-epochs 10\n"
        ),
    )
    p.add_argument("--source",        required=True,
                   help="[required] Checkpoint name (without .pth extension)")
    p.add_argument("--qat-epochs",    type=int, default=5,
                   help="QAT fine-tuning epochs (default: 5)")
    p.add_argument("--qat-lr",        type=float, default=1e-4,
                   help="QAT learning rate (default: 1e-4)")
    p.add_argument("--calib-batches", type=int, default=32,
                   help="Calibration batches for post-strip PTQ (default: 32)")
    p.add_argument("--firmware-dir",  type=Path, default=MODELS_DIR,
                   help="Destination directory for model_data.h (default: models/)")
    p.add_argument("--show-table",    action="store_true",
                   help="Imprime la tabla comparativa de modelos al finalizar")
    return p.parse_args()


def prepare_for_qat(
    model: torch.nn.Module, meta: dict, device: torch.device
) -> torch.nn.Module:
    """
    Aplica el flujo PT2E de torchao: exporta el modelo con `torch.export`
    (con la dimensión 0 dinámica para soportar cualquier batch size),
    inserta fake-quant nodes con `prepare_qat_pt2e`, y habilita que los
    helpers existentes `train()`/`evaluate()` puedan seguir llamando
    `.train()`/`.eval()` sobre el GraphModule resultante.
    """
    model = model.to(device)
    model.train()
    example = torch.zeros(4, *infer_input_shape(meta), device=device)
    batch_dim = Dim("batch", min=1, max=4096)
    exported = torch.export.export(
        model, (example,),
        dynamic_shapes=({0: batch_dim},),
        strict=False,
    ).module()
    quantizer = X86InductorQuantizer().set_global(
        get_default_x86_inductor_quantization_config(is_qat=True)
    )
    prepared = prepare_qat_pt2e(exported, quantizer)
    allow_exported_model_train_eval(prepared)
    return prepared


def strip_fake_quant(
    prepared: torch.nn.Module, meta: dict, device: torch.device
) -> torch.nn.Module:
    """
    Vuelca los pesos entrenados del modelo PT2E a una instancia limpia de
    StandardCNN/DepthwiseCNN. La API PT2E preserva las claves del state_dict
    original (no fusiona Conv+BN antes del strip), así que una carga directa
    es suficiente.
    """
    channels = tuple(meta.get("channels", (32, 64, 128)))
    fresh = build_model(
        arch=meta["arch"], input_mode=meta["input_mode"], channels=channels,
    ).to(device)
    result = fresh.load_state_dict(prepared.state_dict(), strict=False)
    if result.missing_keys:
        raise RuntimeError(
            f"Strip incompleto — faltan claves en el prepared_model: {result.missing_keys}"
        )
    return fresh


def main() -> None:
    args = parse_args()
    device = get_device()

    source_path = MODELS_DIR / f"{args.source}.pth"
    if not source_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {source_path}")

    output_name   = f"{args.source}_quantized_qat_torchao"
    stripped_name = f"{args.source}_qat_torchao_stripped"
    espdl_path    = MODELS_DIR / f"{output_name}.espdl"
    stripped_path = MODELS_DIR / f"{stripped_name}.pth"

    model, meta = load_checkpoint(source_path, device)
    train_loader, test_loader = get_dataloaders(
        meta["input_mode"], meta["resolution"], batch_size=64
    )

    print(f"\n{'='*60}")
    print(f"QAT (torchao PT2E) → PTQ (esp-ppq): {args.source}")
    print("=" * 60)

    baseline_acc = evaluate(model, test_loader, device)
    print(f"  Accuracy baseline FP32:                    {baseline_acc:.4f}")

    prepared = prepare_for_qat(model, meta, device)
    print(f"  QAT con fake-quant: {args.qat_epochs} epochs @ lr={args.qat_lr}")
    train(prepared, train_loader, test_loader,
          epochs=args.qat_epochs, lr=args.qat_lr, device=device)

    qat_acc = evaluate(prepared, test_loader, device)
    print(f"  Accuracy post-QAT (con fake-quant activo): {qat_acc:.4f}")

    stripped = strip_fake_quant(prepared, meta, device)
    stripped_acc = evaluate(stripped, test_loader, device)
    print(f"  Accuracy post-strip (FP32 puro):           {stripped_acc:.4f}")

    save_checkpoint(stripped_path, stripped, {**meta, "accuracy": stripped_acc})

    calib_data = get_calibration_data(train_loader, args.calib_batches, device)
    quantized_model, _ = run_esppq_ptq(stripped, calib_data, meta)
    export_espdl(quantized_model, espdl_path, meta)

    generate_model_data_h(
        espdl_path, args.firmware_dir, meta,
        meta["normalize_mean"], meta["normalize_std"],
        classes=STL10_CLASSES,
        source_script="script03b-quantize-torchao.py",
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
