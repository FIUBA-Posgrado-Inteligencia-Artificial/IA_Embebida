"""
ESP32-CAM — script02-prune: Structured pruning
===============================================
Applies structured pruning to any checkpoint produced by script01-train.
Fine-tuning is a separate stage (script03-finetune.py).

Usage:
    uv run python scripts/script02-prune.py --source depthwise_gray32 --method activation --ratio 0.3
    uv run python scripts/script02-prune.py --source depthwise_gray32 --method weight     --ratio 0.5
    uv run python scripts/script02-prune.py --source standard_rgb32   --method weight     --ratio 0.3
    uv run python scripts/script02-prune.py --source standard_rgb96   --method activation --ratio 0.3
    uv run python scripts/script02-prune.py --source depthwise_gray32 --method none

Flags:
    --source      [required] checkpoint name to prune (without .pth)
    --method      pruning strategy: none | weight | activation  (default: activation)
    --ratio       fraction of channels to remove, e.g. 0.3 = 30%  (default: 0.3)
    --batch-size  mini-batch size                                 (default: 64)
"""

import argparse
import copy
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "pipelib"))
from esp32cam_utils import (
    MODELS_DIR, STL10_CLASSES,
    collect_available_results, count_macs, count_parameters,
    evaluate, get_dataloaders, get_device, load_checkpoint,
    model_size_kb, print_comparison_table, prune_depthwise_cnn,
    prune_standard_cnn, save_checkpoint,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Structured pruning on any checkpoint produced by script01-train.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python scripts/script02-prune.py --source depthwise_gray32 --method activation --ratio 0.3\n"
            "  uv run python scripts/script02-prune.py --source depthwise_gray32 --method weight     --ratio 0.5\n"
            "  uv run python scripts/script02-prune.py --source standard_rgb32   --method weight     --ratio 0.3\n"
            "  uv run python scripts/script02-prune.py --source standard_rgb96   --method activation --ratio 0.3\n"
            "  uv run python scripts/script02-prune.py --source depthwise_gray32 --method none\n"
        ),
    )
    p.add_argument("--source",          required=True,
                   help="[required] Checkpoint name to prune (without .pth extension)")
    p.add_argument("--method",          default="activation",
                   choices=["none", "weight", "activation"],
                   help="Pruning strategy: none | weight (L1) | activation (default: activation)")
    p.add_argument("--ratio",           type=float, default=0.3,
                   help="Fraction of channels to remove, e.g. 0.3 = 30%% (default: 0.3)")
    p.add_argument("--batch-size",      type=int,   default=64,
                   help="Mini-batch size (default: 64)")
    p.add_argument("--output",          default=None,
                   help="Nombre del checkpoint de salida (sin .pth). Default: "
                        "<source>_pruned_<method>_<ratio%%>. Útil en cadenas de "
                        "pruning iterativo, donde el nombre acumulado supera el "
                        "límite de 255 bytes del filesystem.")
    p.add_argument("--show-table",      action="store_true",
                   help="Imprime la tabla comparativa de modelos al finalizar")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    source_path = MODELS_DIR / f"{args.source}.pth"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Checkpoint no encontrado: {source_path}\n"
            f"Ejecutar primero el script de entrenamiento correspondiente."
        )

    ratio_pct = int(args.ratio * 100)
    # Sufijo corto: _pr + inicial del método + ratio% (ej: _pra30 = pruned
    # activation 30%). Mantener en sync con infer_stage_from_name.
    output_name = args.output or f"{args.source}_pr{args.method[0]}{ratio_pct}"
    output_path = MODELS_DIR / f"{output_name}.pth"

    if output_path.exists():
        print(f"[INFO] Checkpoint de pruning encontrado — cargando {output_path.name}")
        model, meta = load_checkpoint(output_path, device)
    else:
        print(f"\n{'='*60}")
        print(f"Pruning: {args.source}  método={args.method}  ratio={args.ratio}")
        print("=" * 60)

        base_model, base_meta = load_checkpoint(source_path, device)
        arch       = base_meta["arch"]
        input_mode = base_meta["input_mode"]
        resolution = base_meta["resolution"]

        train_loader, test_loader = get_dataloaders(input_mode, resolution, args.batch_size)

        if args.method == "none":
            model = copy.deepcopy(base_model)
            print("[INFO] method=none: sin pruning, copiando modelo base")
        elif arch == "standard":
            model = prune_standard_cnn(
                base_model, args.ratio, args.method,
                train_loader if args.method == "activation" else None,
                device if args.method == "activation" else None,
            )
        elif arch == "depthwise":
            model = prune_depthwise_cnn(
                base_model, args.ratio, args.method,
                train_loader if args.method == "activation" else None,
                device if args.method == "activation" else None,
            )
        else:
            raise ValueError(f"Arquitectura desconocida en checkpoint: '{arch}'")

        model = model.to(device)
        in_channels = 1 if input_mode == "gray" else 3

        acc    = evaluate(model, test_loader, device)
        macs   = count_macs(model, (1, in_channels, resolution, resolution))
        params = count_parameters(model)
        size   = model_size_kb(model)
        print(f"  Acc post-pruning — acc={acc:.4f}  MACs={macs:,}  {size:.1f} KB")

        channels = (
            model.block1.conv.out_channels if arch == "standard" else model.block1.pw.out_channels,
            model.block2.conv.out_channels if arch == "standard" else model.block2.pw.out_channels,
            model.block3.conv.out_channels if arch == "standard" else model.block3.pw.out_channels,
        )

        save_checkpoint(output_path, model, {
            "arch":           arch,
            "input_mode":     input_mode,
            "resolution":     resolution,
            "num_classes":    10,
            "classes":        STL10_CLASSES,
            "channels":       channels,
            "accuracy":       round(acc, 4),
            "history":        [],
            "normalize_mean": base_meta["normalize_mean"],
            "normalize_std":  base_meta["normalize_std"],
            "source":         args.source,
            "pruning_method": args.method,
            "pruning_ratio":  args.ratio,
        })

    if args.show_table:
        print()
        print_comparison_table(collect_available_results())


if __name__ == "__main__":
    main()
