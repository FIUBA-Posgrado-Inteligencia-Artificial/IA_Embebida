"""
ESP32-CAM — script01-train: Train CNN variants on STL-10
=========================================================
Trains a CNN (standard or depthwise separable) on STL-10 for the given
architecture, color mode, and resolution. If the checkpoint already exists,
it is loaded and training is skipped (use --force to override).

Usage:
    cd examples/esp32-cam
    uv run python script01-train.py --arch standard  --input-mode rgb  --resolution 96
    uv run python script01-train.py --arch standard  --input-mode rgb  --resolution 32
    uv run python script01-train.py --arch standard  --input-mode gray --resolution 32
    uv run python script01-train.py --arch depthwise --input-mode rgb  --resolution 32
    uv run python script01-train.py --arch depthwise --input-mode gray --resolution 32

Flags:
    --arch         [required] standard | depthwise
    --input-mode   [required] rgb | gray
    --resolution   [required] positive integer in px (STL-10 native: 96)
                              a WARNING is shown if the value exceeds 96
    --epochs       number of training epochs        (default: 30)
    --lr           learning rate                    (default: 1e-3)
    --batch-size   mini-batch size                  (default: 64)
    --force        re-train even if checkpoint exists
"""

import argparse

from esp32cam_utils import (
    MODELS_DIR, STL10_CLASSES,
    _NORM_STATS,
    build_model, collect_available_results, count_macs,
    count_parameters, evaluate, get_dataloaders, get_device,
    load_checkpoint, model_size_kb, print_comparison_table,
    save_checkpoint, train,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a CNN (standard or depthwise) on STL-10.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python script01-train.py --arch standard  --input-mode rgb  --resolution 96\n"
            "  uv run python script01-train.py --arch standard  --input-mode rgb  --resolution 32\n"
            "  uv run python script01-train.py --arch standard  --input-mode gray --resolution 32\n"
            "  uv run python script01-train.py --arch depthwise --input-mode rgb  --resolution 32\n"
            "  uv run python script01-train.py --arch depthwise --input-mode gray --resolution 32\n"
        ),
    )
    p.add_argument("--arch",        required=True, choices=["standard", "depthwise"],
                   help="[required] CNN architecture: standard (Conv2d) or depthwise separable")
    p.add_argument("--input-mode",  required=True, choices=["rgb", "gray"],
                   help="[required] Color mode: rgb (3 channels) or gray (1 channel)")
    p.add_argument("--resolution",  required=True, type=int,
                   help="[required] Square input size in px (STL-10 native: 96); a WARNING is shown if larger")
    p.add_argument("--epochs",      type=int,   default=30,
                   help="Number of training epochs (default: 30)")
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
    p.add_argument("--batch-size",  type=int,   default=64,
                   help="Mini-batch size (default: 64)")
    p.add_argument("--force",       action="store_true",
                   help="Re-train even if a checkpoint already exists")
    p.add_argument("--show-table",  action="store_true",
                   help="Imprime la tabla comparativa de modelos al finalizar")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    arch       = args.arch
    input_mode = args.input_mode
    resolution = args.resolution

    _STL10_NATIVE = 96
    if resolution > _STL10_NATIVE:
        print(f"[WARNING] La resolución solicitada ({resolution}px) supera la resolución "
              f"nativa del dataset ({_STL10_NATIVE}px). La imagen se escalará hacia arriba, "
              f"lo que no agrega información real.")

    checkpoint_name = f"{arch}_{input_mode}{resolution}"
    checkpoint_path = MODELS_DIR / f"{checkpoint_name}.pth"
    device = get_device()

    if checkpoint_path.exists() and not args.force:
        print(f"\n[INFO] Checkpoint encontrado — cargando {checkpoint_path.name}")
        model, meta = load_checkpoint(checkpoint_path, device)
        print(f"[INFO] Accuracy guardada: {meta['accuracy']:.4f}")
    else:
        print(f"\n{'='*60}")
        print(f"Entrenando: {arch} | {input_mode} | {resolution}×{resolution}")
        print("=" * 60)

        model = build_model(arch, input_mode)
        train_loader, test_loader = get_dataloaders(input_mode, resolution, args.batch_size)
        history = train(model, train_loader, test_loader, args.epochs, args.lr, device)
        acc = evaluate(model, test_loader, device)

        in_channels = 1 if input_mode == "gray" else 3
        macs   = count_macs(model, (1, in_channels, resolution, resolution))
        params = count_parameters(model)
        size   = model_size_kb(model)

        print(f"\nResultados — acc={acc:.4f}  params={params:,}  MACs={macs:,}  {size:.1f} KB")

        save_checkpoint(checkpoint_path, model, {
            "arch":           arch,
            "input_mode":     input_mode,
            "resolution":     resolution,
            "num_classes":    10,
            "classes":        STL10_CLASSES,
            "channels":       (32, 64, 128),
            "accuracy":       round(acc, 4),
            "history":        history,
            "normalize_mean": _NORM_STATS[input_mode]["mean"],
            "normalize_std":  _NORM_STATS[input_mode]["std"],
            "source":         None,
        })

    if args.show_table:
        print()
        print_comparison_table(collect_available_results())


if __name__ == "__main__":
    main()
