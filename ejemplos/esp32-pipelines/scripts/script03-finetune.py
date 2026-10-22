"""
ESP32-CAM — script03-finetune: Fine-tuning de un checkpoint
============================================================
Reentrena N epochs cualquier checkpoint .pth (tipicamente post-pruning).
Repetible: train -> prune -> finetune -> prune -> finetune.

Usage:
    uv run python scripts/script03-finetune.py --source depthwise_rgb32_pruned_activation_30 --epochs 10
"""

import argparse

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "pipelib"))
from esp32cam_utils import (
    MODELS_DIR, collect_available_results, count_macs, count_parameters,
    evaluate, get_dataloaders, get_device, load_checkpoint,
    model_size_kb, print_comparison_table, save_checkpoint, train,
)


def output_name_for(source: str, epochs: int) -> str:
    return f"{source}_ft{epochs}"


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tuning de cualquier checkpoint producido por train/prune.")
    p.add_argument("--source", required=True,
                   help="[required] Checkpoint a finetunear (sin .pth)")
    p.add_argument("--epochs", type=int, default=10, help="Epochs (default: 10)")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    p.add_argument("--batch-size", type=int, default=64, help="Mini-batch (default: 64)")
    p.add_argument("--show-table", action="store_true",
                   help="Imprime la tabla comparativa de modelos al finalizar")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    source_path = MODELS_DIR / f"{args.source}.pth"
    if not source_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {source_path}")

    output_name = output_name_for(args.source, args.epochs)
    output_path = MODELS_DIR / f"{output_name}.pth"

    if output_path.exists():
        print(f"[INFO] Checkpoint de finetune encontrado — cargando {output_path.name}")
        return

    model, meta = load_checkpoint(source_path, device)
    train_loader, test_loader = get_dataloaders(
        meta["input_mode"], meta["resolution"], args.batch_size)

    print(f"Fine-tuning {args.source}: {args.epochs} epochs, lr={args.lr}")
    train(model, train_loader, test_loader, args.epochs, args.lr, device)

    in_channels = 1 if meta["input_mode"] == "gray" else 3
    acc = evaluate(model, test_loader, device)
    print(f"  acc={acc:.4f}  MACs={count_macs(model, (1, in_channels, meta['resolution'], meta['resolution'])):,}  "
          f"{model_size_kb(model):.1f} KB")

    new_meta = dict(meta)
    new_meta.update({
        "accuracy": round(acc, 4),
        "source": args.source,
        "finetune_epochs": args.epochs,
        "finetune_lr": args.lr,
    })
    save_checkpoint(output_path, model, new_meta)

    if args.show_table:
        print()
        print_comparison_table(collect_available_results())


if __name__ == "__main__":
    main()
