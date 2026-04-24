"""
ESP32-CAM — script03c-quantize-brevitas: QAT real con brevitas, con dos puentes
===============================================================================
Diferencia clave frente a 03a y 03b:
  - 03a usa esp-ppq (PTQ, sin QAT real).
  - 03b usa torchao (QAT real) y "precondiciona" el modelo: strip → esp-ppq PTQ.
    Las escalas aprendidas en QAT se descartan (las re-calibra esp-ppq).
  - 03c usa brevitas (QAT real) y ofrece DOS puentes al .espdl:

    --bridge preconditioned  (default)
        mismo camino que 03b: QAT → strip → esp-ppq PTQ → .espdl.
        brevitas entrena con fake-quant de scale FLOAT (Int8*PerTensorFloat),
        después de strip el .pth es un FP32 hardened que pasa por esp-ppq.

    --bridge qdq
        QAT end-to-end con scales PoT → export ONNX QDQ → espdl_quantize_onnx.
        brevitas entrena con fake-quant de scale POWER_OF_TWO
        (Int8*PerTensorFixedPoint, compatible con ESP-DL por construcción),
        exporta un ONNX con nodos QDQ cargando las escalas aprendidas, y los
        pasa a esp-ppq.espdl_quantize_onnx para que genere el .espdl respetando
        esas escalas (ver al final del script la verificación experimental:
        si esp-ppq re-calibra, el Puente B degenera al Puente A vía ONNX).

Uso:
    # Puente A (preconditioned)
    uv run python script03c-quantize-brevitas.py --source depthwise_gray32_pruned_activation_30

    # Puente B (qdq) — end-to-end con PoT scales
    uv run python script03c-quantize-brevitas.py --source depthwise_gray32_pruned_activation_30 --bridge qdq

Flags:
    --source         [requerido] nombre del checkpoint FP32 (sin .pth)
    --bridge         preconditioned | qdq                          (default: preconditioned)
    --qat-epochs     epochs de QAT con fake-quant                  (default: 5)
    --qat-lr         learning rate del QAT                         (default: 1e-4)
    --calib-batches  batches de calibración para el PTQ / el QDQ   (default: 32)
    --firmware-dir   destino del model_data.h                      (default: firmware/main/)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.nn import QuantConv2d, QuantLinear
from brevitas.quant import (
    Int8ActPerTensorFixedPoint, Int8ActPerTensorFloat,
    Int8WeightPerTensorFixedPoint, Int8WeightPerTensorFloat,
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

# Mapeo scale_mode → (weight_quant, act_quant).
# "float"         → escalas float (entrenamiento QAT estándar).
# "power_of_two"  → escalas PoT (compatible con ESP-DL por construcción).
_QUANT_CONFIGS = {
    "float":        (Int8WeightPerTensorFloat,      Int8ActPerTensorFloat),
    "power_of_two": (Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint),
}


# ---------------------------------------------------------------------------
# Twin brevitas de StandardCNN / DepthwiseCNN
# ---------------------------------------------------------------------------

class QuantConvBlock(nn.Module):
    """Equivalente brevitas de esp32cam_utils.ConvBlock."""
    def __init__(self, in_channels: int, out_channels: int,
                 weight_quant, act_quant):
        super().__init__()
        self.conv = QuantConv2d(
            in_channels, out_channels, 3, padding=1,
            weight_quant=weight_quant, input_quant=act_quant,
            return_quant_tensor=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class QuantDSBlock(nn.Module):
    """Equivalente brevitas de esp32cam_utils.DSBlock (depthwise separable)."""
    def __init__(self, in_channels: int, out_channels: int,
                 weight_quant, act_quant):
        super().__init__()
        self.dw = QuantConv2d(
            in_channels, in_channels, 3, padding=1, groups=in_channels,
            weight_quant=weight_quant, input_quant=act_quant,
            return_quant_tensor=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.pw = QuantConv2d(
            in_channels, out_channels, 1,
            weight_quant=weight_quant, input_quant=act_quant,
            return_quant_tensor=False,
        )
        self.pw_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.dw_bn(self.dw(x)))
        x = F.relu(self.pw_bn(self.pw(x)))
        return x


class QuantStandardCNN(nn.Module):
    def __init__(self, in_channels, num_classes, channels, weight_quant, act_quant):
        super().__init__()
        c1, c2, c3 = channels
        self.block1 = QuantConvBlock(in_channels, c1, weight_quant, act_quant)
        self.block2 = QuantConvBlock(c1, c2, weight_quant, act_quant)
        self.block3 = QuantConvBlock(c2, c3, weight_quant, act_quant)
        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = QuantLinear(
            c3, num_classes,
            weight_quant=weight_quant, input_quant=act_quant,
            bias=True, return_quant_tensor=False,
        )

    def forward(self, x):
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)


class QuantDepthwiseCNN(nn.Module):
    def __init__(self, in_channels, num_classes, channels, weight_quant, act_quant):
        super().__init__()
        c1, c2, c3 = channels
        self.block1 = QuantDSBlock(in_channels, c1, weight_quant, act_quant)
        self.block2 = QuantDSBlock(c1, c2, weight_quant, act_quant)
        self.block3 = QuantDSBlock(c2, c3, weight_quant, act_quant)
        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = QuantLinear(
            c3, num_classes,
            weight_quant=weight_quant, input_quant=act_quant,
            bias=True, return_quant_tensor=False,
        )

    def forward(self, x):
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)


def build_brevitas_twin(meta: dict, scale_mode: str) -> nn.Module:
    """
    Construye el twin brevitas con las mismas dimensiones que el checkpoint
    original (incluyendo canales podados si viene de 02-prune).
    """
    if scale_mode not in _QUANT_CONFIGS:
        raise ValueError(f"scale_mode desconocido: {scale_mode}")
    weight_quant, act_quant = _QUANT_CONFIGS[scale_mode]

    in_channels = 1 if meta["input_mode"] == "gray" else 3
    channels = tuple(meta.get("channels", (32, 64, 128)))
    num_classes = meta.get("num_classes", 10)

    cls = {"standard": QuantStandardCNN, "depthwise": QuantDepthwiseCNN}[meta["arch"]]
    return cls(
        in_channels=in_channels, num_classes=num_classes, channels=channels,
        weight_quant=weight_quant, act_quant=act_quant,
    )


def copy_weights_into_twin(src: nn.Module, twin: nn.Module) -> None:
    """
    Copia weights + buffers del modelo FP32 al twin brevitas.

    Asume que la topología de ambos es 1:1 (ConvBlock↔QuantConvBlock,
    DSBlock↔QuantDSBlock), por lo que los nombres de submódulos matchean y
    los tensores `weight`/`bias`/estadísticas de BN son intercambiables.
    """
    src_state = src.state_dict()
    missing = twin.load_state_dict(src_state, strict=False)
    # Las claves "missing" del twin corresponden a buffers internos de los
    # quantizers de brevitas (se inicializan solos en el primer forward).
    # Las "unexpected" no deberían existir: si aparecen, la topología diverge.
    if missing.unexpected_keys:
        raise RuntimeError(
            "Topología del twin no coincide con el checkpoint: "
            f"claves inesperadas {missing.unexpected_keys}"
        )


def copy_weights_back_to_fp32(twin: nn.Module, dst: nn.Module) -> None:
    """
    Ruta inversa del strip: copia los pesos entrenados del twin brevitas a
    una instancia limpia de StandardCNN/DepthwiseCNN.
    """
    twin_state = twin.state_dict()
    dst_keys = set(dst.state_dict().keys())
    clean = {k: v for k, v in twin_state.items() if k in dst_keys}
    missing = dst_keys - set(clean.keys())
    if missing:
        raise RuntimeError(f"Strip incompleto — faltan claves: {missing}")
    dst.load_state_dict(clean)


# ---------------------------------------------------------------------------
# Puentes
# ---------------------------------------------------------------------------

def bridge_preconditioned(twin, meta, test_loader, calib_data,
                          source_name, device) -> tuple[Path, nn.Module]:
    """Puente A: strip el twin → fresh FP32 → esp-ppq PTQ → .espdl."""
    fresh = build_model(arch=meta["arch"], input_mode=meta["input_mode"],
                        channels=tuple(meta.get("channels", (32, 64, 128)))).to(device)
    copy_weights_back_to_fp32(twin, fresh)

    stripped_acc = evaluate(fresh, test_loader, device)
    print(f"  Accuracy post-strip (FP32 puro):           {stripped_acc:.4f}")

    stripped_path = MODELS_DIR / f"{source_name}_qat_brevitas_stripped.pth"
    save_checkpoint(stripped_path, fresh, {**meta, "accuracy": stripped_acc})

    quantized, _ = run_esppq_ptq(fresh, calib_data, meta)
    espdl_path = MODELS_DIR / f"{source_name}_quantized_qat_brevitas.espdl"
    export_espdl(quantized, espdl_path, meta)
    return espdl_path, quantized


def _patch_brevitas_onnx_compat() -> None:
    """
    Shim de compatibilidad: brevitas 0.12.1 lee el opset ONNX desde
    `torch.onnx.symbolic_helper._export_onnx_opset_version` (atributo privado
    removido en torch ≥ 2.8). Le seteamos el valor ahí para que el export
    no caiga al path de fallback que intenta importar el inexistente
    `torch.onnx._globals`.

    Opset 14 es el mínimo que soporta QuantizeLinear/DequantizeLinear con
    escalas per-channel, suficiente para nuestros modelos per-tensor.
    """
    import torch.onnx.symbolic_helper as sh
    if not hasattr(sh, "_export_onnx_opset_version"):
        sh._export_onnx_opset_version = 14


def bridge_qdq(twin, meta, calib_data, source_name, device) -> tuple[Path, nn.Module | None]:
    """
    Puente B: ONNX QDQ → espdl_quantize_onnx.

    Requiere que el twin haya sido entrenado con scale_mode='power_of_two'.
    """
    _patch_brevitas_onnx_compat()
    from brevitas.export import export_onnx_qcdq

    onnx_path = MODELS_DIR / f"{source_name}_qat_brevitas.onnx"
    espdl_path = MODELS_DIR / f"{source_name}_quantized_qat_brevitas_qdq.espdl"

    twin.eval()
    example = torch.randn(1, *infer_input_shape(meta), device=device)
    # dynamo=False fuerza el path legacy (TorchScript-based). El path dynamo
    # falla porque brevitas usa `assert` sobre expresiones que torch.export
    # no puede resolver simbólicamente.
    export_onnx_qcdq(twin, args=example, export_path=str(onnx_path), dynamo=False)
    print(f"[INFO] ONNX QDQ exportado: {onnx_path}")

    try:
        from esp_ppq.api import espdl_quantize_onnx

        espdl_quantize_onnx(
            onnx_import_file=str(onnx_path),
            espdl_export_file=str(espdl_path),
            calib_dataloader=calib_data,
            calib_steps=min(len(calib_data), 4),
            input_shape=[1] + infer_input_shape(meta),
            target="c",
            num_of_bits=8,
            device=str(device),
        )
        print(f"[INFO] .espdl generado vía QDQ bridge: {espdl_path}")
        return espdl_path, None
    except ImportError:
        warnings.warn(
            "[ADVERTENCIA] esp-ppq no disponible; el ONNX QDQ quedó en "
            f"{onnx_path} pero no se pudo convertir a .espdl."
        )
        return espdl_path, None
    except Exception as e:
        warnings.warn(f"[ADVERTENCIA] espdl_quantize_onnx falló: {e}")
        return espdl_path, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QAT con brevitas + exportación a .espdl (dos puentes).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python script03c-quantize-brevitas.py --source depthwise_gray32_pruned_activation_30\n"
            "  uv run python script03c-quantize-brevitas.py --source depthwise_gray32_pruned_activation_30 --bridge qdq\n"
            "  uv run python script03c-quantize-brevitas.py --source standard_rgb32 --qat-epochs 10\n"
        ),
    )
    p.add_argument("--source",        required=True,
                   help="[required] Checkpoint name (without .pth extension)")
    p.add_argument("--bridge",        default="preconditioned",
                   choices=["preconditioned", "qdq"],
                   help="Puente al .espdl: preconditioned (strip+esp-ppq) | qdq (ONNX+espdl_quantize_onnx)")
    p.add_argument("--qat-epochs",    type=int, default=5,
                   help="QAT fine-tuning epochs (default: 5)")
    p.add_argument("--qat-lr",        type=float, default=1e-4,
                   help="QAT learning rate (default: 1e-4)")
    p.add_argument("--calib-batches", type=int, default=32,
                   help="Calibration batches for PTQ / QDQ bridge (default: 32)")
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

    fp32_model, meta = load_checkpoint(source_path, device)
    train_loader, test_loader = get_dataloaders(
        meta["input_mode"], meta["resolution"], batch_size=64
    )

    print(f"\n{'='*60}")
    print(f"QAT (brevitas, bridge={args.bridge}): {args.source}")
    print("=" * 60)

    baseline_acc = evaluate(fp32_model, test_loader, device)
    print(f"  Accuracy baseline FP32:                    {baseline_acc:.4f}")

    # Puente A usa scales float; Puente B usa scales power-of-two.
    scale_mode = "float" if args.bridge == "preconditioned" else "power_of_two"
    twin = build_brevitas_twin(meta, scale_mode=scale_mode).to(device)
    copy_weights_into_twin(fp32_model, twin)

    print(f"  QAT (scale_mode={scale_mode}): {args.qat_epochs} epochs @ lr={args.qat_lr}")
    train(twin, train_loader, test_loader,
          epochs=args.qat_epochs, lr=args.qat_lr, device=device)

    twin.eval()
    qat_acc = evaluate(twin, test_loader, device)
    print(f"  Accuracy post-QAT (fake-quant activo):     {qat_acc:.4f}")

    calib_data = get_calibration_data(train_loader, args.calib_batches, device)

    if args.bridge == "preconditioned":
        espdl_path, _ = bridge_preconditioned(
            twin, meta, test_loader, calib_data, args.source, device
        )
    else:  # qdq
        espdl_path, _ = bridge_qdq(twin, meta, calib_data, args.source, device)

    generate_model_data_h(
        espdl_path, args.firmware_dir, meta,
        meta["normalize_mean"], meta["normalize_std"],
        classes=STL10_CLASSES,
        source_script="script03c-quantize-brevitas.py",
    )

    # Sidecar metadata for build preflight (matches model_data.h constants).
    input_channels = 1 if meta["input_mode"] == "gray" else 3
    write_model_data_meta(
        args.firmware_dir / f"{espdl_path.stem}_model_data.meta.json",
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
    print(f"  Header generado: {args.firmware_dir / f'{espdl_path.stem}_model_data.h'}")
    print(f"  Para buildear y flashear:")
    print(f"    uv run python script04-build.py --source {espdl_path.stem} [--port /dev/ttyUSBN] [--monitor]\n")

    if args.show_table:
        print_comparison_table(collect_available_results())


if __name__ == "__main__":
    main()
