"""
Utilidades compartidas por los scripts script03{a,b,c}-quantize*.

Agrupa:
  - helpers de calibración e inferencia de shape del input,
  - wrapper del flujo PTQ de esp-ppq,
  - exportación a formato .espdl,
  - generación del header C (model_data.h) para el firmware ESP-IDF.

Estas funciones se usan tanto por el flujo puro-PTQ (03a) como por los flujos
QAT que terminan en PTQ via esp-ppq (03b, 03c).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Calibración
# ---------------------------------------------------------------------------

def get_calibration_data(
    dataloader, n_batches: int, device: torch.device
) -> list[torch.Tensor]:
    """Lista de batches de entrada para calibración PTQ."""
    data = []
    for i, (x, _) in enumerate(dataloader):
        if i >= n_batches:
            break
        data.append(x.to(device))
    return data


def infer_input_shape(meta: dict) -> list[int]:
    """Shape [C, H, W] según la metadata del checkpoint."""
    in_ch = 1 if meta["input_mode"] == "gray" else 3
    res = meta["resolution"]
    return [in_ch, res, res]


# ---------------------------------------------------------------------------
# esp-ppq PTQ
# ---------------------------------------------------------------------------

def run_esppq_ptq(
    model,
    calib_data: list[torch.Tensor],
    meta: dict,
    target: str = "c",
) -> tuple[torch.nn.Module, bool]:
    """
    Cuantización PTQ con esp-ppq. Retorna (quantized_graph, success).

    Usa espdl_quantize_torch con skip_export=True para que onnxsim pueda resolver
    Reshape estáticos antes de calibración (evita el error "Reshape Op Execution Error"
    que ocurre con quantize_torch_model en opset 18).
    """
    import os, tempfile
    try:
        from esp_ppq.api import espdl_quantize_torch, QuantizationSettingFactory
        from esp_ppq import TargetPlatform

        device_str = str(next(model.parameters()).device)
        setting = QuantizationSettingFactory.espdl_setting()

        # espdl_quantize_torch requiere un path de salida para derivar el dir
        # y escribir el ONNX intermedio; con skip_export=True no escribe el .espdl.
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_espdl = os.path.join(tmp_dir, "tmp.espdl")
            graph = espdl_quantize_torch(
                model=model,
                espdl_export_file=tmp_espdl,
                calib_dataloader=calib_data,
                calib_steps=len(calib_data),
                input_shape=[1] + infer_input_shape(meta),
                target=target,
                collate_fn=lambda x: x.to(device_str),
                setting=setting,
                device=device_str,
                error_report=False,
                skip_export=True,
            )
        return graph, True
    except ImportError:
        warnings.warn(
            "[ADVERTENCIA] esp-ppq no disponible. "
            "Instalar con: uv sync (ya declarado en pyproject.toml)\n"
            "Exportando solo el modelo float como fallback."
        )
        return model, False
    except Exception as e:
        warnings.warn(f"[ADVERTENCIA] ESP-PPQ falló: {e}\nExportando modelo float como fallback.")
        return model, False


# Targets soportados por el pipeline. La clave es el string que espera
# `espdl_quantize_torch(target=...)`; el valor es el nombre del TargetPlatform
# de esp-ppq usado al exportar el .espdl.
#
# "c"       → kernels genéricos en C, corre en cualquier chip (ESP32 clásico).
# "esp32s3" → kernels SIMD del S3; el .espdl NO es intercambiable con "c".
_TARGET_PLATFORMS = {
    "c":       "ESPDL_INT8",
    "esp32s3": "ESPDL_S3_INT8",
}

VALID_TARGETS = tuple(_TARGET_PLATFORMS)


def resolve_target_platform(target: str):
    """Traduce un target del pipeline al TargetPlatform de esp-ppq."""
    if target not in _TARGET_PLATFORMS:
        raise ValueError(
            f"target desconocido: {target!r}. Válidos: {sorted(_TARGET_PLATFORMS)}"
        )
    from esp_ppq import TargetPlatform
    return getattr(TargetPlatform, _TARGET_PLATFORMS[target])


def export_espdl(quantized_model, output_path: Path, meta: dict,
                 target: str = "c") -> bool:
    """Exporta el modelo cuantizado a formato .espdl. Fallback: guarda .pth."""
    try:
        from esp_ppq.api import export_ppq_graph
        from esp_ppq.IR.base.graph import BaseGraph
        if not isinstance(quantized_model, BaseGraph):
            raise TypeError(f"Se esperaba BaseGraph, recibido {type(quantized_model).__name__}")
        platform = resolve_target_platform(target)
        export_ppq_graph(quantized_model, platform, str(output_path))
        print(f"[INFO] Exportado: {output_path} (target={target})")
        return True
    except Exception as e:
        warnings.warn(f"[ADVERTENCIA] Export .espdl falló: {e}")
        try:
            if hasattr(quantized_model, "state_dict"):
                torch.save(
                    {"model_state_dict": quantized_model.state_dict(), **meta},
                    output_path.with_suffix(".pth"),
                )
        except Exception:
            pass
        print("[INFO] Fallback: guardado como .pth (sin formato espdl nativo)")
        return False


# ---------------------------------------------------------------------------
# Firmware C header
# ---------------------------------------------------------------------------

def generate_model_data_h(
    espdl_path: Path,
    firmware_dir: Path,
    meta: dict,
    mean_list: list[float],
    std_list: list[float],
    classes: list[str],
    source_script: str = "script04a-quantize.py",
    input_exponent: int = -7,
) -> Path:
    """
    Genera firmware/main/model_data.h a partir del .espdl.

    Args:
        espdl_path: ruta al archivo .espdl exportado (si no existe, se genera
            un header con array vacío).
        firmware_dir: destino del header.
        meta: metadata del checkpoint (input_mode, resolution).
        mean_list, std_list: estadísticas de normalización en escala [0,1].
        classes: nombres de clases (en orden de índice).
        input_exponent: exponente de cuantización de la ENTRADA del .espdl
            (escala 2^exp). El preprocess del firmware debe cuantizar con
            este factor; asumir -7 (1/128) satura si el modelo calibró -6.
        source_script: nombre del script que llama (para el comentario header).

    Retorna el Path del header generado.
    """
    firmware_dir.mkdir(parents=True, exist_ok=True)
    header_path = firmware_dir / f"{espdl_path.stem}_model_data.h"

    in_ch = 1 if meta["input_mode"] == "gray" else 3
    res = meta["resolution"]

    mean_uint8 = [int(m * 255) for m in mean_list]
    std_uint8 = [int(s * 255) for s in std_list]

    if espdl_path.exists() and espdl_path.suffix == ".espdl":
        raw = espdl_path.read_bytes()
        hex_bytes = ", ".join(f"0x{b:02X}" for b in raw)
        array_len = len(raw)
    else:
        hex_bytes = "/* modelo no disponible en formato .espdl */"
        array_len = 0

    n_classes = len(classes)
    classes_str = ", ".join(f'"{c}"' for c in classes)

    rgb_mean_lines = (
        f"#define MODEL_INPUT_MEAN_1     {mean_uint8[1]}\n"
        f"#define MODEL_INPUT_MEAN_2     {mean_uint8[2]}\n"
        if in_ch == 3 else ""
    )
    rgb_std_lines = (
        f"#define MODEL_INPUT_STD_1      {std_uint8[1]}\n"
        f"#define MODEL_INPUT_STD_2      {std_uint8[2]}\n"
        if in_ch == 3 else ""
    )

    content = (
        f"/* Generado automáticamente por {source_script} — no editar */\n"
        f"/* Fuente: {espdl_path.name} */\n"
        f"\n"
        f"#pragma once\n"
        f"#include <stddef.h>\n"
        f"#include <stdint.h>\n"
        f"\n"
        f"/* Configuración del input */\n"
        f"#define MODEL_INPUT_W          {res}\n"
        f"#define MODEL_INPUT_H          {res}\n"
        f"#define MODEL_INPUT_CHANNELS   {in_ch}   /* 1=gray, 3=RGB */\n"
        f"/* Exponente de cuantización de la entrada: int8 = normalizado * 2^-EXP.\n"
        f" * Viene de la calibración del .espdl — el preprocess DEBE usarlo. */\n"
        f"#define MODEL_INPUT_EXPONENT   ({input_exponent})\n"
        f"\n"
        f"/* Normalización (escala uint8 [0,255]) */\n"
        f"#define MODEL_INPUT_MEAN_0     {mean_uint8[0]}\n"
        f"{rgb_mean_lines}"
        f"#define MODEL_INPUT_STD_0      {std_uint8[0]}\n"
        f"{rgb_std_lines}"
        f"\n"
        f"/* Clases */\n"
        f"#define MODEL_NUM_CLASSES      {n_classes}\n"
        f"static const char* const MODEL_CLASSES[{n_classes}] = {{{classes_str}}};\n"
        f"\n"
        f"/* Datos del modelo */\n"
        f"/* Alineado a 16 bytes: esp-dl v3 sólo mapea el modelo in-place desde\n"
        f" * flash (MODEL_LOCATION_IN_FLASH_RODATA) si la dirección está alineada\n"
        f" * a 16 bytes. Si no lo está, cae a un camino de copia a heap que pide\n"
        f" * la memoria con capability de PSRAM — en placas sin PSRAM (p.ej. el\n"
        f" * módulo remarcado del ESP32-S3 relevado en 2026-08-07) ese malloc\n"
        f" * devuelve NULL y esp-dl no lo chequea antes de escribir, lo que\n"
        f" * produce un StoreProhibited panic en el primer boot. No quitar este\n"
        f" * atributo por prolijidad: sin él el crash es silencioso y sólo se\n"
        f" * manifiesta en hardware sin PSRAM. */\n"
        f"static const uint8_t model_data[] __attribute__((aligned(16))) = {{{hex_bytes}}};\n"
        f"static const size_t  model_data_len = {array_len};\n"
    )
    header_path.write_text(content, encoding="utf-8")
    print(f"[INFO] Generado: {header_path}")
    return header_path


def get_input_exponent(graph) -> int:
    """Exponente de cuantización de la ENTRADA del grafo cuantizado (2^exp).

    espdl usa escalas power-of-two; el preprocess del firmware debe cuantizar
    con este mismo exponente (MODEL_INPUT_EXPONENT en model_data.h)."""
    import math
    var = next(iter(graph.inputs.values()))
    op = var.dest_ops[0]
    cfg = op.config.input_quantization_config[op.inputs.index(var)]
    exp = math.log2(float(cfg.scale))
    if abs(exp - round(exp)) > 1e-6:
        raise ValueError(f"Escala de entrada no power-of-two: {float(cfg.scale)}")
    return int(round(exp))
