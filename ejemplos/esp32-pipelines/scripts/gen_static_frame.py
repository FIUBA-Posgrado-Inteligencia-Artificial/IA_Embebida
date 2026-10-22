"""
ESP32-CAM — gen_static_frame: imagen fija embebida para placas sin cámara
==========================================================================
Genera `static_frame.h`: una única imagen de STL-10 (test set), reescalada a
la resolución pedida (default 320x240) y empaquetada en RGB565 (mismo layout
que `fb->buf` de la OV2640: `RRRRRGGG GGGBBBBB`, uint16 nativo), para que las
placas sin cámara (`BOARD_HAS_CAMERA=0`) tengan un frame determinístico y
realista para ejercitar el pipeline de inferencia/web en vez de un gradiente
sintético.

No depende de ningún modelo entrenado: sólo del dataset STL-10.

Uso:
    uv run python scripts/gen_static_frame.py --class-name car
    uv run python scripts/gen_static_frame.py --class-name car --size 320x240 \
        --output firmware/main/static_frame.h
"""

import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torchvision.datasets import STL10

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "pipelib"))
from esp32cam_utils import STL10_CLASSES

# Este script vive en <example>/scripts/; la raíz del ejemplo es parents[1].
_EXAMPLE_ROOT = Path(__file__).resolve().parents[1]

_DATASETS_ROOT = _EXAMPLE_ROOT.parent / "datasets"

DEFAULT_OUTPUT = _EXAMPLE_ROOT / "firmware" / "main" / "static_frame.h"


def rgb888_to_rgb565(r: int, g: int, b: int) -> int:
    """Empaqueta un pixel RGB888 (uint8 c/u) al layout RGB565 nativo del
    sensor: RRRRRGGG GGGBBBBB (mismo criterio que
    `preprocess_px_rgb565_to_rgb888` en el firmware, invertido)."""
    return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)


def _parse_size(spec: str) -> tuple[int, int]:
    w_str, _, h_str = spec.partition("x")
    return int(w_str), int(h_str)


def load_first_image_of_class(class_name: str) -> torch.Tensor:
    """Devuelve la primera imagen del test set de STL-10 de la clase pedida,
    como tensor float CHW en [0,1] a la resolución nativa (96x96)."""
    if class_name not in STL10_CLASSES:
        raise ValueError(
            f"clase desconocida: {class_name!r} (válidas: {STL10_CLASSES})")
    class_idx = STL10_CLASSES.index(class_name)

    dataset = STL10(root=str(_DATASETS_ROOT), split="test", download=True,
                     transform=transforms.ToTensor())
    for img, label in dataset:
        if int(label) == class_idx:
            return img
    raise RuntimeError(f"no se encontró ninguna imagen de clase {class_name!r}")


def image_to_rgb565_rows(img01: torch.Tensor, width: int, height: int) -> list[int]:
    """Reescala img01 (CHW float [0,1], cualquier tamaño) a (height, width) y
    devuelve una lista plana (row-major) de valores RGB565 uint16."""
    resized = transforms.Resize((height, width))(img01)
    px = (resized * 255.0).round().clamp(0, 255).to(torch.uint8)  # CHW
    px = px.permute(1, 2, 0)  # HWC
    out = []
    for y in range(height):
        for x in range(width):
            r, g, b = (int(v) for v in px[y, x])
            out.append(rgb888_to_rgb565(r, g, b))
    return out


def generate_header(class_name: str, width: int, height: int, output: Path) -> Path:
    img01 = load_first_image_of_class(class_name)
    pixels = image_to_rgb565_rows(img01, width, height)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        f.write("/* AUTO-GENERADO por gen_static_frame.py — no editar a mano. */\n")
        f.write("#ifndef STATIC_FRAME_HEADER_GUARD_\n"
                 "#define STATIC_FRAME_HEADER_GUARD_\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define STATIC_FRAME_W {width}\n")
        f.write(f"#define STATIC_FRAME_H {height}\n")
        f.write(f'#define STATIC_FRAME_CLASS "{class_name}"\n\n')
        f.write(f"static const uint16_t STATIC_FRAME[{width * height}] = {{\n")
        for i in range(0, len(pixels), 12):
            row = pixels[i:i + 12]
            f.write("    " + ", ".join(f"0x{v:04X}" for v in row) + ",\n")
        f.write("};\n\n")
        f.write("#endif /* STATIC_FRAME_HEADER_GUARD_ */\n")
    return output


def main():
    p = argparse.ArgumentParser(
        description="Genera static_frame.h a partir de una imagen de STL-10")
    p.add_argument("--class-name", default="car", choices=STL10_CLASSES,
                    help="clase STL-10 a usar (default: car)")
    p.add_argument("--size", default="320x240",
                    help="resolución de salida WxH (default: 320x240)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"header de salida (default: {DEFAULT_OUTPUT})")
    args = p.parse_args()

    width, height = _parse_size(args.size)
    out = generate_header(args.class_name, width, height, args.output)
    size_kb = out.stat().st_size / 1024
    print(f"[OK] {out} ({width}x{height}, clase={args.class_name}, "
          f"{size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
