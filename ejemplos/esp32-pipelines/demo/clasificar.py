"""Clasifica imágenes en la PC con los modelos de la demo (grande o chico).

Es la versión "de alto nivel" de lo que después corre en la ESP32-CAM: el
mismo modelo, en PyTorch y en FP32, sobre una o más imágenes.

Uso (desde examples/esp32-pipelines/):
    uv run python demo/clasificar.py --modelo grande demo/imagenes/avion.png
    uv run python demo/clasificar.py --modelo chico demo/imagenes/*.png
"""

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR.parent / "pipelib"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from esp32cam_utils import STL10_CLASSES, count_macs, count_parameters, load_checkpoint  # noqa: E402

MODELOS = {
    "grande": DEMO_DIR / "demo_grande.pth",
    "chico": DEMO_DIR / "demo_chico.pth",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", choices=MODELOS, default="grande",
                   help="modelo a usar (default: %(default)s)")
    p.add_argument("imagenes", nargs="+", type=Path, help="una o más imágenes")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    # load_checkpoint imprime la ruta absoluta del .pth: no aporta en la demo.
    with contextlib.redirect_stdout(io.StringIO()):
        model, meta = load_checkpoint(MODELOS[args.modelo], device)
    model.eval()

    res = meta["resolution"]
    preprocess = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize(meta["normalize_mean"], meta["normalize_std"]),
    ])
    in_ch = 1 if meta["input_mode"] == "gray" else 3
    print(f"\nModelo {args.modelo}: entrada {res}x{res}, "
          f"{count_parameters(model):,} parámetros, "
          f"{count_macs(model, (1, in_ch, res, res)) / 1e6:.2f} M MACs, "
          f"accuracy (test) {meta['accuracy']:.1%}\n")

    for path in args.imagenes:
        img = Image.open(path).convert("L" if in_ch == 1 else "RGB")
        x = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            t0 = time.perf_counter()
            probs = torch.softmax(model(x), dim=1)[0]
            ms = (time.perf_counter() - t0) * 1000
        top = torch.topk(probs, 3)
        top3 = "  |  ".join(f"{STL10_CLASSES[int(i)]} {float(p):.0%}"
                            for p, i in zip(top.values, top.indices))
        print(f"{path.name:<14} -> {top3}   ({ms:.1f} ms en PC)")


if __name__ == "__main__":
    main()
