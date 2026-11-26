"""
Datos del TP: CIFAR-100 con un split fijo train/val y la normalización oficial.

Todo el dataset entra en la GPU (50 000 imágenes de 32×32×3 en uint8 son
~150 MB), así que la augmentation también corre en la GPU y el entrenamiento
no queda limitado por los DataLoaders.

Reglas que fija este módulo (no cambiarlas, las usa `evaluar.py`):
  - Entrada del modelo: tensor float [N, 3, 32, 32], RGB, normalizado con
    MEAN/STD de abajo.
  - Split: de las 50 000 imágenes de train, 45 000 son train y 5 000 son val
    (permutación con semilla SEMILLA_SPLIT). El test oficial (10 000) NO se
    usa para entrenar ni para elegir modelos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"  # caché local, se baja una vez

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)
SEMILLA_SPLIT = 2026
N_VAL = 5000
NUM_CLASES = 100


HF_URL = "https://huggingface.co/api/datasets/uoft-cs/cifar100/parquet/cifar100/{split}/0.parquet"


def _descargar_npz(root: Path) -> Path:
    """Baja CIFAR-100 (mirror de Hugging Face, mucho más rápido que el sitio de
    Toronto) y lo guarda como cifar100.npz en el cache compartido."""
    import io
    import urllib.request

    import pyarrow.parquet as pq
    from PIL import Image

    destino = root / "cifar100.npz"
    if destino.exists():
        return destino
    root.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for split in ("train", "test"):
        local = root / f"cifar100_{split}.parquet"
        if not local.exists():
            print(f"[datos] descargando CIFAR-100 {split} ...", flush=True)
            urllib.request.urlretrieve(HF_URL.format(split=split), local)
        tabla = pq.read_table(local).to_pydict()
        col_img = "img" if "img" in tabla else "image"
        imgs = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))
                         for d in tabla[col_img]])
        arrays[f"x_{split}"] = imgs.astype(np.uint8)
        arrays[f"y_{split}"] = np.asarray(tabla["fine_label"], dtype=np.int64)
    np.savez(destino, **arrays)
    return destino


def _cargar(train: bool, root: Path = DATASETS_DIR):
    d = np.load(_descargar_npz(root))
    split = "train" if train else "test"
    x = torch.from_numpy(d[f"x_{split}"]).permute(0, 3, 1, 2).contiguous()  # N,3,32,32 uint8
    y = torch.from_numpy(d[f"y_{split}"]).long()
    return x, y


def cargar_splits(root: Path = DATASETS_DIR):
    """Devuelve dict con 'train', 'val', 'test' → (x_uint8 [N,3,32,32], y)."""
    x, y = _cargar(True, root)
    g = torch.Generator().manual_seed(SEMILLA_SPLIT)
    perm = torch.randperm(len(x), generator=g)
    idx_val, idx_tr = perm[:N_VAL], perm[N_VAL:]
    xt, yt = _cargar(False, root)
    return {
        "train": (x[idx_tr], y[idx_tr]),
        "val": (x[idx_val], y[idx_val]),
        "test": (xt, yt),
    }


def normalizar(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [N,3,H,W] → float normalizado (mismo device)."""
    mean = torch.tensor(MEAN, device=x_uint8.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x_uint8.device).view(1, 3, 1, 1)
    return (x_uint8.float() / 255.0 - mean) / std


def augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random crop con padding de ceros + flip horizontal, por muestra, en GPU.

    `x` ya normalizado [N,3,32,32]."""
    n, _, h, w = x.shape
    dev = x.device
    xp = F.pad(x, (pad, pad, pad, pad))
    dy = torch.randint(0, 2 * pad + 1, (n,), device=dev)
    dx = torch.randint(0, 2 * pad + 1, (n,), device=dev)
    ar_h = torch.arange(h, device=dev)
    ar_w = torch.arange(w, device=dev)
    rows = (dy[:, None] + ar_h)[:, :, None].expand(n, h, w)
    cols = (dx[:, None] + ar_w)[:, None, :].expand(n, h, w)
    out = xp[torch.arange(n, device=dev)[:, None, None], :, rows, cols]  # n,h,w,c
    out = out.permute(0, 3, 1, 2)
    flip = torch.rand(n, device=dev) < 0.5
    out = torch.where(flip[:, None, None, None], out.flip(3), out)
    return out.contiguous()


def batches(x: torch.Tensor, y: torch.Tensor, bs: int, shuffle: bool):
    n = len(x)
    idx = torch.randperm(n, device=x.device) if shuffle else torch.arange(n, device=x.device)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield x[j], y[j]
