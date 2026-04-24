"""
Utilidades compartidas para el proyecto ESP32-CAM.

Secciones:
  DATASET   — STL-10 loaders con augmentation configurable
  TRAINING  — loop de entrenamiento y evaluación
  MODELS    — StandardCNN, DepthwiseCNN y factory build_model()
  METRICS   — count_parameters, count_macs, model_size_kb
  CHECKPOINT — save/load checkpoint con metadata
  ANALYSIS  — compute_entropy, print_comparison_table
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_DATASETS_ROOT = Path(__file__).resolve().parent.parent / "datasets"
_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

STL10_CLASSES = [
    "airplane", "bird", "car", "deer", "dog",
    "frog", "horse", "monkey", "ship", "truck",
]

# Estadísticas de normalización de STL-10 (calculadas sobre el train set)
_NORM_STATS = {
    "rgb":  {"mean": [0.4467, 0.4398, 0.4066], "std": [0.2241, 0.2215, 0.2239]},
    "gray": {"mean": [0.4407],                  "std": [0.2220]},
}

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

def get_dataloaders(
    input_mode: str,
    resolution: int,
    batch_size: int = 64,
    data_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Retorna (train_loader, test_loader) para STL-10.

    Args:
        input_mode: 'rgb' o 'gray'
        resolution: resolución cuadrada deseada en píxeles (STL-10 nativo: 96)
        batch_size: tamaño de lote
        data_dir: directorio de caché; por defecto examples/datasets/
    """
    if data_dir is None:
        data_dir = _DATASETS_ROOT
    if input_mode not in ("rgb", "gray"):
        raise ValueError(f"input_mode debe ser 'rgb' o 'gray', no '{input_mode}'")
    if resolution <= 0:
        raise ValueError(f"resolution debe ser un entero positivo, no {resolution}")

    mean = _NORM_STATS[input_mode]["mean"]
    std  = _NORM_STATS[input_mode]["std"]

    _STL10_NATIVE = 96

    def _base_transforms(augment: bool) -> list:
        t = []
        if resolution != _STL10_NATIVE:
            t.append(transforms.Resize((resolution, resolution)))
        if augment:
            t.append(transforms.RandomHorizontalFlip())
            t.append(transforms.RandomCrop(resolution, padding=max(2, resolution // 12)))
            if input_mode == "rgb":
                t.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        if input_mode == "gray":
            t.append(transforms.Grayscale(num_output_channels=1))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return t

    train_transform = transforms.Compose(_base_transforms(augment=True))
    test_transform  = transforms.Compose(_base_transforms(augment=False))

    train_ds = datasets.STL10(data_dir, split="train", download=True, transform=train_transform)
    test_ds  = datasets.STL10(data_dir, split="test",  download=True, transform=test_transform)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=pin)
    return train_loader, test_loader


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")
    return device


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += X.size(0)
    return total_loss / total, correct / total


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            correct += (model(X).argmax(1) == y).sum().item()
            total   += X.size(0)
    return correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    device: torch.device | None = None,
    verbose: bool = True,
) -> list[dict]:
    if device is None:
        device = get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        te_acc = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": tr_loss,
                         "train_acc": tr_acc, "test_acc": te_acc})
        if verbose:
            print(f"  Epoch {epoch:2d}/{epochs}  "
                  f"loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                  f"test_acc={te_acc:.4f}  ({elapsed:.1f}s)")
    return history


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model: nn.Module, input_size: tuple) -> int:
    """Cuenta MACs de Conv2d y Linear usando forward hooks."""
    macs = [0]
    hooks = []

    def _hook(module, inp, out):
        if isinstance(module, nn.Conv2d):
            out_h, out_w = out.shape[2], out.shape[3]
            macs[0] += (
                module.out_channels
                * (module.in_channels // module.groups)
                * module.kernel_size[0]
                * module.kernel_size[1]
                * out_h * out_w
            )
        elif isinstance(module, nn.Linear):
            macs[0] += module.in_features * module.out_features

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(_hook))

    device = next(model.parameters()).device
    dummy = torch.zeros(input_size, device=device)
    model.eval()
    with torch.no_grad():
        model(dummy)
    for h in hooks:
        h.remove()
    return macs[0]


def model_size_kb(model: nn.Module) -> float:
    return count_parameters(model) * 4 / 1024


# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d + BN + ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class DSBlock(nn.Module):
    """Depthwise separable: DW(3×3) + PW(1×1), ambos con BN + ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dw    = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.pw    = nn.Conv2d(in_channels, out_channels, 1)
        self.pw_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.dw_bn(self.dw(x)))
        x = F.relu(self.pw_bn(self.pw(x)))
        return x


class StandardCNN(nn.Module):
    """
    CNN con 3 bloques convolucionales estándar.
    Agnóstica a la resolución gracias a AdaptiveAvgPool.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        channels: tuple[int, int, int] = (32, 64, 128),
    ):
        super().__init__()
        c1, c2, c3 = channels
        self.block1     = ConvBlock(in_channels, c1)
        self.block2     = ConvBlock(c1, c2)
        self.block3     = ConvBlock(c2, c3)
        self.pool       = nn.MaxPool2d(2)
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)


class DepthwiseCNN(nn.Module):
    """
    CNN con 3 bloques depthwise separables.
    Misma topología que StandardCNN pero con menor costo computacional.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        channels: tuple[int, int, int] = (32, 64, 128),
    ):
        super().__init__()
        c1, c2, c3 = channels
        self.block1     = DSBlock(in_channels, c1)
        self.block2     = DSBlock(c1, c2)
        self.block3     = DSBlock(c2, c3)
        self.pool       = nn.MaxPool2d(2)
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)


def build_model(
    arch: str,
    input_mode: str,
    channels: tuple[int, int, int] = (32, 64, 128),
) -> nn.Module:
    """
    Factory de modelos.
    arch: 'standard' | 'depthwise'
    input_mode: 'rgb' | 'gray'
    channels: canales de cada bloque (útil para reconstruir modelos podados)
    """
    in_channels = 1 if input_mode == "gray" else 3
    if arch == "standard":
        return StandardCNN(in_channels=in_channels, channels=channels)
    if arch == "depthwise":
        return DepthwiseCNN(in_channels=in_channels, channels=channels)
    raise ValueError(f"arch desconocida: '{arch}'. Usar 'standard' o 'depthwise'.")


# ---------------------------------------------------------------------------
# CHECKPOINT
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, model: nn.Module, metadata: dict) -> None:
    """
    Guarda estado del modelo + metadata en un .pth.
    metadata debe incluir: arch, input_mode, resolution, num_classes,
    normalize_mean, normalize_std, accuracy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **metadata}, path)
    print(f"[INFO] Checkpoint guardado: {path}")


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[nn.Module, dict]:
    """
    Carga un checkpoint .pth y reconstruye el modelo.
    Retorna (model, metadata).
    """
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}

    arch        = meta["arch"]
    in_channels = 1 if meta["input_mode"] == "gray" else 3
    channels    = tuple(meta.get("channels", (32, 64, 128)))

    if arch == "standard":
        model = StandardCNN(in_channels=in_channels, channels=channels)
    elif arch == "depthwise":
        model = DepthwiseCNN(in_channels=in_channels, channels=channels)
    else:
        raise ValueError(f"arch desconocida en checkpoint: '{arch}'")

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    acc = meta.get("accuracy")
    acc_str = f"{acc:.4f}" if acc is not None else "?"
    print(f"[INFO] Checkpoint cargado: {path}  (acc={acc_str})")
    return model, meta


def _quantized_parent_name(name: str) -> str | None:
    """Para 'foo_quantized_ptq' retorna 'foo'. None si no es cuantizado."""
    idx = name.find("_quantized_")
    return name[:idx] if idx >= 0 else None


def _read_pth_metadata(pth: Path) -> dict:
    """Carga metadata de un .pth y computa params/MACs/size_kb del modelo FP32."""
    ckpt = torch.load(pth, map_location="cpu", weights_only=False)
    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    meta["name"] = pth.stem
    in_channels = 1 if meta["input_mode"] == "gray" else 3
    channels    = tuple(meta.get("channels", (32, 64, 128)))
    resolution  = meta["resolution"]
    if meta["arch"] == "standard":
        m = StandardCNN(in_channels=in_channels, channels=channels)
    else:
        m = DepthwiseCNN(in_channels=in_channels, channels=channels)
    meta["params"]  = count_parameters(m)
    meta["macs"]    = count_macs(m, (1, in_channels, resolution, resolution))
    meta["size_kb"] = model_size_kb(m)
    return meta


def collect_available_results(models_dir: Path | None = None) -> list[dict]:
    """Lee metadata de todos los .pth y .espdl en models_dir.

    Para .espdl: hereda accuracy/arch/etc del .pth padre (pre-cuantización),
    hereda params/MACs (quantización no cambia la topología del grafo) y usa
    el tamaño real del .espdl como size_kb. La accuracy mostrada es la del
    modelo FP32 padre — la caída por cuantización requeriría evaluar el .espdl.
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    results = []
    espdl_names: set[str] = set()

    for espdl in sorted(models_dir.glob("*.espdl")):
        name = espdl.stem
        parent_name = _quantized_parent_name(name)
        if parent_name is None:
            print(f"[WARN] {espdl.name}: no puedo inferir padre (falta '_quantized_')")
            continue
        parent_pth = models_dir / f"{parent_name}.pth"
        if not parent_pth.exists():
            print(f"[WARN] {espdl.name}: padre {parent_pth.name} no encontrado")
            continue
        try:
            meta = _read_pth_metadata(parent_pth)
            meta["name"] = name
            meta["size_kb"] = espdl.stat().st_size / 1024
            results.append(meta)
            espdl_names.add(name)
        except Exception as exc:
            print(f"[WARN] No se pudo leer {espdl.name}: {exc}")

    for pth in sorted(models_dir.glob("*.pth")):
        if pth.stem in espdl_names:
            continue  # el .espdl ya lo cubrió con el tamaño real
        try:
            results.append(_read_pth_metadata(pth))
        except Exception as exc:
            print(f"[WARN] No se pudo leer {pth.name}: {exc}")
    return results


# ---------------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------------

def compute_entropy(probs: torch.Tensor) -> float:
    """Entropía de Shannon normalizada a [0,1]. probs debe ser una distribución (suma=1)."""
    import math
    probs = probs.float()
    probs = probs / probs.sum()  # normalizar por si acaso
    probs = probs.clamp(min=1e-9)
    n = probs.numel()
    h = -(probs * probs.log()).sum().item()
    h_max = math.log(n)
    return h / h_max if h_max > 0 else 0.0


def print_comparison_table(results: list[dict]) -> None:
    """Imprime tabla comparativa de todos los modelos disponibles."""
    if not results:
        print("  (sin modelos disponibles)")
        return
    name_width = max(len("Modelo"), max(len(r["name"]) for r in results)) + 2
    header = (
        f"{'Modelo':<{name_width}} {'Acc':>7} {'Params':>10} "
        f"{'MACs':>12} {'KB':>8} {'Arch':>10} {'Input':>12}"
    )
    print("\n" + "=" * len(header))
    print("TABLA COMPARATIVA DE MODELOS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        mode = f"{r['input_mode']}{r['resolution']}"
        print(
            f"{r['name']:<{name_width}} {r.get('accuracy', 0):>7.4f} "
            f"{r.get('params', 0):>10,} {r.get('macs', 0):>12,} "
            f"{r.get('size_kb', 0):>8.1f} {r['arch']:>10} {mode:>12}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# PRUNING
# ---------------------------------------------------------------------------

def _l1_importance(conv: nn.Conv2d) -> torch.Tensor:
    """Importancia de cada filtro de salida: norma L1 de los pesos."""
    return conv.weight.data.abs().sum(dim=(1, 2, 3))


def _activation_importance(
    model: nn.Module,
    blocks: list[nn.Module],
    dataloader: DataLoader,
    device: torch.device,
    n_batches: int = 10,
) -> list[torch.Tensor]:
    """
    Importancia de los canales de salida de cada bloque, estimada como
    la media de la activación absoluta por canal durante n_batches de forward.
    """
    accum: list[list[torch.Tensor]] = [[] for _ in blocks]
    hooks = []

    def make_hook(i: int):
        def hook(module: nn.Module, inp, out: torch.Tensor):
            # out: (B, C, H, W)  →  mean abs por canal: (C,)
            accum[i].append(out.detach().abs().mean(dim=(0, 2, 3)).cpu())
        return hook

    for i, block in enumerate(blocks):
        hooks.append(block.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(dataloader):
            if batch_idx >= n_batches:
                break
            model(x.to(device))

    for h in hooks:
        h.remove()

    return [torch.stack(a).mean(0) for a in accum]


def _select_indices(importance: torch.Tensor, ratio: float) -> torch.Tensor:
    """Devuelve índices de los canales a conservar (top-k por importancia)."""
    n_keep = max(1, int(importance.numel() * (1 - ratio)))
    return importance.topk(n_keep).indices.sort().values


def prune_standard_cnn(
    model: StandardCNN,
    ratio: float,
    method: str,
    dataloader: DataLoader | None = None,
    device: torch.device | None = None,
) -> StandardCNN:
    """
    Pruning estructurado de StandardCNN: elimina filtros por norma L1 o activación.
    Retorna un nuevo modelo más pequeño con los pesos de los canales supervivientes.
    """
    if method == "weight":
        imp1 = _l1_importance(model.block1.conv)
        imp2 = _l1_importance(model.block2.conv)
        imp3 = _l1_importance(model.block3.conv)
    elif method == "activation":
        if dataloader is None or device is None:
            raise ValueError("activation pruning requiere dataloader y device")
        imp1, imp2, imp3 = _activation_importance(
            model, [model.block1, model.block2, model.block3], dataloader, device
        )
    else:
        raise ValueError(f"method debe ser 'weight' o 'activation', no '{method}'")

    idx1 = _select_indices(imp1, ratio)
    idx2 = _select_indices(imp2, ratio)
    idx3 = _select_indices(imp3, ratio)

    in_ch = model.block1.conv.in_channels
    pruned = StandardCNN(
        in_channels=in_ch,
        channels=(len(idx1), len(idx2), len(idx3)),
    )

    with torch.no_grad():
        # block1: seleccionar filtros de salida idx1
        pruned.block1.conv.weight.copy_(model.block1.conv.weight[idx1])
        pruned.block1.conv.bias = None if model.block1.conv.bias is None else \
            nn.Parameter(model.block1.conv.bias[idx1].clone())
        for attr in ("weight", "bias", "running_mean", "running_var"):
            getattr(pruned.block1.bn, attr).copy_(
                getattr(model.block1.bn, attr)[idx1]
            )

        # block2: entradas idx1, salidas idx2
        pruned.block2.conv.weight.copy_(model.block2.conv.weight[idx2][:, idx1])
        pruned.block2.conv.bias = None if model.block2.conv.bias is None else \
            nn.Parameter(model.block2.conv.bias[idx2].clone())
        for attr in ("weight", "bias", "running_mean", "running_var"):
            getattr(pruned.block2.bn, attr).copy_(
                getattr(model.block2.bn, attr)[idx2]
            )

        # block3: entradas idx2, salidas idx3
        pruned.block3.conv.weight.copy_(model.block3.conv.weight[idx3][:, idx2])
        pruned.block3.conv.bias = None if model.block3.conv.bias is None else \
            nn.Parameter(model.block3.conv.bias[idx3].clone())
        for attr in ("weight", "bias", "running_mean", "running_var"):
            getattr(pruned.block3.bn, attr).copy_(
                getattr(model.block3.bn, attr)[idx3]
            )

        # classifier: seleccionar columnas idx3 (entrada)
        pruned.classifier.weight.copy_(model.classifier.weight[:, idx3])
        pruned.classifier.bias.copy_(model.classifier.bias)

    return pruned


def write_model_data_meta(path, meta: dict) -> None:
    """Escribe un sidecar JSON junto a model_data.h para el preflight de build.

    Valida que `meta` contenga las llaves: num_classes, input_w, input_h,
    input_channels, mean, std, classes. Escribe ordenado y estable.
    """
    import json as _json
    from pathlib import Path as _Path

    required = ("num_classes", "input_w", "input_h", "input_channels",
                "mean", "std", "classes")
    for key in required:
        if key not in meta:
            raise KeyError(f"meta missing required key: {key}")
    _Path(path).write_text(_json.dumps(meta, indent=2, ensure_ascii=False),
                           encoding="utf-8")


def prune_depthwise_cnn(
    model: DepthwiseCNN,
    ratio: float,
    method: str,
    dataloader: DataLoader | None = None,
    device: torch.device | None = None,
) -> DepthwiseCNN:
    """
    Pruning estructurado de DepthwiseCNN.
    Para cada DSBlock se poda el canal de salida de la PW conv
    (= canal de salida del bloque completo).
    """
    if method == "weight":
        imp1 = _l1_importance(model.block1.pw)
        imp2 = _l1_importance(model.block2.pw)
        imp3 = _l1_importance(model.block3.pw)
    elif method == "activation":
        if dataloader is None or device is None:
            raise ValueError("activation pruning requiere dataloader y device")
        imp1, imp2, imp3 = _activation_importance(
            model, [model.block1, model.block2, model.block3], dataloader, device
        )
    else:
        raise ValueError(f"method debe ser 'weight' o 'activation', no '{method}'")

    idx1 = _select_indices(imp1, ratio)
    idx2 = _select_indices(imp2, ratio)
    idx3 = _select_indices(imp3, ratio)

    in_ch = model.block1.dw.in_channels
    pruned = DepthwiseCNN(
        in_channels=in_ch,
        channels=(len(idx1), len(idx2), len(idx3)),
    )

    def copy_bn(src_bn, dst_bn, idx):
        for attr in ("weight", "bias", "running_mean", "running_var"):
            getattr(dst_bn, attr).copy_(getattr(src_bn, attr)[idx])

    with torch.no_grad():
        # block1
        pruned.block1.dw.weight.copy_(model.block1.dw.weight)
        copy_bn(model.block1.dw_bn, pruned.block1.dw_bn, torch.arange(in_ch))
        pruned.block1.pw.weight.copy_(model.block1.pw.weight[idx1])
        copy_bn(model.block1.pw_bn, pruned.block1.pw_bn, idx1)

        # block2
        pruned.block2.dw.weight.copy_(model.block2.dw.weight[idx1])
        copy_bn(model.block2.dw_bn, pruned.block2.dw_bn, idx1)
        pruned.block2.pw.weight.copy_(model.block2.pw.weight[idx2][:, idx1])
        copy_bn(model.block2.pw_bn, pruned.block2.pw_bn, idx2)

        # block3
        pruned.block3.dw.weight.copy_(model.block3.dw.weight[idx2])
        copy_bn(model.block3.dw_bn, pruned.block3.dw_bn, idx2)
        pruned.block3.pw.weight.copy_(model.block3.pw.weight[idx3][:, idx2])
        copy_bn(model.block3.pw_bn, pruned.block3.pw_bn, idx3)

        # classifier
        pruned.classifier.weight.copy_(model.classifier.weight[:, idx3])
        pruned.classifier.bias.copy_(model.classifier.bias)

    return pruned
