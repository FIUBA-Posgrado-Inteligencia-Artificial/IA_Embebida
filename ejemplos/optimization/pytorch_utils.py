"""
Utilidades compartidas para los scripts de PyTorch (secciones 7-11).

Provee:
- get_dataloaders()      : DataLoaders de Fashion MNIST
- get_device()           : cuda / cpu
- train_epoch()          : un epoch de entrenamiento
- evaluate()             : accuracy en un DataLoader
- train()                : loop completo de entrenamiento con reporte por epoch
- count_parameters()     : cantidad de parámetros entrenables
- count_macs()           : MACs para Conv2d y Linear (sin dependencias externas)
- print_model_stats()    : imprime parámetros, MACs y tamaño estimado
- save_results()         : guarda resultados en JSON
"""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Shared dataset cache across all example topics
_DATASETS_ROOT = Path(__file__).resolve().parent.parent / "datasets"
_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def get_dataloaders(batch_size: int = 256, data_dir: Path | str | None = None) -> tuple:
    """
    Retorna (train_loader, test_loader) para Fashion MNIST.
    Descarga el dataset la primera vez en `data_dir`.
    Por defecto usa examples/datasets/ compartido entre todos los temas.
    Normaliza a media=0.5, std=0.5 → rango aproximado [-1, 1].
    """
    if data_dir is None:
        data_dir = _DATASETS_ROOT
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_ds = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Dispositivo
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Retorna GPU si está disponible, si no CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")
    return device


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Un epoch de entrenamiento.
    Retorna (loss_promedio, accuracy).
    """
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
    """Retorna accuracy (float en [0,1]) sobre el DataLoader dado."""
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
    """
    Entrena `model` por `epochs` epochs con Adam + CrossEntropy.
    Retorna lista de dicts {epoch, train_loss, train_acc, test_acc}.
    """
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
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "test_acc": te_acc})
        if verbose:
            print(f"  Epoch {epoch:2d}/{epochs}  loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}  ({elapsed:.1f}s)")
    return history


# ---------------------------------------------------------------------------
# Conteo de complejidad
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Cantidad total de parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model: nn.Module, input_size: tuple = (1, 1, 28, 28)) -> int:
    """
    Cuenta MACs (Multiply-Accumulate operations) para capas Conv2d y Linear.
    No requiere torchinfo — usa hooks para inspeccionar las activaciones reales.

    Retorna el total de MACs para una sola muestra de entrada.
    """
    macs = [0]

    hooks = []

    def make_hook(layer):
        def hook(module, inp, out):
            if isinstance(module, nn.Conv2d):
                # MACs = C_out * C_in/groups * kH * kW * H_out * W_out
                out_h, out_w = out.shape[2], out.shape[3]
                macs[0] += (
                    module.out_channels
                    * (module.in_channels // module.groups)
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                    * out_h * out_w
                )
            elif isinstance(module, nn.Linear):
                # MACs = in_features * out_features
                macs[0] += module.in_features * module.out_features
        return hook

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(make_hook(layer)))

    device = next(model.parameters()).device
    dummy = torch.zeros(input_size, device=device)
    model.eval()
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return macs[0]


def model_size_kb(model: nn.Module) -> float:
    """Tamaño estimado del modelo en KB (parámetros float32)."""
    return count_parameters(model) * 4 / 1024


def print_model_stats(model: nn.Module, label: str = "", input_size: tuple = (1, 1, 28, 28)) -> None:
    """Imprime parámetros, MACs y tamaño estimado del modelo."""
    params = count_parameters(model)
    macs   = count_macs(model, input_size)
    size   = model_size_kb(model)
    header = f"[{label}] " if label else ""
    print(f"  {header}Parámetros: {params:,}  |  MACs: {macs:,}  |  Tamaño: {size:.1f} KB")


# ---------------------------------------------------------------------------
# Persistencia de resultados
# ---------------------------------------------------------------------------

def save_results(output_path: str, results: dict) -> None:
    """Guarda `results` como JSON en `output_path`."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Resultados guardados en '{output_path}'")
