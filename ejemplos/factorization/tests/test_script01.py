"""Tests de script01.py (SVD en el MLP).

Uso:
    cd examples/factorization
    uv run --with pytest python -m pytest tests -q
"""

import pytest
import torch
import torch.nn as nn

from pytorch_utils import count_macs
from script01 import MLP, aplicar_svd, energia_capturada, factorizar_linear, rango_limite


def test_rango_limite():
    assert rango_limite(784, 64) == pytest.approx(784 * 64 / 848)  # ≈ 59.17


def test_factorizar_rango_completo_es_exacto():
    lin = nn.Linear(20, 12)
    x = torch.randn(5, 20)
    assert torch.allclose(factorizar_linear(lin, 12)(x), lin(x), atol=1e-5)


def test_factorizar_macs():
    capa = factorizar_linear(nn.Linear(784, 64), 16)
    assert count_macs(capa, (1, 784)) == 16 * (784 + 64)  # 13 568


def test_aplicar_svd_respeta_limite():
    m = aplicar_svd(MLP(), 32)  # solo fc1 (límite ≈ 59) tiene límite > 32
    lineales = [c for c in m.net if isinstance(c, (nn.Linear, nn.Sequential))]
    assert isinstance(lineales[0], nn.Sequential)
    assert all(isinstance(c, nn.Linear) for c in lineales[1:])
    assert count_macs(m) == 32 * (784 + 64) + 64 * 32 + 32 * 16 + 16 * 10  # 29 856


def test_energia_capturada():
    assert energia_capturada(torch.tensor([4.0, 3.0]), 1) == pytest.approx(16 / 25)
