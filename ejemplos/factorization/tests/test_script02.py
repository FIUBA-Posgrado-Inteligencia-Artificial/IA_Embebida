"""Tests de script02.py (convoluciones).

Uso:
    cd examples/factorization
    uv run --with pytest python -m pytest tests -q
"""

import torch
import torch.nn as nn

from pytorch_utils import count_macs
from script02 import CNNBase, CNNDepthwise, CNNEspacial, aplicar_tucker2, contar_capas, tucker2


def test_tucker2_rango_completo_es_exacto():
    conv = nn.Conv2d(8, 12, 3, padding=1)
    x = torch.randn(2, 8, 10, 10)
    assert torch.allclose(tucker2(conv, 12, 8)(x), conv(x), atol=1e-4)


def test_tucker2_con_stride_es_exacto():
    conv = nn.Conv2d(8, 12, 3, stride=2, padding=1)
    x = torch.randn(2, 8, 10, 10)
    assert torch.allclose(tucker2(conv, 12, 8)(x), conv(x), atol=1e-4)
    assert tucker2(conv, 4, 2)(x).shape == conv(x).shape


def test_tucker2_ratio_chico():
    m = aplicar_tucker2(CNNBase(), 0.05)  # int(16·0.05) = 0 → se recorta a 1
    assert m(torch.randn(2, 1, 28, 28)).shape == (2, 10)


def test_macs_depthwise():
    bloque = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.Conv2d(16, 32, 1))
    assert count_macs(bloque, (1, 16, 14, 14)) == 14 * 14 * 16 * (9 + 32)  # 128 576


def test_macs_espacial():
    bloque = nn.Sequential(nn.Conv2d(16, 16, (3, 1), padding=(1, 0)), nn.Conv2d(16, 32, (1, 3), padding=(0, 1)))
    assert count_macs(bloque, (1, 16, 14, 14)) == 14 * 14 * (16 * 16 * 3 + 32 * 16 * 3)  # 451 584


def test_variantes_misma_salida():
    for M in (CNNBase, CNNDepthwise, CNNEspacial):
        assert M()(torch.randn(2, 1, 28, 28)).shape == (2, 10)


def test_contar_capas():
    assert contar_capas(CNNBase()) == 4
    assert contar_capas(CNNDepthwise()) == 6
    assert contar_capas(aplicar_tucker2(CNNBase(), 0.5)) == 8
