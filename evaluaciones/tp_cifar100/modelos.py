"""
Modelos del TP.

`construir(cfg)` arma una CNN a partir de una configuración declarativa, para
describir arquitecturas sin escribir una clase por modelo.

    cfg = {
        "stem": (canales, stride),                  # conv 3x3 + BN + ReLU
        "stages": [(tipo, canales, n_bloques, stride, expansion), ...],
        "head": canales | None,                     # conv 1x1 antes del pooling (opcional)
    }

Tipos de bloque:
  - "basic": bloque residual de ResNet (conv3x3-BN-ReLU-conv3x3-BN + atajo).
             El atajo es una conv 1x1 + BN cuando cambia la forma (opción B:
             ESP-DL no tiene el zero-padding por canales de la opción A).
  - "ir":    inverted residual de MobileNetV2 (1x1 expand → dw3x3 → 1x1 project).
  - "dws":   depthwise separable (dw3x3 → 1x1), sin atajo.

Pueden usar cualquier arquitectura: esto es sólo una ayuda. El baseline
(ResNet-56) está definido acá.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn(cin, cout, k=3, s=1, g=1, act=True):
    capas = [nn.Conv2d(cin, cout, k, s, k // 2, groups=g, bias=False), nn.BatchNorm2d(cout)]
    if act:
        capas.append(nn.ReLU(inplace=True))
    return nn.Sequential(*capas)


class Basic(nn.Module):
    def __init__(self, cin, cout, s, exp=1):
        super().__init__()
        self.c1 = conv_bn(cin, cout, 3, s)
        self.c2 = conv_bn(cout, cout, 3, 1, act=False)
        self.atajo = None if (s == 1 and cin == cout) else conv_bn(cin, cout, 1, s, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x if self.atajo is None else self.atajo(x)
        return self.relu(self.c2(self.c1(x)) + r)


class InvRes(nn.Module):
    def __init__(self, cin, cout, s, exp=4):
        super().__init__()
        mid = cin * exp
        capas = []
        if exp != 1:
            capas.append(conv_bn(cin, mid, 1))
        capas += [conv_bn(mid, mid, 3, s, g=mid), conv_bn(mid, cout, 1, act=False)]
        self.f = nn.Sequential(*capas)
        self.res = s == 1 and cin == cout

    def forward(self, x):
        y = self.f(x)
        return x + y if self.res else y


class DWS(nn.Module):
    def __init__(self, cin, cout, s, exp=1):
        super().__init__()
        self.f = nn.Sequential(conv_bn(cin, cin, 3, s, g=cin), conv_bn(cin, cout, 1))

    def forward(self, x):
        return self.f(x)


BLOQUES = {"basic": Basic, "ir": InvRes, "dws": DWS}


class Red(nn.Module):
    def __init__(self, cfg: dict, num_clases: int = 100):
        super().__init__()
        c, s = cfg["stem"]
        capas = [conv_bn(3, c, 3, s)]
        for tipo, cout, n, stride, exp in cfg["stages"]:
            for i in range(n):
                capas.append(BLOQUES[tipo](c, cout, stride if i == 0 else 1, exp))
                c = cout
        if cfg.get("head"):
            capas.append(conv_bn(c, cfg["head"], 1))
            c = cfg["head"]
        self.features = nn.Sequential(*capas)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c, num_clases)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.fc(torch.flatten(x, 1))


def resnet_cifar(n: int, anchos=(16, 32, 64)) -> dict:
    """ResNet de He et al. para CIFAR: profundidad 6n+2."""
    a, b, c = anchos
    return {"stem": (a, 1), "stages": [("basic", a, n, 1, 1), ("basic", b, n, 2, 1), ("basic", c, n, 2, 1)]}


CONFIGS: dict[str, dict] = {
    # Baseline provisto (y teacher para KD).
    "resnet56": resnet_cifar(9),
    "resnet20": resnet_cifar(3),
}


def construir(nombre_o_cfg, num_clases: int = 100) -> Red:
    cfg = CONFIGS[nombre_o_cfg] if isinstance(nombre_o_cfg, str) else nombre_o_cfg
    return Red(cfg, num_clases)


def exportar_onnx(modelo: nn.Module, ruta, opset: int = 13) -> None:
    """Exporta con la entrada fija [1,3,32,32], como la espera evaluar.py."""
    modelo = modelo.eval().cpu()
    x = torch.randn(1, 3, 32, 32)
    torch.onnx.export(modelo, x, str(ruta), input_names=["input"], output_names=["logits"],
                      opset_version=opset, do_constant_folding=True, dynamo=False)
