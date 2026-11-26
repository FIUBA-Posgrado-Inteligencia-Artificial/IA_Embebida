"""
entrenar.py — entrena un modelo de modelos.CONFIGS sobre CIFAR-100, con KD opcional.

Uso:
    uv run python entrenar.py resnet56 --epocas 200                       # baseline
    uv run python entrenar.py resnet20 --epocas 200 --teacher baseline_resnet56.pth  # con KD

Guarda en salidas/ (o en --salidas): <nombre>.pth (state_dict + cfg), <nombre>.onnx (entrada
[1,3,32,32]) y <nombre>.log.json con la historia de val.

Receta: SGD (momentum 0.9, nesterov), lr 0.1 con cosine, weight decay 5e-4,
batch 128, crop+flip, AMP. Con --teacher: pérdida de Hinton,
alpha·T²·KL(softmax(s/T) || softmax(t/T)) + (1−alpha)·CE.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from datos import augment, batches, cargar_splits, normalizar
from modelos import CONFIGS, construir, exportar_onnx

SALIDAS = Path(__file__).resolve().parent / "salidas"  # default de --salidas


@torch.no_grad()
def evaluar(modelo, x, y, bs=1000):
    modelo.eval()
    ok = 0
    for xb, yb in batches(x, y, bs, shuffle=False):
        with torch.autocast("cuda", dtype=torch.float16, enabled=xb.is_cuda):
            ok += (modelo(xb).argmax(1) == yb).sum().item()
    return ok / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo", choices=sorted(CONFIGS))
    ap.add_argument("--epocas", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--teacher", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nombre", default=None, help="nombre de salida (default: el del modelo)")
    ap.add_argument("--salidas", type=Path, default=SALIDAS, help="carpeta de salida")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.salidas.mkdir(parents=True, exist_ok=True)
    nombre = args.nombre or args.modelo

    d = cargar_splits()
    xtr, ytr = normalizar(d["train"][0].to(dev)), d["train"][1].to(dev)
    xva, yva = normalizar(d["val"][0].to(dev)), d["val"][1].to(dev)

    modelo = construir(args.modelo).to(dev).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in modelo.parameters())
    print(f"[{nombre}] parámetros: {n_params:,}")

    teacher = None
    if args.teacher:
        ck = torch.load(args.teacher, map_location=dev)
        teacher = construir(ck["cfg"]).to(dev).to(memory_format=torch.channels_last)
        teacher.load_state_dict(ck["state_dict"])
        teacher.eval()
        print(f"[{nombre}] KD desde {args.teacher.name} (val {evaluar(teacher, xva, yva):.4f})")

    opt = torch.optim.SGD(modelo.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                          weight_decay=args.wd)
    pasos = args.epocas * ((len(xtr) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=pasos,
                                                pct_start=0.05, anneal_strategy="cos",
                                                div_factor=10, final_div_factor=1000)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")

    hist, mejor, t0 = [], 0.0, time.time()
    for ep in range(1, args.epocas + 1):
        modelo.train()
        for xb, yb in batches(xtr, ytr, args.bs, shuffle=True):
            xb = augment(xb).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                s = modelo(xb)
                loss = F.cross_entropy(s, yb)
                if teacher is not None:
                    with torch.no_grad():
                        t = teacher(xb)
                    kd = F.kl_div(F.log_softmax(s.float() / args.T, 1), F.softmax(t.float() / args.T, 1),
                                  reduction="batchmean") * args.T ** 2
                    loss = args.alpha * kd + (1 - args.alpha) * loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        if ep % 5 == 0 or ep == args.epocas:
            acc = evaluar(modelo, xva, yva)
            mejor = max(mejor, acc)
            hist.append({"epoca": ep, "val": acc, "seg": round(time.time() - t0)})
            print(f"[{nombre}] ep {ep:3d}  val {acc:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    modelo = modelo.float().to(memory_format=torch.contiguous_format)
    cfg = CONFIGS[args.modelo]
    torch.save({"cfg": cfg, "state_dict": modelo.state_dict()}, args.salidas / f"{nombre}.pth")
    exportar_onnx(modelo, args.salidas / f"{nombre}.onnx")
    (args.salidas / f"{nombre}.log.json").write_text(json.dumps(
        {"modelo": args.modelo, "args": {k: str(v) for k, v in vars(args).items()},
         "parametros": n_params, "val_final": hist[-1]["val"], "historia": hist}, indent=2))
    print(f"[{nombre}] listo: val final {hist[-1]['val']:.4f}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
