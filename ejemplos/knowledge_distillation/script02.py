"""
Sección 11 — Knowledge Distillation: Feature-based KD (Avanzado)
=================================================================
Implementa destilación basada en características (feature-based KD):
el Student imita no sólo las predicciones finales del Teacher (response-based,
script01.py), sino también sus representaciones internas (feature maps).

Fundamento (Romero et al. 2015 — FitNets):
  L_total = (1-λ) · CE(student_logits, labels)
           + λ    · MSE(student_hint, teacher_hint.detach())

  donde teacher_hint y student_hint son feature maps de capas intermedias
  seleccionadas (las "hint layers"). Si las dimensiones difieren, se usa
  una capa de proyección lineal (regressor).

Arquitecturas:
  Teacher:  CNN  Conv(1→6)→Pool→Conv(6→16)→Pool→FC(256→120→84→10)
  Student:  MLP  Flatten→FC(784→64→32→16→10)
  Hint (Teacher): salida de la segunda Conv (antes del pool) → (16, 4, 4)
  Hint (Student): salida de la capa FC(784→64) → (64,) → proyectada a (16,4,4)

Comparación:
  - MLP Baseline (sin KD)
  - KD response-based (T=4, α=0.7)   — referencia del script01.py
  - KD feature-based  (λ=0.5)        — este script

Uso:
    cd examples/knowledge_distillation
    uv run python script02.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_utils import (
    count_parameters,
    evaluate,
    get_dataloaders,
    get_device,
    model_size_kb,
    print_model_stats,
    save_results,
    train,
    train_epoch,
)

OUTPUT_DIR       = "outputs"
EPOCHS_TEACHER   = 10
EPOCHS_STUDENT   = 10
BATCH_SIZE       = 256
LR               = 1e-3
LAMBDA_FEAT      = 0.5   # peso de la pérdida de feature matching


# ---------------------------------------------------------------------------
# Arquitecturas
# ---------------------------------------------------------------------------

class TeacherCNN(nn.Module):
    """
    CNN Teacher con hook points para exponer feature maps intermedias.
    Misma arquitectura que script01.py.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2))
        # Hint layer: salida de conv2 ANTES del pool → shape (16, 4, 4) con input 28×28
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 120), nn.ReLU(),
            nn.Linear(120, 84),  nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.conv1(x)                    # → (B, 6, 12, 12)
        feat = self.relu2(self.conv2(x))     # → (B, 16, 8, 8)  ← hint antes del pool
        x = self.pool2(feat)                 # → (B, 16, 4, 4)
        x = self.classifier(x)
        return x, feat  # retorna logits Y feature map de hint


class StudentMLP(nn.Module):
    """
    MLP Student con una capa "hint" que será proyectada para imitar al Teacher.
    Misma arquitectura que script01.py más una capa de hint.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 64)   # ← hint layer (salida antes de ReLU)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.flatten(x)
        hint = self.relu1(self.fc1(x))   # (B, 64) ← hint
        x = self.relu2(self.fc2(hint))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x, hint


class HintProjector(nn.Module):
    """
    Proyecta el hint del Student (B, 64) al mismo espacio que el hint del Teacher
    (B, 16, 8, 8) → aplanado (B, 16*8*8 = 1024).
    Usamos una proyección lineal simple para mantenerlo entrenable.
    """
    def __init__(self, student_hint_dim: int = 64, teacher_hint_dim: int = 16 * 8 * 8):
        super().__init__()
        self.proj = nn.Linear(student_hint_dim, teacher_hint_dim)

    def forward(self, student_hint):
        return self.proj(student_hint)


# ---------------------------------------------------------------------------
# Training con KD response-based (referencia)
# ---------------------------------------------------------------------------

class KDLossResponse(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T = temperature; self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, s_logits, t_logits, labels):
        loss_hard = self.ce(s_logits, labels)
        s_soft = F.log_softmax(s_logits / self.T, dim=1)
        t_soft = F.softmax(t_logits   / self.T, dim=1)
        loss_soft = F.kl_div(s_soft, t_soft, reduction="batchmean") * (self.T ** 2)
        return (1 - self.alpha) * loss_hard + self.alpha * loss_soft


def train_response_kd(student, teacher, train_loader, test_loader, epochs, device):
    """Entrena student con KD response-based (soft targets)."""
    student = student.to(device)
    teacher.eval()
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    kd_loss = KDLossResponse(temperature=4.0, alpha=0.7)
    history = []
    for epoch in range(1, epochs + 1):
        student.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            s_logits, _ = student(X)
            with torch.no_grad():
                t_logits, _ = teacher(X)
            loss = kd_loss(s_logits, t_logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            correct    += (s_logits.argmax(1) == y).sum().item()
            total      += X.size(0)
        te_acc = evaluate_student(student, test_loader, device)
        history.append({"epoch": epoch, "test_acc": te_acc})
        print(f"  Epoch {epoch:2d}/{epochs}  loss={total_loss/total:.4f}  test_acc={te_acc:.4f}")
    return history


# ---------------------------------------------------------------------------
# Training con KD feature-based
# ---------------------------------------------------------------------------

def train_feature_kd(student, teacher, projector, train_loader, test_loader,
                     epochs, lam, device):
    """
    Entrena student con feature-based KD:
      L = (1-λ)·CE(student_logits, labels) + λ·MSE(proj(student_hint), teacher_hint_flat)
    """
    student   = student.to(device)
    projector = projector.to(device)
    teacher.eval()
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(projector.parameters()), lr=LR
    )
    ce_loss = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, epochs + 1):
        student.train()
        projector.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            s_logits, s_hint = student(X)
            with torch.no_grad():
                _, t_hint = teacher(X)

            # Proyectar hint del student al espacio del teacher
            s_hint_proj = projector(s_hint)                    # (B, 16*8*8)
            t_hint_flat = t_hint.flatten(1)                    # (B, 16*8*8)

            loss_ce   = ce_loss(s_logits, y)
            loss_feat = F.mse_loss(s_hint_proj, t_hint_flat.detach())
            loss      = (1 - lam) * loss_ce + lam * loss_feat

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            correct    += (s_logits.argmax(1) == y).sum().item()
            total      += X.size(0)

        te_acc = evaluate_student(student, test_loader, device)
        history.append({"epoch": epoch, "test_acc": te_acc})
        print(f"  Epoch {epoch:2d}/{epochs}  loss={total_loss/total:.4f}  test_acc={te_acc:.4f}")

    return history


def evaluate_student(student: nn.Module, loader, device) -> float:
    student.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits, _ = student(X)
            correct += (logits.argmax(1) == y).sum().item()
            total   += X.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    # ------------------------------------------------------------------
    # 1. Entrenar Teacher
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 1 — Entrenando Teacher CNN")
    print("="*60)
    teacher = TeacherCNN().to(device)

    # Wrapper para train() de pytorch_utils (espera forward(x) → logits)
    class TeacherWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): logits, _ = self.m(x); return logits

    tw = TeacherWrapper(teacher).to(device)
    train(tw, train_loader, test_loader, epochs=EPOCHS_TEACHER, lr=LR, device=device)
    teacher_acc = evaluate_student(teacher, test_loader, device)
    print(f"\nTeacher acc: {teacher_acc:.4f}  Paráms: {count_parameters(teacher):,}")
    results["teacher"] = {"acc": round(teacher_acc, 4), "params": count_parameters(teacher)}

    # ------------------------------------------------------------------
    # 2. Baseline MLP (sin KD)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 2 — Baseline MLP (sin KD)")
    print("="*60)
    student_base = StudentMLP().to(device)

    class StudentWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): logits, _ = self.m(x); return logits

    sw = StudentWrapper(student_base).to(device)
    train(sw, train_loader, test_loader, epochs=EPOCHS_STUDENT, lr=LR, device=device)
    base_acc = evaluate_student(student_base, test_loader, device)
    print(f"\nBaseline MLP acc: {base_acc:.4f}  Paráms: {count_parameters(student_base):,}")
    results["baseline_mlp"] = {"acc": round(base_acc, 4), "params": count_parameters(student_base)}

    # ------------------------------------------------------------------
    # 3. KD Response-based (referencia, script01)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 3 — KD Response-based (T=4, α=0.7) [referencia]")
    print("="*60)
    student_resp = StudentMLP().to(device)
    train_response_kd(student_resp, teacher, train_loader, test_loader,
                      EPOCHS_STUDENT, device)
    resp_acc = evaluate_student(student_resp, test_loader, device)
    print(f"\nKD Response-based acc: {resp_acc:.4f}  Δbaseline: {resp_acc-base_acc:+.4f}")
    results["kd_response"] = {"acc": round(resp_acc, 4), "params": count_parameters(student_resp)}

    # ------------------------------------------------------------------
    # 4. KD Feature-based (este script)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 4 — KD Feature-based (λ=0.5)")
    print("  Student hint:   salida de fc1 (B, 64)")
    print("  Teacher hint:   salida de conv2+ReLU antes del pool (B, 16, 8, 8)")
    print("  Proyector:      Linear(64 → 16*8*8 = 1024)")
    print("="*60)
    student_feat = StudentMLP().to(device)
    projector    = HintProjector(student_hint_dim=64,
                                  teacher_hint_dim=16 * 8 * 8).to(device)
    hist_feat = train_feature_kd(student_feat, teacher, projector,
                                  train_loader, test_loader,
                                  EPOCHS_STUDENT, LAMBDA_FEAT, device)
    feat_acc = evaluate_student(student_feat, test_loader, device)
    print(f"\nKD Feature-based acc: {feat_acc:.4f}  Δbaseline: {feat_acc-base_acc:+.4f}")
    results["kd_feature"] = {
        "acc": round(feat_acc, 4),
        "params": count_parameters(student_feat),
        "lambda": LAMBDA_FEAT,
    }

    # ------------------------------------------------------------------
    # 5. Tabla comparativa
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("TABLA COMPARATIVA")
    print("="*60)
    print(f"{'Modelo':<35} {'Acc':>8} {'Paráms':>10} {'Δbaseline':>10}")
    print("-" * 65)
    rows = [
        ("Teacher CNN",                   teacher_acc, count_parameters(teacher),      None),
        ("Baseline MLP (sin KD)",         base_acc,    count_parameters(student_base), 0.0),
        ("KD Response-based (T=4, α=0.7)", resp_acc,   count_parameters(student_resp), resp_acc - base_acc),
        ("KD Feature-based (λ=0.5)",      feat_acc,    count_parameters(student_feat), feat_acc - base_acc),
    ]
    for name, acc, params, delta in rows:
        d_str = f"{delta:+.4f}" if delta is not None else "—"
        print(f"  {name:<33} {acc:>8.4f} {params:>10,} {d_str:>10}")

    # ------------------------------------------------------------------
    # 6. Gráficos
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) Comparación de accuracy
    ax = axes[0]
    names = ["Teacher\nCNN", "Baseline\nMLP", "KD\nResponse", "KD\nFeature"]
    accs  = [teacher_acc, base_acc, resp_acc, feat_acc]
    colors = ["#4C72B0", "#DD8452", "#8be9fd", "#55A868"]
    bars  = ax.bar(names, accs, color=colors)
    ax.set_ylim(min(accs) - 0.05, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Comparación de accuracy")
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.4f}",
                ha="center", fontsize=8)

    # (b) Curva de aprendizaje: feature-based
    ax = axes[1]
    feat_accs_hist = [h["test_acc"] for h in hist_feat]
    ax.plot(range(1, len(feat_accs_hist) + 1), feat_accs_hist,
            "o-", color="#55A868", label="KD Feature-based", linewidth=2)
    ax.axhline(base_acc, color="#DD8452", linestyle="--", label=f"Baseline ({base_acc:.4f})")
    ax.axhline(resp_acc, color="#8be9fd", linestyle=":",  label=f"KD Response ({resp_acc:.4f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Curva de aprendizaje: KD Feature-based")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Knowledge Distillation: Response-based vs Feature-based\n"
                 "Feature-based: el student imita representaciones internas del teacher",
                 fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "kd_feature_resultados.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\n[INFO] Gráfico guardado en '{plot_path}'")
    plt.close()

    save_results(os.path.join(OUTPUT_DIR, "kd_feature_resultados.json"), results)

    print("\n[CONCLUSIÓN]")
    print(f"  Baseline MLP:         {base_acc:.4f}")
    print(f"  KD Response-based:    {resp_acc:.4f}  ({resp_acc-base_acc:+.4f})")
    print(f"  KD Feature-based:     {feat_acc:.4f}  ({feat_acc-base_acc:+.4f})")
    print(f"  Feature-based vs Response: {feat_acc-resp_acc:+.4f}")
    print(f"\n  Al imitar representaciones intermedias, el student recibe señales")
    print(f"  más ricas que sólo las predicciones finales, lo que puede mejorar")
    print(f"  la transferencia del 'dark knowledge' del teacher.")


if __name__ == "__main__":
    main()
