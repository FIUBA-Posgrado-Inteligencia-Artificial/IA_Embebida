"""
Sección 11 — Knowledge Distillation (KD)
==========================================
Demuestra destilación del conocimiento basada en respuestas (response-based KD):
  - Teacher: CNN (grande, alta accuracy)
  - Student: MLP (pequeño, entrena con y sin KD)

La pérdida de KD combina:
  (1-α) · CE(logits_student, hard_labels)
  + α   · KL(softmax(logits_student/T), softmax(logits_teacher/T))

Experimentos:
  1. Baseline MLP (sin KD, hard labels)
  2. KD MLP con temperatura fija T=4, α=0.7
  3. Barrido de temperaturas T ∈ {1, 2, 4, 8} para ver su efecto
  4. Análisis de "dark knowledge": confusiones del student KD ≈ confusiones del teacher

Uso:
    cd examples/knowledge_distillation
    uv run python script01.py
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
EPOCHS_TEACHER   = 10   # epochs para entrenar el teacher CNN
EPOCHS_STUDENT   = 10   # epochs para entrenar cada student
EPOCHS_SWEEP     = 3    # epochs para el barrido de temperaturas
BATCH_SIZE       = 256
LR               = 1e-3

FASHION_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ---------------------------------------------------------------------------
# Arquitecturas
# ---------------------------------------------------------------------------

class TeacherCNN(nn.Module):
    """CNN: Conv(1→6) → MaxPool → Conv(6→16) → MaxPool → FC(256→120→84→10)."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 120), nn.ReLU(),
            nn.Linear(120, 84),  nn.ReLU(),
            nn.Linear(84, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


class StudentMLP(nn.Module):
    """MLP pequeño: Flatten → Linear(784→64→32→16→10)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 64), nn.ReLU(),
            nn.Linear(64, 32),  nn.ReLU(),
            nn.Linear(32, 16),  nn.ReLU(),
            nn.Linear(16, 10),
        )
    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Pérdida KD
# ---------------------------------------------------------------------------

class KDLoss(nn.Module):
    """
    Pérdida de Knowledge Distillation:
        L = (1-alpha) * CE(student_logits, labels)
          + alpha     * T² * KL(student_soft, teacher_soft)

    donde soft = softmax(logits / T).
    El factor T² compensa la escala del gradiente del término KL.
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha
        self.ce    = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Hard label loss
        loss_hard = self.ce(student_logits, labels)

        # Soft target loss (KL divergence)
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft = F.softmax(teacher_logits   / self.T, dim=1)
        loss_soft = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (self.T ** 2)

        return (1 - self.alpha) * loss_hard + self.alpha * loss_soft


# ---------------------------------------------------------------------------
# Training con KD
# ---------------------------------------------------------------------------

def train_with_kd(
    student: nn.Module,
    teacher: nn.Module,
    train_loader,
    test_loader,
    epochs: int,
    temperature: float = 4.0,
    alpha: float = 0.7,
    lr: float = LR,
    device: torch.device = None,
) -> list[dict]:
    """
    Entrena `student` con KD usando `teacher` (frozen) como guía.
    Retorna historial de entrenamiento.
    """
    if device is None:
        device = get_device()
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    kd_loss_fn = KDLoss(temperature=temperature, alpha=alpha)
    history = []

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            student_logits = student(X)
            with torch.no_grad():
                teacher_logits = teacher(X)
            loss = kd_loss_fn(student_logits, teacher_logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            correct    += (student_logits.argmax(1) == y).sum().item()
            total      += X.size(0)
        te_acc = evaluate(student, test_loader, device)
        history.append({"epoch": epoch, "train_loss": total_loss/total,
                         "train_acc": correct/total, "test_acc": te_acc})
        print(f"  Epoch {epoch:2d}/{epochs}  loss={total_loss/total:.4f}  "
              f"train_acc={correct/total:.4f}  test_acc={te_acc:.4f}")

    return history


# ---------------------------------------------------------------------------
# Análisis de dark knowledge: matriz de confusión de errores
# ---------------------------------------------------------------------------

def get_confusion_errors(model: nn.Module, loader, device: torch.device) -> np.ndarray:
    """
    Retorna una matriz 10×10 donde entry[i,j] = fracción de ejemplos de clase i
    predichos como clase j (incluyendo la diagonal correcta).
    Normalizada por fila.
    """
    model.eval()
    confusion = np.zeros((10, 10))
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(1).cpu().numpy()
            labels = y.cpu().numpy()
            for true, pred in zip(labels, preds):
                confusion[true, pred] += 1
    row_sums = confusion.sum(axis=1, keepdims=True)
    return confusion / (row_sums + 1e-8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    # ---------------------------------------------------------------
    # 1. Entrenar Teacher CNN
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 1 — Entrenando Teacher CNN")
    print("="*60)
    teacher = TeacherCNN().to(device)
    train(teacher, train_loader, test_loader, epochs=EPOCHS_TEACHER, lr=LR, device=device)
    teacher_acc = evaluate(teacher, test_loader, device)
    print(f"\nTeacher acc: {teacher_acc:.4f}")
    print_model_stats(teacher, "Teacher CNN")
    results["teacher"] = {"acc": round(teacher_acc, 4), "params": count_parameters(teacher)}

    # ---------------------------------------------------------------
    # 2. Baseline MLP (sin KD)
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 2 — Entrenando Baseline MLP (sin KD)")
    print("="*60)
    student_base = StudentMLP().to(device)
    train(student_base, train_loader, test_loader, epochs=EPOCHS_STUDENT, lr=LR, device=device)
    base_acc = evaluate(student_base, test_loader, device)
    print(f"\nBaseline MLP acc: {base_acc:.4f}")
    print_model_stats(student_base, "Baseline MLP")
    results["student_baseline"] = {"acc": round(base_acc, 4), "params": count_parameters(student_base)}

    # ---------------------------------------------------------------
    # 3. KD MLP (T=4, alpha=0.7)
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 3 — Entrenando KD MLP (T=4, α=0.7)")
    print("="*60)
    student_kd = StudentMLP().to(device)
    train_with_kd(student_kd, teacher, train_loader, test_loader,
                  epochs=EPOCHS_STUDENT, temperature=4.0, alpha=0.7, device=device)
    kd_acc = evaluate(student_kd, test_loader, device)
    print(f"\nKD MLP acc: {kd_acc:.4f}  (mejora vs baseline: {kd_acc - base_acc:+.4f})")
    results["student_kd_T4"] = {"acc": round(kd_acc, 4), "params": count_parameters(student_kd)}

    # ---------------------------------------------------------------
    # 4. Barrido de temperaturas
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("PASO 4 — Barrido de temperaturas T ∈ {1, 2, 4, 8}")
    print("="*60)
    temps = [1, 2, 4, 8]
    sweep_accs = []
    for T in temps:
        print(f"\n  T={T} ---")
        s = StudentMLP().to(device)
        train_with_kd(s, teacher, train_loader, test_loader,
                      epochs=EPOCHS_SWEEP, temperature=float(T), alpha=0.7,
                      device=device)
        acc_T = evaluate(s, test_loader, device)
        sweep_accs.append(acc_T)
        results[f"student_kd_T{T}"] = {"acc": round(acc_T, 4), "temperature": T}
        print(f"  T={T}  acc={acc_T:.4f}")

    # ---------------------------------------------------------------
    # 5. Tabla comparativa
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("TABLA COMPARATIVA")
    print("="*60)
    print(f"{'Modelo':<30} {'Acc':>8} {'Paráms':>10}")
    print("-" * 50)
    print(f"{'Teacher CNN':<30} {teacher_acc:>8.4f} {count_parameters(teacher):>10,}")
    print(f"{'Baseline MLP':<30} {base_acc:>8.4f} {count_parameters(student_base):>10,}")
    print(f"{'KD MLP (T=4, α=0.7)':<30} {kd_acc:>8.4f} {count_parameters(student_kd):>10,}")
    print(f"\n  KD mejora al baseline en: {kd_acc - base_acc:+.4f}")

    # ---------------------------------------------------------------
    # 6. Gráficos
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) Comparación de accuracy
    ax = axes[0]
    models_acc = [teacher_acc, base_acc, kd_acc]
    labels_acc = ["Teacher\nCNN", "Baseline\nMLP", "KD MLP\n(T=4)"]
    colors_acc = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(labels_acc, models_acc, color=colors_acc)
    ax.set_ylim(min(models_acc) - 0.05, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy: Teacher vs Student")
    for bar, v in zip(bars, models_acc):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.4f}",
                ha="center", fontsize=8)
    ax.annotate("", xy=(2, kd_acc), xytext=(1, base_acc),
                arrowprops=dict(arrowstyle="->", color="#55A868", lw=1.5))
    ax.text(1.5, (base_acc + kd_acc)/2, f"+{kd_acc-base_acc:.4f}",
            ha="center", va="bottom", fontsize=8, color="#55A868")

    # (b) Barrido de temperaturas
    ax = axes[1]
    ax.plot(temps, sweep_accs, "o-", color="#bd93f9", linewidth=2, markersize=8)
    ax.axhline(base_acc, color="#DD8452", linestyle="--", label=f"Baseline MLP ({base_acc:.4f})")
    ax.set_xlabel("Temperatura T")
    ax.set_ylabel("Accuracy")
    ax.set_title("Efecto de la Temperatura en KD")
    ax.set_xticks(temps)
    ax.legend(fontsize=8)
    for t, a in zip(temps, sweep_accs):
        ax.text(t, a + 0.001, f"{a:.4f}", ha="center", fontsize=7)

    # (c) Dark knowledge: errores del student KD ≈ errores del teacher
    ax = axes[2]
    conf_teacher = get_confusion_errors(teacher, test_loader, device)
    conf_kd      = get_confusion_errors(student_kd, test_loader, device)
    # Comparar off-diagonal (errores): correlación entre patrones de error
    diag = np.eye(10, dtype=bool)
    teacher_errors = conf_teacher[~diag]
    kd_errors      = conf_kd[~diag]
    ax.scatter(teacher_errors * 100, kd_errors * 100, alpha=0.5, s=20, color="#4C72B0")
    max_err = max(teacher_errors.max(), kd_errors.max()) * 100
    ax.plot([0, max_err], [0, max_err], "r--", alpha=0.5, label="Confusiones iguales")
    corr = np.corrcoef(teacher_errors, kd_errors)[0, 1]
    ax.set_xlabel("Error Teacher (%)")
    ax.set_ylabel("Error KD Student (%)")
    ax.set_title(f"'Dark Knowledge': patrones de error\n(correlación r={corr:.3f})")
    ax.legend(fontsize=7)

    fig.suptitle("Knowledge Distillation: CNN Teacher → MLP Student", fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "kd_resultados.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\n[INFO] Gráfico guardado en '{plot_path}'")
    plt.close()

    save_results(os.path.join(OUTPUT_DIR, "kd_resultados.json"), results)

    print("\n[CONCLUSIÓN]")
    print(f"  KD mejora al Baseline MLP en {kd_acc - base_acc:+.4f} sin aumentar parámetros.")
    print(f"  Los patrones de error del KD Student se correlacionan con los del Teacher (r={corr:.3f}),")
    print(f"  mostrando que el 'dark knowledge' (relaciones inter-clase) se transfiere.")


if __name__ == "__main__":
    main()
