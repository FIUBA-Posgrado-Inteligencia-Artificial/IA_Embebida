#!/usr/bin/env python3
"""
fit_models_1d.py
================
Genera datos sintéticos  y = a*x + b + c*sin(d*x + e) + ruido
y ajusta 8 modelos de complejidad creciente, mostrando las curvas
ajustadas en un gráfico comparativo.

Modelos:
  1. Regresión lineal   (y = ax + b)
  2. Lineal + senoide   (y = ax + b + c*sin(d*x + e))
  3. Neurona sola ReLU  (1 entrada → 1 ReLU → 1 salida)
  4. Capa densa sin hidden (lineal, equivale a 1 neurona lineal)
  5. MLP 1 capa hidden, pocas neuronas
  6. MLP 1 capa hidden, muchas neuronas
  7. MLP 2 capas hidden, pocas neuronas
  8. MLP 2 capas hidden, muchas neuronas

Uso:
  python fit_models_1d.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import curve_fit

# ─── Reproducibilidad ───────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)

# ─── 1. Generación de datos ─────────────────────────────────────────
N = 300
x_np = np.linspace(-3, 3, N).astype(np.float32)

# Parámetros reales
a_true, b_true = 1.5, 0.5
c_true, d_true, e_true = 2.0, 2.5, 0.3
noise_std = 0.4

y_true = a_true * x_np + b_true + c_true * np.sin(d_true * x_np + e_true)
y_np = y_true + np.random.normal(0, noise_std, N).astype(np.float32)

# Tensores para PyTorch
x_t = torch.from_numpy(x_np).unsqueeze(1)  # (N, 1)
y_t = torch.from_numpy(y_np).unsqueeze(1)  # (N, 1)

# Grilla densa para graficar curvas suaves
x_plot = np.linspace(-3, 3, 600).astype(np.float32)
x_plot_t = torch.from_numpy(x_plot).unsqueeze(1)


# ─── 2. Definición de modelos ───────────────────────────────────────

# --- 2a. Regresión lineal (mínimos cuadrados analítico) ---
def fit_linear(x, y):
    """Regresión lineal por mínimos cuadrados."""
    A = np.vstack([x, np.ones_like(x)]).T
    coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return coefs  # [a, b]


# --- 2b. Lineal + senoide (optimización no lineal) ---
def model_lin_sin(x, a, b, c, d, e):
    return a * x + b + c * np.sin(d * x + e)


# --- 2c-h. Modelos neuronales con PyTorch ---

class SingleNeuronReLU(nn.Module):
    """Una sola neurona con activación ReLU."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class DenseNoHidden(nn.Module):
    """Capa densa directa (sin capas ocultas) → equivale a regresión lineal."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    """MLP genérico con N capas ocultas."""
    def __init__(self, hidden_sizes):
        super().__init__()
        layers = []
        in_dim = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_torch_model(model, x, y, lr=0.01, epochs=3000, verbose=False):
    """Entrena un modelo de PyTorch con MSE loss."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 500 == 0:
            print(f"  epoch {epoch+1:5d}  loss={loss.item():.4f}")
    return model


# ─── 3. Ajuste de todos los modelos ─────────────────────────────────
print("=" * 60)
print("Ajustando modelos...")
print("=" * 60)

results = {}  # nombre → y_plot

# 1) Regresión lineal
print("\n[1/8] Regresión lineal")
coefs = fit_linear(x_np, y_np)
y_lin = coefs[0] * x_plot + coefs[1]
results["Regresión lineal"] = y_lin
print(f"      a={coefs[0]:.3f}, b={coefs[1]:.3f}")

# 2) Lineal + senoide
print("\n[2/8] Lineal + senoide (ax+b+c·sin(dx+e))")
try:
    popt, _ = curve_fit(model_lin_sin, x_np, y_np,
                        p0=[1.0, 0.0, 1.0, 2.0, 0.0],
                        maxfev=10000)
    y_lsin = model_lin_sin(x_plot, *popt)
    results["ax+b+sin"] = y_lsin
    print(f"      a={popt[0]:.3f}, b={popt[1]:.3f}, "
          f"c={popt[2]:.3f}, d={popt[3]:.3f}, e={popt[4]:.3f}")
except RuntimeError as exc:
    print(f"      ⚠ curve_fit no convergió: {exc}")
    results["ax+b+sin"] = np.full_like(x_plot, np.nan)

# 3) Neurona sola con ReLU
print("\n[3/8] Neurona sola con ReLU")
m3 = SingleNeuronReLU()
train_torch_model(m3, x_t, y_t, lr=0.01, epochs=3000)
with torch.no_grad():
    results["Neurona ReLU"] = m3(x_plot_t).squeeze().numpy()

# 4) Capa densa sin hidden
print("\n[4/8] Capa densa (sin hidden)")
m4 = DenseNoHidden()
train_torch_model(m4, x_t, y_t, lr=0.01, epochs=3000)
with torch.no_grad():
    results["Densa (sin hidden)"] = m4(x_plot_t).squeeze().numpy()

# 5) MLP 1 hidden, pocas neuronas
print("\n[5/8] MLP 1 hidden (8 neuronas)")
m5 = MLP([8])
train_torch_model(m5, x_t, y_t, lr=0.01, epochs=5000)
with torch.no_grad():
    results["MLP [8]"] = m5(x_plot_t).squeeze().numpy()

# 6) MLP 1 hidden, muchas neuronas
print("\n[6/8] MLP 1 hidden (128 neuronas)")
m6 = MLP([128])
train_torch_model(m6, x_t, y_t, lr=0.005, epochs=5000)
with torch.no_grad():
    results["MLP [128]"] = m6(x_plot_t).squeeze().numpy()

# 7) MLP 2 hidden, pocas neuronas
print("\n[7/8] MLP 2 hidden (8, 8)")
m7 = MLP([8, 8])
train_torch_model(m7, x_t, y_t, lr=0.01, epochs=5000)
with torch.no_grad():
    results["MLP [8,8]"] = m7(x_plot_t).squeeze().numpy()

# 8) MLP 2 hidden, muchas neuronas
print("\n[8/8] MLP 2 hidden (64, 64)")
m8 = MLP([64, 64])
train_torch_model(m8, x_t, y_t, lr=0.005, epochs=5000)
with torch.no_grad():
    results["MLP [64,64]"] = m8(x_plot_t).squeeze().numpy()

print("\n" + "=" * 60)
print("¡Todos los modelos ajustados!")
print("=" * 60)


# ─── 4. Gráfico comparativo ─────────────────────────────────────────

colors = [
    "#ff5555",  # rojo
    "#50fa7b",  # verde
    "#ffb86c",  # naranja
    "#8be9fd",  # cyan
    "#bd93f9",  # púrpura
    "#ff79c6",  # rosa
    "#f1fa8c",  # amarillo
    "#6272a4",  # gris azulado
]

fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor="#282a36")
fig.suptitle("Ajuste de modelos 1D:  y = ax + b + c·sin(dx+e) + ruido",
             fontsize=16, color="white", fontweight="bold", y=0.98)

for idx, (name, y_pred) in enumerate(results.items()):
    ax = axes[idx // 4, idx % 4]
    ax.set_facecolor("#282a36")

    # Datos
    ax.scatter(x_np, y_np, s=6, alpha=0.4, color="#f8f8f2", label="Datos", zorder=1)

    # Curva real
    ax.plot(x_plot, a_true * x_plot + b_true + c_true * np.sin(d_true * x_plot + e_true),
            color="#6272a4", linewidth=1.2, linestyle="--", alpha=0.6, label="Real", zorder=2)

    # Predicción
    ax.plot(x_plot, y_pred, color=colors[idx], linewidth=2.5, label=name, zorder=3)

    ax.set_title(name, color="white", fontsize=12, fontweight="bold", pad=8)
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(y_np.min() - 1, y_np.max() + 1)
    for spine in ax.spines.values():
        spine.set_color("#44475a")
    ax.legend(fontsize=7, loc="upper left",
              facecolor="#44475a", edgecolor="#6272a4", labelcolor="white")

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Guardar
output_path = "fit_models_1d.png"
plt.savefig(output_path, dpi=150, facecolor="#282a36")
print(f"\nGráfico guardado en: {output_path}")

plt.show()


# ─── 5. Gráfico auxiliar: solo datos + curva real ───────────────────

fig2, ax2 = plt.subplots(figsize=(7, 5), facecolor="#282a36")
ax2.set_facecolor("#282a36")

y_real_plot = a_true * x_plot + b_true + c_true * np.sin(d_true * x_plot + e_true)

ax2.scatter(x_np, y_np, s=10, alpha=0.5, color="#f8f8f2", label="Datos con ruido", zorder=1)
ax2.plot(x_plot, y_real_plot, color="#6272a4", linewidth=2.0, linestyle="--",
         label=r"$y = ax + b + c \cdot \sin(dx + e)$", zorder=2)

ax2.set_xlabel("x", color="white", fontsize=12)
ax2.set_ylabel("y", color="white", fontsize=12)
ax2.tick_params(colors="white", labelsize=9)
for spine in ax2.spines.values():
    spine.set_color("#44475a")
ax2.legend(fontsize=11, facecolor="#44475a", edgecolor="#6272a4",
           labelcolor="white", loc="upper left")
ax2.set_xlim(-3.2, 3.2)
ax2.grid(True, color="#44475a", alpha=0.4)

fig2.tight_layout()

output_path2 = "fit_datos_reales.png"
fig2.savefig(output_path2, dpi=150, facecolor="#282a36", bbox_inches="tight")
print(f"Gráfico auxiliar guardado en: {output_path2}")

plt.show()

