#!/usr/bin/env python3
"""
Script simplificado para generar el gráfico de regresión lineal.
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=50, noise_level=5):
    """Genera datos sintéticos para regresión lineal."""
    np.random.seed(42)
    # Generar X aleatorio
    X = np.linspace(0, 10, n_samples)
    # Relación lineal y = mx + b + ruido
    m = 3.5
    b = 10
    y = m * X + b + np.random.normal(0, noise_level, n_samples)
    return X.reshape(-1, 1), y

def linear_regression_fit(X, y):
    """Implementa regresión lineal usando la solución cerrada (OLS)."""
    # Añadir columna de 1s para el sesgo (bias intercept)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Calcular beta = (X^T * X)^-1 * X^T * y
    beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return beta

def main():
    """Función principal del script."""
    # Generar datos
    X, y = generate_data(n_samples=40, noise_level=6)
    
    # Ajustar modelo (obtener parámetros OLS)
    beta = linear_regression_fit(X, y)
    bias = beta[0]
    weight = beta[1]
    
    # Crear figura
    plt.figure(figsize=(8, 6))
    
    # Plotear datos observados
    plt.scatter(X, y, c='dodgerblue', alpha=0.7, s=60, label='Datos observados', edgecolors='black')
    
    # Crear recta de regresión
    x_range = np.linspace(X.min()-1, X.max()+1, 100).reshape(-1, 1)
    y_pred = weight * x_range + bias
    
    # Plotear la recta ajustada
    plt.plot(x_range, y_pred, 'r-', linewidth=3, label='Ajuste Lineal OLS')
    
    # Plotear los residuos (errores cuadráticos) para ilustrar qué se minimiza
    y_pred_points = weight * X.flatten() + bias
    for i in range(len(X)):
        # Plotear la línea residual para el primer punto para la leyenda, u otras sin label
        if i == 0:
            plt.plot([X[i][0], X[i][0]], [y[i], y_pred_points[i]], 'gray', linestyle='--', alpha=0.5, label='Residuales (Error)')
        else:
            plt.plot([X[i][0], X[i][0]], [y[i], y_pred_points[i]], 'gray', linestyle='--', alpha=0.5)
            
    # Configurar gráfico
    plt.xlabel('Variable Independiente $X$', fontsize=18)
    plt.ylabel('Variable Dependiente $y$', fontsize=18)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar figura
    output_path = "../figures/00_intro-regression.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figura guardada en: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    main()
