#!/usr/bin/env python3
"""
Script simplificado para generar solo el gráfico de regresión logística.
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """Función sigmoide."""
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def logistic_regression_fit(X, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    """Implementa regresión logística usando descenso de gradiente."""
    n_samples, n_features = X.shape
    
    # Inicializar parámetros
    weights = np.zeros(n_features)
    bias = 0
    prev_cost = float('inf')
    
    for i in range(max_iterations):
        # Forward pass
        z = np.dot(X, weights) + bias
        predictions = sigmoid(z)
        
        # Calcular costo
        cost = -np.mean(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
        
        # Calcular gradientes
        dw = (1/n_samples) * np.dot(X.T, (predictions - y))
        db = (1/n_samples) * np.sum(predictions - y)
        
        # Actualizar parámetros
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        # Verificar convergencia
        if i > 0 and abs(prev_cost - cost) < tolerance:
            print(f"Convergencia alcanzada en la iteración {i}")
            break
        
        prev_cost = cost
        
        # Ajustar learning rate si es necesario
        if i > 100 and i % 100 == 0:
            if cost > prev_cost:
                learning_rate *= 0.9
    
    return weights, bias

def logistic_regression_predict(X, weights, bias):
    """Predice las probabilidades usando el modelo entrenado."""
    z = np.dot(X, weights) + bias
    return sigmoid(z)

def generate_data(n_samples=100, noise_level=0.3):
    """Genera datos sintéticos para regresión logística."""
    np.random.seed(42)
    
    # Generar puntos de clase 0 (más hacia la izquierda)
    n_class0 = n_samples // 2
    x_class0 = np.random.normal(-2, 1.5, n_class0)
    y_class0 = np.zeros(n_class0)
    
    # Generar puntos de clase 1 (más hacia la derecha)
    n_class1 = n_samples - n_class0
    x_class1 = np.random.normal(2, 1.5, n_class1)
    y_class1 = np.ones(n_class1)
    
    # Agregar ruido para crear superposición
    x_class0 += np.random.normal(0, noise_level, n_class0)
    x_class1 += np.random.normal(0, noise_level, n_class1)
    
    # Combinar datos
    X = np.concatenate([x_class0, x_class1]).reshape(-1, 1)
    y = np.concatenate([y_class0, y_class1])
    
    return X, y

def main():
    """Función principal del script."""
    # Generar datos con menos puntos
    X, y = generate_data(n_samples=80, noise_level=0.3)
    
    # Entrenar modelo con parámetros mejorados
    weights, bias = logistic_regression_fit(X, y, learning_rate=0.5, max_iterations=5000)
    
    # Crear figura (solo el primer gráfico)
    plt.figure(figsize=(8, 6))
    
    # Plotear datos
    plt.scatter(X[y==0], y[y==0], c='red', alpha=0.7, s=50, label='Clase 0', edgecolors='black')
    plt.scatter(X[y==1], y[y==1], c='blue', alpha=0.7, s=50, label='Clase 1', edgecolors='black')
    
    # Crear curva sigmoide
    x_range = np.linspace(X.min()-1, X.max()+1, 1000).reshape(-1, 1)
    y_prob = logistic_regression_predict(x_range, weights, bias)
    
    # Plotear la curva sigmoide
    plt.plot(x_range, y_prob, 'g-', linewidth=3, label='Probabilidad P(Clase=1)')
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Umbral de decisión (0.5)')
    
    # Configurar gráfico
    plt.xlabel('Característica X', fontsize=18)
    plt.ylabel('Clase / Probabilidad', fontsize=18)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    
    # Guardar figura
    output_path = "../figures/00_intro-logistic_regression.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figura guardada en: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    main()
