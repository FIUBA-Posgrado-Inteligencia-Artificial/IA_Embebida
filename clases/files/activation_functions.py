#!/usr/bin/env python3
"""
Script para generar las 6 funciones de activación más comunes en deep learning.
Genera imágenes individuales para ReLU, Sigmoid, Tanh, LeakyReLU, ELU y Swish.
"""

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    """Función ReLU (Rectified Linear Unit)."""
    return np.maximum(0, x)

def sigmoid(x):
    """Función Sigmoid."""
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def tanh(x):
    """Función Tanh."""
    return np.tanh(x)

def leaky_relu(x, alpha=0.1):
    """Función LeakyReLU."""
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    """Función ELU (Exponential Linear Unit)."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish(x):
    """Función Swish."""
    return x * sigmoid(x)

def plot_activation_function(x, y, title, filename, color='blue'):
    """
    Genera y guarda una imagen de función de activación.
    
    Args:
        x: Valores del eje x
        y: Valores del eje y (función evaluada)
        title: Título de la función
        filename: Nombre del archivo para guardar
        color: Color de la línea
    """
    plt.figure(figsize=(6, 4))
    
    # Plotear la función
    plt.plot(x, y, color=color, linewidth=3)
    
    # Configurar el gráfico
    plt.xlabel('x', fontsize=20)
    plt.ylabel('f(x)', fontsize=20)
    # plt.title(title, fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Configurar límites de los ejes
    plt.xlim(-6, 6)
    plt.ylim(-2, 2)
    
    # Agregar líneas de referencia
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # Ajustar layout y guardar
    plt.tight_layout()
    output_path = f"../figures/{filename}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figura guardada: {filename}")
    plt.close()

def main():
    """Función principal que genera todas las funciones de activación."""
    print("Generando funciones de activación...")
    
    # Crear rango de valores x
    x = np.linspace(-6, 6, 1000)
    
    # 1. ReLU
    y_relu = relu(x)
    plot_activation_function(x, y_relu, 'ReLU', '02-activation_relu.png', 'red')
    
    # 2. Sigmoid
    y_sigmoid = sigmoid(x)
    plot_activation_function(x, y_sigmoid, 'Sigmoid', '02-activation_sigmoid.png', 'blue')
    
    # 3. Tanh
    y_tanh = tanh(x)
    plot_activation_function(x, y_tanh, 'Tanh', '02-activation_tanh.png', 'green')
    
    # 4. LeakyReLU
    y_leaky_relu = leaky_relu(x, alpha=0.04)
    plot_activation_function(x, y_leaky_relu, 'LeakyReLU', '02-activation_leakyrelu.png', 'orange')
    
    # 5. ELU
    y_elu = elu(x, alpha=1.0)
    plot_activation_function(x, y_elu, 'ELU', '02-activation_elu.png', 'purple')
    
    # 6. Swish
    y_swish = swish(x)
    plot_activation_function(x, y_swish, 'Swish', '02-activation_swish.png', 'brown')
    
    print("\n¡Todas las funciones de activación han sido generadas exitosamente!")
    print("Archivos guardados en: ../figures/")

if __name__ == "__main__":
    main()
