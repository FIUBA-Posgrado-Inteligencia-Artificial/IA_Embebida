#!/usr/bin/env python3
"""
Big O Notation Visualization Script
Generates a comprehensive plot showing various time complexities
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import math

# Set style for better aesthetics
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_big_o_plot():
    """Create a comprehensive Big O notation visualization"""
    
    # Create figure with high DPI for better quality
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Define input sizes (n)
    n = np.linspace(1, 20, 1000)
    
    # Define Big O functions
    functions = {
        'O(1)': np.ones_like(n),
        'O(log n)': np.log2(n),
        'O(n)': n,
        'O(n log n)': n * np.log2(n),
        'O(n²)': n**2,
        'O(n³)': n**3,
        'O(2ⁿ)': 2**n,
        'O(n!)': np.array([math.sqrt(2 * math.pi * x) * (x / math.e)**x if x <= 10 else np.inf for x in n])
    }
    
    # Color palette for better visual distinction
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f'   # gray
    ]
    
    # Plot each function and store scaled values for arrow calculation
    scaled_functions = {}
    for i, (name, values) in enumerate(functions.items()):
        # Handle infinity values for n!
        finite_mask = np.isfinite(values)
        if name == 'O(n!)':
            # Only plot up to n=10 for factorial
            plot_n = n[finite_mask]
            plot_values = values[finite_mask]
        else:
            plot_n = n
            plot_values = values
        
        # Scale values for better visualization
        if name == 'O(1)':
            scaled_values = plot_values * 2.0
        elif name == 'O(log n)':
            scaled_values = plot_values * 1.6
        elif name == 'O(n)':
            scaled_values = plot_values * 0.7
        elif name == 'O(n log n)':
            scaled_values = plot_values * 0.3
        elif name == 'O(n²)':
            scaled_values = plot_values * 0.16
        elif name == 'O(n³)':
            scaled_values = plot_values * 0.03
        elif name == 'O(2ⁿ)':
            scaled_values = plot_values * 0.1 * 2
        elif name == 'O(n!)':
            scaled_values = plot_values * 0.01 * 4
        
        # Store scaled values for arrow calculation
        scaled_functions[name] = (plot_n, scaled_values)
        
        # Plot with different line styles for better distinction
        line_style = '-' if i < 4 else '--' if i < 6 else '-.'
        line_width = 5.0 if i < 4 else 4.0
        
        ax.plot(plot_n, scaled_values, 
                label=name, 
                color=colors[i], 
                linewidth=line_width,
                linestyle=line_style,
                alpha=0.6)
    
    
    # Customize the plot
    ax.set_xlabel('Tamaño de la entrada (n)', fontsize=30, fontweight='bold')
    ax.set_ylabel('Complejidad temporal', fontsize=30, fontweight='bold')
    
    # Set axis limits for better visualization
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 25)
    
    # Customize grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove legend
    
    # Add black arrow axes
    # X-axis arrow
    ax.annotate('', xy=(20, 0), xytext=(1, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=5))
    
    # Y-axis arrow
    ax.annotate('', xy=(1, 25), xytext=(1, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=5))
    
    # Remove axis numbers
    ax.set_xticks([])
    ax.set_yticks([])

    # Define xy (arrow head) and xytext (arrow tail) coordinates for annotation arrows
    annotation_xy = {
        'O(1)':       (15.0,  2.0),
        'O(log n)':   (15.0,  6.3),
        'O(n)':       (15.0, 11.0),
        'O(n log n)': (15.0, 18.3),
        'O(n²)':      (11.8, 21.0),
        'O(n³)':      ( 9.1, 21.0),
        'O(2ⁿ)':      ( 6.8, 21.0),
        'O(n!)':      ( 5.7, 21.0)
    }
    
    annotation_xytext = {
        'O(1)':       (15.0,  4.0),
        'O(log n)':   (15.0,  8.0),
        'O(n)':       (15.0, 13.0),
        'O(n log n)': (15.0, 23.0),
        'O(n²)':      (13.0, 23.0),
        'O(n³)':      (10.0, 23.0),
        'O(2ⁿ)':      ( 7.5, 23.0),
        'O(n!)':      ( 4.0, 23.0)
    }
    
    # Add annotations with arrows for all curves
    # O(1) - Constant
    ax.annotate('O(1)', xy=annotation_xy['O(1)'], xytext=annotation_xytext['O(1)'],
                arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=2),
                fontsize=26, color='#1f77b4', fontweight='bold')
    
    # O(log n) - Logarithmic (same height as O(n), more to the right)
    ax.annotate('O(log n)', xy=annotation_xy['O(log n)'], xytext=annotation_xytext['O(log n)'],
                arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=2),
                fontsize=26, color='#ff7f0e', fontweight='bold')
    
    # O(n) - Linear
    ax.annotate('O(n)', xy=annotation_xy['O(n)'], xytext=annotation_xytext['O(n)'],
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2),
                fontsize=26, color='#2ca02c', fontweight='bold')
    
    # O(n log n) - Linearithmic (above the curve)
    ax.annotate('O(n log n)', xy=annotation_xy['O(n log n)'], xytext=annotation_xytext['O(n log n)'],
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=2),
                fontsize=26, color='#d62728', fontweight='bold')
    
    # O(n²) - Quadratic (above the curve)
    ax.annotate('O(n²)', xy=annotation_xy['O(n²)'], xytext=annotation_xytext['O(n²)'],
                arrowprops=dict(arrowstyle='->', color='#9467bd', lw=2),
                fontsize=26, color='#9467bd', fontweight='bold')
    
    # O(n³) - Cubic (higher up)
    ax.annotate('O(n³)', xy=annotation_xy['O(n³)'], xytext=annotation_xytext['O(n³)'],
                arrowprops=dict(arrowstyle='->', color='#8c564b', lw=2),
                fontsize=26, color='#8c564b', fontweight='bold')
    
    # O(2ⁿ) - Exponential (left side)
    ax.annotate('O(2ⁿ)', xy=annotation_xy['O(2ⁿ)'], xytext=annotation_xytext['O(2ⁿ)'],
                arrowprops=dict(arrowstyle='->', color='#e377c2', lw=2),
                fontsize=26, color='#e377c2', fontweight='bold')
    
    # O(n!) - Factorial (left side)
    ax.annotate('O(n!)', xy=annotation_xy['O(n!)'], xytext=annotation_xytext['O(n!)'],
                arrowprops=dict(arrowstyle='->', color='#7f7f7f', lw=2),
                fontsize=26, color='#7f7f7f', fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax

def main():
    """Main function to generate and save the plot"""
    print("Generando visualización de notación Big O...")
    
    # Create the plot
    fig, ax = create_big_o_plot()
    
    # Save the plot
    output_path = '/home/fzacchigna/projects/phd-thesis/figures/general_bigo.png'
    fig.savefig(output_path, 
                dpi=150, 
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none',
                format='png')
    
    print(f"Gráfico guardado en: {output_path}")
    
    print("¡Visualización completada!")

if __name__ == "__main__":
    main()
