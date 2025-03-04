import os
from collections import defaultdict

# Ruta de la carpeta con las señales
directory = "/home/manuel_acevedo/Escritorio/sebas/codes/EMG/senales"  # Cambiar por la ruta correcta

def count_signals(directory):
    """Cuenta cuántos archivos empiezan con el mismo prefijo."""
    prefix_counts = defaultdict(int)
    
    files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    
    if not files:
        print("No se encontraron archivos CSV en la carpeta.")
        return {}

    for file in files:
        prefix = file.split("_")[0]  # Tomar la parte antes del primer guión bajo
        prefix_counts[prefix] += 1
    
    return prefix_counts

def main():
    counts = count_signals(directory)
    
    if counts:
        print("Conteo de señales por prefijo:")
        for prefix, count in counts.items():
            print(f"{prefix}: {count}")
    else:
        print("No hay señales para contar.")

if __name__ == "__main__":
    main()