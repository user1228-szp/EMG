import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Ruta de la carpeta con las señales
directory = "/home/manuel_acevedo/Escritorio/sebas/codes/EMG/senales"  # Cambiar por la ruta correcta
SAMPLING_RATE = 1000  # Frecuencia de muestreo en Hz

# Crear carpeta para guardar las gráficas
output_directory = os.path.join(directory, "graficas")
os.makedirs(output_directory, exist_ok=True)

def load_signal(file):
    """Carga una señal desde un archivo CSV y genera la columna de tiempo."""
    data = pd.read_csv(file, header=None).values.flatten()
    time = np.arange(len(data)) / SAMPLING_RATE
    return time, data

def plot_signal(time, signal, filename):
    """Grafica la señal y la guarda en la carpeta de salida."""
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal, label=filename)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.title(f'Señal: {filename}')
    plt.legend()
    plt.grid()

    save_path = os.path.join(output_directory, f"{filename}_plot.png")
    plt.savefig(save_path)
    plt.close()  # Cerrar la figura para ahorrar memoria

    print(f"Gráfico guardado: {save_path}")

def main():
    files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    
    if not files:
        print("No se encontraron archivos CSV en la carpeta.")
        return

    for file in files:
        file_path = os.path.join(directory, file)
        time, signal = load_signal(file_path)
        plot_signal(time, signal, file.replace(".csv", ""))

if __name__ == "__main__":
    main()
