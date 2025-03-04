# Importar librerías necesarias
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Crear carpeta para guardar gráficas
results_folder = "results_graphs"
os.makedirs(results_folder, exist_ok=True)

# Contador de gráficas
graph_count = 0
signal_counts = 0

# Función para ajustar la longitud de la señal con recorte de 240 puntos
def window_signal(signal, window_size):
    if len(signal) >= window_size:
        return signal[:window_size]  # Recorte de la señal
    return np.pad(signal, (0, window_size - len(signal)), 'constant')

# Función para análisis con ventanas traslapantes
def sliding_window(signal, window_size, overlap):
    step = int(window_size * (1 - overlap))
    windows = [signal[i:i + window_size] for i in range(0, len(signal) - window_size + 1, step)]
    return np.array(windows)

# Configuración del modelo
window_size = 240
overlap = 0.5  # 50% de traslape
data = []
labels = []
directory = "/home/manuel_acevedo/Escritorio/sebas/codes/EMG/senales"  # Cambiar por la ruta correcta

# Leer y procesar señales
for filename in os.listdir(directory):
    if filename.startswith("e"):
        print(f"Omitiendo {filename}...")
        continue
    try:
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath, header=None)
        signal = df[0].values
        windowed_signals = sliding_window(signal, window_size, overlap)
        for ws in windowed_signals:
            adjusted_signal = window_signal(ws, window_size)
            data.append(adjusted_signal)
            labels.append(filename.split('_')[0])
            signal_counts += 1
    except Exception as e:
        print(f"Error al leer {filename}: {e}")

print(f"Número total de señales procesadas: {signal_counts}")

# Crear DataFrame
df_data = pd.DataFrame(data)
df_data['label'] = labels

# Codificación de etiquetas
encoder = LabelEncoder()
df_data['label'] = encoder.fit_transform(df_data['label'])
labels = to_categorical(df_data['label'])

# Guardar LabelEncoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# División con estratificación
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, temp_idx in splitter.split(df_data.iloc[:, :-1], df_data['label']):
    X_train, X_temp = df_data.iloc[train_idx, :-1], df_data.iloc[temp_idx, :-1]
    y_train, y_temp = labels[train_idx], labels[temp_idx]

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.333, random_state=42, stratify=np.argmax(y_temp, axis=1))

# Normalización
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Redimensionar para LSTM
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Modelo LSTM
model = Sequential([
    LSTM(64, input_shape=(window_size, 1), return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenamiento
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32, callbacks=[early_stopping])
model.save('modelo_LSTM_model_RT2.h5')

# Evaluación
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Matriz de Confusión
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title('Matriz de Confusión')
plt.savefig(os.path.join(results_folder, 'confusion_matrix.png'))

# Clasificación de nueva señal
response = input("¿Hay una señal a clasificar? (S/N): ").strip().upper()
if response == "S":
    filepath = input("Ruta del archivo de la señal: ").strip()
    try:
        df_signal = pd.read_csv(filepath, header=None)
        signal = df_signal[0].values
        windowed_signals = sliding_window(signal, window_size, overlap)
        predictions = []
        for ws in windowed_signals:
            adjusted_signal = window_signal(ws, window_size)
            signal_data = np.expand_dims(scaler.transform([adjusted_signal]), axis=-1)
            prediction = model.predict(signal_data)
            predictions.append(np.argmax(prediction))
        final_class = max(set(predictions), key=predictions.count)
        print(f"La señal pertenece a la familia: {encoder.classes_[final_class]}")
    except Exception as e:
        print(f"Error al procesar la señal: {e}")
