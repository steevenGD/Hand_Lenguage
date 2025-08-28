import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Configuración
SEQUENCE_DIR = "sequences"
SEQUENCE_LENGTH = 60  # Debe coincidir con el usado en la captura

# Cargar secuencias y etiquetas
X = []
y = []
for fname in os.listdir(SEQUENCE_DIR):
    if fname.endswith('.npy'):
        label = fname.split('_')[0]
        arr = np.load(os.path.join(SEQUENCE_DIR, fname))
        if arr.shape[0] == SEQUENCE_LENGTH:
            X.append(arr)
            y.append(label)
        else:
            print(f"[WARN] {fname} tiene longitud {arr.shape[0]}, se omite.")
X = np.array(X)
y = np.array(y)
print(f"Total secuencias cargadas: {len(X)}")

# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

# Definir modelo LSTM
model = keras.Sequential([
    keras.layers.Masking(mask_value=0., input_shape=(SEQUENCE_LENGTH, X.shape[2])),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.LSTM(64),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(y_cat.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Entrenar
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Evaluar
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.3f}")

# Guardar modelo y etiquetas
model.save('modelo_gestos_lstm.h5')
np.save('labels_lstm.npy', le.classes_)
print("Modelo y etiquetas guardados.")