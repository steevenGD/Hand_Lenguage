# Sistema de Reconocimiento de Lenguaje de Señas en Tiempo Real

Un sistema interactivo de aprendizaje de lenguaje de señas que utiliza **MediaPipe**, **OpenCV**, **TensorFlow** y **Tkinter** para proporcionar reconocimiento de gestos en tiempo real con retroalimentación inmediata.

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Requisitos del Sistema](#-requisitos-del-sistema)
- [Instalación](#-instalación)
- [Ejecución del Proyecto](#-ejecución-del-proyecto)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Uso de la Aplicación](#-uso-de-la-aplicación)
- [Entrenamiento del Modelo](#-entrenamiento-del-modelo)

## ✨ Características

- ✅ **Detección en tiempo real** de manos con MediaPipe
- ✅ **Reconocimiento de gestos** usando redes neuronales LSTM
- ✅ **Interfaz gráfica intuitiva** con Tkinter
- ✅ **Modo de práctica** para aprendizaje autodidacta
- ✅ **Retroalimentación inmediata** (Correcto/Incorrecto/No detectado)
- ✅ **5 gestos básicos** del lenguaje de señas americano:
  - 👋 Hola
  - 👋 Adiós  
  - ✌️ Paz
  - 👍 Sí
  - ❤️ Te quiero
- ✅ **Arquitectura modular** para fácil expansión
- ✅ **Captura de datos** para entrenamiento personalizado

## 🖥️ Requisitos del Sistema

### Hardware Mínimo
- **Procesador:** Intel i3 o AMD equivalente
- **RAM:** 4 GB mínimo (8 GB recomendado)
- **Cámara web:** HD (720p) o superior
- **Espacio en disco:** 2 GB libres

### Software
- **Sistema Operativo:** Windows 10/11, macOS 10.14+, o Ubuntu 18.04+
- **Python:** 3.8 - 3.10 ⚠️ **IMPORTANTE: MediaPipe no es compatible con Python 3.11+**
- **Cámara web** funcional

## 📦 Instalación

### Paso 1: Verificar la versión de Python

```bash
python --version
# o
python3 --version
```

**⚠️ IMPORTANTE:** Asegúrate de tener Python 3.8, 3.9 o 3.10. Si tienes Python 3.11+, debes instalar una versión compatible.

### Paso 2: Clonar el repositorio

```bash
git clone https://github.com/steevenGD/Hand_Lenguage.git
cd Hand_Lenguage
```

### Paso 3: Crear entorno virtual (RECOMENDADO)

#### En Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### En macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Paso 4: Instalar dependencias

```bash
pip install --upgrade pip
pip install mediapipe opencv-python tensorflow numpy pillow scikit-learn
```

**Lista completa de dependencias:**
```
mediapipe>=0.10.0
opencv-python>=4.8.0
tensorflow>=2.13.0
numpy>=1.24.0
pillow>=10.0.0
scikit-learn>=1.3.0
```

## 🚀 Ejecución del Proyecto

### Opción 1: Ejecutar la aplicación principal

#### En PyCharm:
1. Abre `Hand_lenguage.py`
2. Haz clic en el botón **Run** ▶️ o presiona **Shift+F10**

#### En Visual Studio Code:
1. Abre `Hand_lenguage.py`
2. Presiona **F5** o ve a **Run → Start Debugging**

#### Desde la terminal:
```bash
# Asegúrate de que el entorno virtual esté activado
python Hand_lenguage.py
```

### Opción 2: Capturar nuevos gestos (Opcional)

```bash
python Capture_Gesture.py
```

**Controles de captura:**
- **1-19:** Seleccionar gesto (1=Reposo, 2=Hola, 3=Adiós, 4=Sí, 5=Paz, 6=Te quiero)
- **s:** Iniciar captura de secuencia (60 frames) para modelos LSTM.
- **q:** Salir

### Opción 3: Entrenar el modelo (Opcional)

```bash
python train_gesture_model.py
```

## 📁 Estructura del Proyecto

```
Hand_Lenguage/
├── Hand_lenguage.py          # Aplicación principal con interfaz gráfica
├── Capture_Gesture.py        # Sistema de captura de gestos
├── train_gesture_model.py    # Entrenamiento del modelo LSTM
├── modelo_gestos_lstm.keras  # Modelo entrenado
├── labels_lstm.npy          # Etiquetas del modelo
├── gestures_data.csv        # Datos de gestos en formato CSV
├── imagenes_gestos/         # Imágenes de referencia
│   ├── hola.png
│   ├── adios.png
│   ├── paz.png
│   ├── si.png
│   └── te quiero.png
├── sequences/               # Secuencias de entrenamiento
│   ├── Hola_*.npy
│   ├── Adios_*.npy
│   └── ...
└── README.md               # Este archivo
```

## 🎮 Uso de la Aplicación

### Pantalla Principal
1. **Practicar:** Modo de aprendizaje libre
2. **Jugar:** Modo de evaluación con puntuación

### Modo Práctica
1. Selecciona un gesto del menú desplegable
2. Observa la imagen de referencia
3. Realiza el gesto frente a la cámara
4. Recibe retroalimentación inmediata:
   - 🟢 **CORRECTO:** Gesto reconocido correctamente
   - 🔴 **INCORRECTO:** Gesto no coincide
   - ⚪ **NO DETECTADO:** No se detectan manos

### Consejos para mejor reconocimiento:
- Mantén buena iluminación
- Usa un fondo simple
- Coloca las manos claramente visibles
- Mantén el gesto por 2-3 segundos
- Asegúrate de que la cámara capture ambas manos si es necesario

## 🧠 Entrenamiento del Modelo

### Arquitectura del Modelo LSTM
- **Capa de entrada:** 126 características (21 puntos × 3 coordenadas × 2 manos)
- **Secuencia temporal:** 60 frames (2 segundos a 30 FPS)
- **Capas LSTM:** 2 capas con 64 unidades cada una
- **Capa densa:** 64 neuronas con activación ReLU
- **Salida:** 5 clases (gestos) con activación softmax

### Proceso de entrenamiento:
1. **Captura de datos:** Usar `Capture_Gesture.py`
2. **Entrenamiento:** Ejecutar `train_gesture_model.py`
3. **Validación:** El modelo se evalúa automáticamente
4. **Guardado:** Se generan `modelo_gestos_lstm.keras` y `labels_lstm.npy`

