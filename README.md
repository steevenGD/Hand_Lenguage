# 🧠 Intérprete de Lenguaje de Señas en Tiempo Real

Este proyecto es un sistema básico de interpretación de lenguaje de señas utilizando **MediaPipe**, **OpenCV** y Python. Captura gestos de la mano desde una cámara en tiempo real y los traduce a texto en pantalla.

---

## 📌 Características

- ✅ Detección en tiempo real de la mano con MediaPipe
- ✅ Reconocimiento de gestos manuales simples como:
  - ✌️ Paz
  - 👌 OK
  - Añadir más XD
- ✅ Visualización de texto traducido en la pantalla
- 🔄 Posibilidad de extenderlo a otros gestos (letras, palabras)
- 🧠 Código modular para futuras integraciones con modelos de Machine Learning

---

## 🚀 Requisitos

- Python 3.10 ✅ (**MediaPipe no es compatible con 3.11+**)
  - https://www.python.org/downloads/release/python-3100/
- pip

---

## 📦 Instalación

### 1. Clonar el repositorio:

```bash
git clone https://github.com/steevenGD/Hand_Lenguage.git
```
### 2. Crear entorno virtual (OPCIONAL):
```bash
python -m venv venv
venv\Scripts\activate     # En Windows
```
### 3. Instalar dependencias:
```bash
pip install mediapipe opencv-python
```
### 4. Comando dentro del entorno virtual (OPCIONAL):
```bash
python Hand_lenguage.py
```


