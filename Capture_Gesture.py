import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Configura los gestos que quieres capturar
GESTURES = ["Hola", "Adios", "Paz", "Te quiero", "Por favor", "Feliz", "Jugar", "Beber", "Amigo", "Familia"]
OUTPUT_FILE = "gestures_data.csv"
SEQUENCE_LENGTH = 60  # Número de frames por secuencia
SEQUENCE_DIR = "sequences"  # Carpeta para guardar secuencias

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Crear directorio para secuencias si no existe
os.makedirs(SEQUENCE_DIR, exist_ok=True)

def save_landmarks(landmarks, handedness, label):
    row = []
    hand_types = []
    for idx, hand in enumerate(landmarks):
        for lm in hand.landmark:
            row.extend([lm.x, lm.y, lm.z])
        # Guardar tipo de mano (Right/Left) si está disponible
        if handedness and len(handedness) > idx:
            hand_types.append(handedness[idx].classification[0].label)
        else:
            hand_types.append('Unknown')
    # Si solo hay una mano, rellena con ceros para la segunda
    if len(landmarks) == 1:
        row.extend([0.0] * (21 * 3))
        hand_types.append('None')
    # Añadir tipo de mano para ambas manos
    row.extend(hand_types)
    row.append(label)
    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def main():
    if not os.path.exists(OUTPUT_FILE):
        # Escribir encabezado
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            header = []
            for i in range(2):  # hasta 2 manos
                for j in range(21):
                    header.extend([f'x{i}_{j}', f'y{i}_{j}', f'z{i}_{j}'])
            header.extend(['hand0_type', 'hand1_type'])
            header.append('label')
            writer.writerow(header)

    cap = cv2.VideoCapture(0)
    print("Gestos disponibles:")
    for idx, gesto in enumerate(GESTURES):
        print(f"  {idx}: {gesto}")
    print("Presiona la tecla del número del gesto (0=Hola, 1=Adios, ...) para seleccionar el gesto.")
    print("Presiona 's' para iniciar la grabación de una secuencia (por defecto, 60 frames) para modelos LSTM.")
    print("Presiona 'q' para salir.")

    selected_label = None
    sequence = []
    recording_sequence = False
    sequence_label = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Recolector de Gestos', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        # Selección de gesto (teclas 0-9)
        if key in [ord(str(i)) for i in range(min(len(GESTURES), 10))]:
            idx = int(chr(key))
            selected_label = GESTURES[idx]
            print(f"Gesto seleccionado: {selected_label}")

        # Grabar secuencia
        if key == ord('s') and selected_label and not recording_sequence:
            print(f"Grabando secuencia para: {selected_label} ({SEQUENCE_LENGTH} frames)")
            sequence = []
            recording_sequence = True
            sequence_label = selected_label

        # Guardar frames de la secuencia
        if recording_sequence:
            if results.multi_hand_landmarks:
                row = []
                for idx, hand in enumerate(results.multi_hand_landmarks):
                    for lm in hand.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                if len(results.multi_hand_landmarks) == 1:
                    row.extend([0.0] * (21 * 3))
                while len(row) < 21 * 3 * 2:
                    row.append(0.0)
                row = row[:21 * 3 * 2]
                sequence.append(row)
            else:
                sequence.append([0.0] * (21 * 3 * 2))
            if len(sequence) >= SEQUENCE_LENGTH:
                sequence_np = np.array(sequence)
                save_path = os.path.join(SEQUENCE_DIR, f"{sequence_label}_{int(np.random.rand()*1e6)}.npy")
                np.save(save_path, sequence_np)
                print(f"Secuencia guardada en {save_path}")
                recording_sequence = False
                sequence = []
                sequence_label = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()