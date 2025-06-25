import cv2
import mediapipe as mp
import csv
import os

# Configuración
GESTURE_NAME = input("Ingrese el nombre del gesto que desea capturar: ").strip().upper()
NUM_SAMPLES = 30  # cantidad de capturas que quieres hacer

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Crear archivo si no existe
file_exists = os.path.isfile('training_data.csv')
with open('training_data.csv', mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        header = ['label']
        for i in range(21):
            header += [f'x{i}', f'y{i}', f'z{i}']
        writer.writerow(header)

# Captura
cap = cv2.VideoCapture(0)
count = 0
while cap.isOpened() and count < NUM_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            row = [GESTURE_NAME]
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            with open('training_data.csv', mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            count += 1
            print(f"[{count}/{NUM_SAMPLES}] Capturado gesto: {GESTURE_NAME}")

    cv2.putText(frame, f'Capturando: {GESTURE_NAME} [{count}/{NUM_SAMPLES}]', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow('Captura de Gestos', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
