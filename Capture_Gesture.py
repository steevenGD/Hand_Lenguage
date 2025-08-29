import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# Configura los gestos que quieres capturar
GESTURES = ["Hola", "Adios", "Paz", "Te quiero", "Por favor", "Feliz", "Jugar", "Beber", "Amigo", "Familia", "Volar", 
            "Comer", "Ayuda", "Gracias", "Lo siento", "Tiempo", "Casa", "Escuela", "Trabajar", "Nombre"]
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

def show_gesture_menu():
    print("\n=== MENÚ DE GESTOS ===")
    for idx, gesto in enumerate(GESTURES):
        print(f"  {idx}: {gesto}")
    print("======================")
    print("Instrucciones:")
    print("- Presiona 'm' para mostrar este menú")
    print("- Presiona el número del gesto para seleccionarlo")
    print("- Presiona 's' para iniciar la grabación de una secuencia")
    print("- Presiona 'q' para salir")
    print("======================")

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
    selected_label = None
    sequence = []
    recording_sequence = False
    sequence_label = None
    countdown_start = 0
    input_buffer = ""
    
    # Mostrar el menú al inicio
    show_gesture_menu()

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
        
        # Mostrar información en la ventana
        if selected_label:
            cv2.putText(frame, f"Gesto: {selected_label}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if recording_sequence:
            elapsed = time.time() - countdown_start
            if elapsed < 0.5:  # Medio segundo de retraso
                remaining = 0.5 - elapsed
                cv2.putText(frame, f"Iniciando en: {remaining:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Grabando... {len(sequence)}/{SEQUENCE_LENGTH}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar buffer de entrada si está activo
        if input_buffer:
            cv2.putText(frame, f"Seleccion: {input_buffer}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Recolector de Gestos', frame)

        key = cv2.waitKey(1) & 0xFF  # Solo tomar los últimos 8 bits
        
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_gesture_menu()
            input_buffer = ""  # Limpiar buffer al mostrar menú
        
        # Manejar entrada numérica para selección de gestos
        elif 48 <= key <= 57:  # Teclas del 0 al 9
            input_buffer += chr(key)
            print(f"Entrada: {input_buffer}")
            
            # Si el buffer tiene suficiente longitud, procesar la selección
            if len(input_buffer) >= len(str(len(GESTURES) - 1)):
                try:
                    idx = int(input_buffer)
                    if 0 <= idx < len(GESTURES):
                        selected_label = GESTURES[idx]
                        print(f"Gesto seleccionado: {selected_label} (índice {idx})")
                        input_buffer = ""  # Limpiar buffer después de una selección válida
                    else:
                        print(f"Índice {idx} fuera de rango. Debe estar entre 0 y {len(GESTURES)-1}.")
                        input_buffer = ""  # Limpiar buffer después de un error
                except ValueError:
                    print("Entrada no válida.")
                    input_buffer = ""  # Limpiar buffer después de un error
        
        # Tecla Enter para confirmar selección si el buffer no está vacío
        elif key == 13 and input_buffer:  # Código ASCII para Enter
            try:
                idx = int(input_buffer)
                if 0 <= idx < len(GESTURES):
                    selected_label = GESTURES[idx]
                    print(f"Gesto seleccionado: {selected_label} (índice {idx})")
                else:
                    print(f"Índice {idx} fuera de rango. Debe estar entre 0 y {len(GESTURES)-1}.")
            except ValueError:
                print("Entrada no válida.")
            input_buffer = ""  # Limpiar buffer después de procesar
        
        # Tecla Escape para cancelar entrada
        elif key == 27:  # Código ASCII para Escape
            input_buffer = ""
            print("Selección cancelada")

        # Grabar secuencia
        if key == ord('s') and selected_label and not recording_sequence:
            print(f"Preparando para grabar secuencia: {selected_label}")
            print("¡Prepárate! La grabación comenzará en 0.5 segundos...")
            recording_sequence = True
            sequence_label = selected_label
            countdown_start = time.time()
            sequence = []

        # Guardar frames de la secuencia (después del retraso)
        if recording_sequence:
            elapsed = time.time() - countdown_start
            if elapsed >= 0.5:  # Esperar medio segundo antes de empezar a grabar
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
                    save_path = os.path.join(SEQUENCE_DIR, f"{sequence_label}_{int(time.time()*1000)}.npy")
                    np.save(save_path, sequence_np)
                    print(f"Secuencia guardada en {save_path}")
                    recording_sequence = False
                    sequence = []
                    sequence_label = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()