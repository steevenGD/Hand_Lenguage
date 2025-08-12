import mediapipe as mp
import numpy as np
from collections import deque
import math

class ASLGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        
        # Historial simplificado
        self.hand_history = deque(maxlen=30)
        self.gesture_history = deque(maxlen=8)
        
        # Umbrales ajustados
        self.movement_threshold = 0.005  # Más sensible
        self.stability_frames = 2  # Menos frames para estabilizar

        # Contador para gestos estáticos
        self.static_gesture_count = 0
        self.last_static_gesture = None
        
    def detectar_gesto_asl(self, manos_landmarks):
        """
        Detecta gestos ASL usando una o dos manos.
        manos_landmarks: lista de objetos hand_landmarks (puede ser 1 o 2)
        """
        try:
            # Si no hay manos, retorna None
            if not manos_landmarks:
                return None
            # Si hay solo una mano, usa la lógica original
            if len(manos_landmarks) == 1:
                hand_landmarks = manos_landmarks[0]
                current_position = self._get_hand_position(hand_landmarks)
                self.hand_history.append(current_position)
                static_gesture = self._detect_static_gesture(hand_landmarks)
                if static_gesture == self.last_static_gesture:
                    self.static_gesture_count += 1
                else:
                    self.static_gesture_count = 1
                    self.last_static_gesture = static_gesture
                # Priorizar gestos en movimiento si hay movimiento
                if len(self.hand_history) >= 5:
                    movement_pattern = self._analyze_movement()
                    final_gesture = self._combine_gesture_and_movement(
                        static_gesture, movement_pattern
                    )
                    if final_gesture:
                        self.gesture_history.append(final_gesture)
                        return self._get_stable_gesture()
                    # Si no hay gesto en movimiento, intentar gesto estático
                    if self.static_gesture_count >= 5:
                        static_only_gesture = self._detect_static_only_gesture(static_gesture)
                        if static_only_gesture:
                            return static_only_gesture
                    return None
                else:
                    # Si no hay suficiente historial para movimiento, solo gesto estático
                    if self.static_gesture_count >= 5:
                        static_only_gesture = self._detect_static_only_gesture(static_gesture)
                        if static_only_gesture:
                            return static_only_gesture
                    return None
            # Si hay dos manos, ejemplo de gesto con dos manos
            elif len(manos_landmarks) == 2:
                # Aquí puedes agregar gestos de dos manos si lo necesitas
                # Por ahora, no se reconoce ningún gesto de dos manos
                return None
            else:
                # Más de dos manos no soportado
                return None
        except Exception as e:
            print(f"Error detectando gesto ASL: {e}")
            return None
    
    def _get_hand_position(self, hand_landmarks):
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        return {
            'center_x': (wrist.x + index_tip.x + middle_tip.x) / 3,
            'center_y': (wrist.y + index_tip.y + middle_tip.y) / 3
        }
    
    def _detect_static_gesture(self, hand_landmarks):
        # Puntos clave
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]

        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]

        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]

        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]

        # Detección de dedos extendidos
        thumb_up = abs(thumb_tip.x - thumb_ip.x) > 0.02
        index_up = index_tip.y < index_pip.y
        middle_up = middle_tip.y < middle_pip.y
        ring_up = ring_tip.y < ring_pip.y
        pinky_up = pinky_tip.y < pinky_pip.y

        fingers = [thumb_up, index_up, middle_up, ring_up, pinky_up]
        total = sum(fingers)

        # ADIOS: Mano abierta (5 dedos) o mano abierta y luego cerrando dedos (detectado por movimiento)
        if total == 5:
            return "open_palm"
        elif total == 0:
            return "closed_fist"
        elif fingers == [True, True, True, False, False]:
            return "three_fingers"
        elif fingers == [False, True, True, False, False]:
            return "peace_sign"
        elif fingers == [True, True, False, False, True]:
            return "love_sign"
        return "unknown"
    
    def _analyze_movement(self):
        """Analiza movimiento para gestos dinámicos, incluyendo 'Adios' (Good-bye) como movimiento de cierre y apertura de dedos."""
        if len(self.hand_history) < 6:
            return "static"

        recent = list(self.hand_history)[-6:]
        movements = []

        for i in range(1, len(recent)):
            dx = recent[i]['center_x'] - recent[i-1]['center_x']
            dy = recent[i]['center_y'] - recent[i-1]['center_y']
            distance = math.sqrt(dx**2 + dy**2)
            if distance > self.movement_threshold:
                movements.append({'dx': dx, 'dy': dy, 'distance': distance})

        if len(movements) < 4:
            return "static"

        # Detectar movimiento de cierre y apertura de dedos (Adios)
        # Si la mano pasa de abierta a cerrada y abierta varias veces
        open_count = 0
        closed_count = 0
        for pos in recent:
            # Simulación: si el centro_y sube y baja, es como abrir/cerrar
            if pos['center_y'] < 0.65:
                open_count += 1
            else:
                closed_count += 1
        if open_count >= 2 and closed_count >= 2:
            return "goodbye_wave"

        # Detectar solo movimiento de saludo (lateral)
        horizontal_moves = [m for m in movements if abs(m['dx']) > abs(m['dy'])]
        if len(horizontal_moves) >= len(movements) * 0.5:
            direction_changes = 0
            last_direction = None
            for m in horizontal_moves:
                current_direction = "right" if m['dx'] > 0 else "left"
                if last_direction and current_direction != last_direction:
                    direction_changes += 1
                last_direction = current_direction
            if direction_changes >= 1:
                return "wave"
        return "static"
    
    def _combine_gesture_and_movement(self, static_gesture, movement_pattern):
        if not static_gesture:
            return None
        # Hola: mano abierta + movimiento lateral o estático
        if static_gesture == "open_palm":
            if movement_pattern == "wave" or movement_pattern == "static":
                return "Hola"
            if movement_pattern == "goodbye_wave":
                return "Adios"
        # No: tres dedos arriba
        if static_gesture == "three_fingers":
            return "No"
        # Si: puño cerrado
        if static_gesture == "closed_fist":
            return "Si"
        # Paz: signo de paz
        if static_gesture == "peace_sign":
            return "Paz"
        # Te quiero: I Love You
        if static_gesture == "love_sign":
            return "Te quiero"
        return None
    
    def _detect_static_only_gesture(self, static_gesture):
        static_gestures = {
            "closed_fist": "Si",
            "three_fingers": "No",
            "open_palm": "Hola",
            "peace_sign": "Paz",
            "love_sign": "Te quiero"
        }
        return static_gestures.get(static_gesture)
    
    def _get_stable_gesture(self):
        if len(self.gesture_history) < self.stability_frames:
            return None
        
        recent_gestures = list(self.gesture_history)[-self.stability_frames:]
        
        if not recent_gestures:
            return None
        
        # Contar frecuencias
        counts = {}
        for gesture in recent_gestures:
            if gesture:
                counts[gesture] = counts.get(gesture, 0) + 1
        
        if not counts:
            return None
        
        most_common = max(counts.items(), key=lambda x: x[1])
        
        # Solo requiere 2 detecciones para ser más rápido
        if most_common[1] >= 2:
            return most_common[0]
        
        return None

    def obtener_descripcion_gesto(self, gesto):
        """Descripciones de los gestos que funcionan"""
        descripciones = {
            "Hola": "Mano abierta con todos los dedos arriba (con o sin movimiento)",
            "Si": "Puño cerrado - todos los dedos abajo",
            "No": "Tres dedos arriba: pulgar + índice + medio",
            "Paz": "Índice y medio arriba (signo de paz/victoria)",
            "Te quiero": "Pulgar + índice + meñique arriba (I Love You en ASL)",
            "Adios": "Mano abierta, moviendo los dedos como si estuvieras diciendo adiós"
        }
        return descripciones.get(gesto, "Gesto no reconocido")

    def obtener_gestos_disponibles(self):
        return ["Hola", "Si", "No", "Paz", "Te quiero", "Adios"]
    
    def reset_history(self):
        """Reinicia el historial para empezar limpio"""
        self.hand_history.clear()
        self.gesture_history.clear()
        self.static_gesture_count = 0
        self.last_static_gesture = None