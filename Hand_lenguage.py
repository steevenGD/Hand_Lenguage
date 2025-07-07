import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
import random
import time
import threading
from PIL import Image, ImageTk
import queue

class HandLanguageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lenguaje de Señas")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')
        
        # Variables del juego
        self.gestos_disponibles = ["Paz", "OK", "Pulgar Arriba", "Puño"]
        self.gesto_actual = random.choice(self.gestos_disponibles)
        self.puntuacion = 0
        self.tiempo_inicio = time.time()
        self.tiempo_limite = 5
        self.gesto_reconocido = False
        self.juego_activo = False
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = None
        
        # Variables para evitar parpadeo
        self.frame_queue = queue.Queue(maxsize=2)
        self.gesto_detectado_actual = None
        self.update_video_job = None
        
        self.crear_interfaz()
        
    def crear_interfaz(self):
        # Título principal
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.pack(pady=10)
        
        tk.Label(title_frame, text="Lenguaje de Señas", 
                font=('Arial', 20, 'bold'), fg='#ecf0f1', bg='#2c3e50').pack()
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Frame izquierdo - Video
        video_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        video_frame.pack(side='left', expand=True, fill='both', padx=(0, 10))
        
        tk.Label(video_frame, text="Cámara", font=('Arial', 14, 'bold'), 
                fg='#ecf0f1', bg='#34495e').pack(pady=5)
        
        self.video_label = tk.Label(video_frame, bg='#34495e', text="Presiona 'Iniciar Juego'", 
                                   font=('Arial', 16), fg='#ecf0f1')
        self.video_label.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Frame derecho - Controles
        control_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        control_frame.pack(side='right', fill='y', padx=(10, 0))
        control_frame.configure(width=300)
        
        # Información del juego
        info_frame = tk.Frame(control_frame, bg='#34495e')
        info_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(info_frame, text="Información del Juego", 
                font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#34495e').pack(pady=5)
        
        # Gesto objetivo
        self.gesto_label = tk.Label(info_frame, text=f"Gesto: {self.gesto_actual}", 
                                   font=('Arial', 16, 'bold'), fg='#e74c3c', bg='#34495e')
        self.gesto_label.pack(pady=5)
        
        # Emoji del gesto
        self.emoji_label = tk.Label(info_frame, text="✌️", font=('Arial', 40), 
                                   fg='#f39c12', bg='#34495e')
        self.emoji_label.pack(pady=10)
        
        # Tiempo restante
        self.tiempo_label = tk.Label(info_frame, text="Tiempo: 5.0s", 
                                    font=('Arial', 14), fg='#27ae60', bg='#34495e')
        self.tiempo_label.pack(pady=5)
        
        # Puntuación
        self.puntuacion_label = tk.Label(info_frame, text="Puntuación: 0", 
                                        font=('Arial', 14, 'bold'), fg='#3498db', bg='#34495e')
        self.puntuacion_label.pack(pady=5)
        
        # Estado del gesto
        self.estado_label = tk.Label(info_frame, text="🔍 Buscando gesto...", 
                                    font=('Arial', 12), fg='#95a5a6', bg='#34495e')
        self.estado_label.pack(pady=10)
        
        # Botones
        button_frame = tk.Frame(control_frame, bg='#34495e')
        button_frame.pack(fill='x', padx=10, pady=20)
        
        self.iniciar_btn = tk.Button(button_frame, text="Iniciar Juego", 
                                    command=self.iniciar_juego, font=('Arial', 12, 'bold'),
                                    bg='#27ae60', fg='white', relief='raised', bd=2)
        self.iniciar_btn.pack(fill='x', pady=5)
        
        self.pausar_btn = tk.Button(button_frame, text="Pausar", 
                                   command=self.pausar_juego, font=('Arial', 12, 'bold'),
                                   bg='#f39c12', fg='white', relief='raised', bd=2, state='disabled')
        self.pausar_btn.pack(fill='x', pady=5)
        
        self.reiniciar_btn = tk.Button(button_frame, text="Reiniciar", 
                                      command=self.reiniciar_juego, font=('Arial', 12, 'bold'),
                                      bg='#e74c3c', fg='white', relief='raised', bd=2)
        self.reiniciar_btn.pack(fill='x', pady=5)
        
        # Instrucciones
        instrucciones_frame = tk.Frame(control_frame, bg='#34495e')
        instrucciones_frame.pack(fill='x', padx=10, pady=10)
        
       

        
        # Información adicional
        info_extra_frame = tk.Frame(control_frame, bg='#34495e')
        info_extra_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(info_extra_frame, text="Consejos:", 
                font=('Arial', 10, 'bold'), fg='#ecf0f1', bg='#34495e').pack()
        
        consejos = [
            "• Mantén la mano en el centro",
            "• Asegúrate de tener buena luz",
            "• Haz gestos claros y definidos"
        ]
        
        for consejo in consejos:
            tk.Label(info_extra_frame, text=consejo, font=('Arial', 8), 
                    fg='#bdc3c7', bg='#34495e', anchor='w').pack(fill='x', pady=1)
    
    def detectar_gesto(self, hand_landmarks):
        """Detecta el gesto basado en las coordenadas de la mano"""
        try:
            # Obtener coordenadas de puntos necesarios
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
            pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
            pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
            
            # Gesto "Paz" (índice y medio levantados, resto abajo)
            if (index_tip.y < index_pip.y and
                middle_tip.y < middle_pip.y and
                ring_tip.y > ring_pip.y and
                pinky_tip.y > pinky_pip.y):
                return "Paz"
            
            # Gesto "OK" (punta del pulgar e índice están cerca)
            elif abs(index_tip.x - thumb_tip.x) < 0.05 and abs(index_tip.y - thumb_tip.y) < 0.05:
                return "OK"
            
            # Gesto "Pulgar Arriba" (solo pulgar levantado)
            elif (thumb_tip.y < thumb_ip.y and
                  index_tip.y > index_pip.y and
                  middle_tip.y > middle_pip.y and
                  ring_tip.y > ring_pip.y and
                  pinky_tip.y > pinky_pip.y):
                return "Pulgar Arriba"
            
            # Gesto "Puño" (todos los dedos cerrados)
            elif (index_tip.y > index_pip.y and
                  middle_tip.y > middle_pip.y and
                  ring_tip.y > ring_pip.y and
                  pinky_tip.y > pinky_pip.y):
                return "Puño"
            
            return None
        except:
            return None
    
    def obtener_emoji(self, gesto):
        emojis = {
            "Paz": "✌️",
            "OK": "👌",
            "Pulgar Arriba": "👍",
            "Puño": "👊"
        }
        return emojis.get(gesto, "🤔")
    
    def iniciar_juego(self):
        self.juego_activo = True
        self.iniciar_btn.config(state='disabled')
        self.pausar_btn.config(state='normal')
        
        # Configurar cámara con parámetros optimizados
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la cámara")
            return
        
        # Configurar propiedades de la cámara para reducir parpadeo
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Iniciar thread para el video
        self.video_thread = threading.Thread(target=self.capturar_video)
        self.video_thread.daemon = True
        self.video_thread.start()
        
        # Iniciar actualización de video en la GUI
        self.actualizar_video()
        
        # Iniciar actualización de interfaz
        self.actualizar_interfaz()
    
    def capturar_video(self):
        """Captura video en un thread separado"""
        with self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1,
            static_image_mode=False
        ) as hands:
            
            while self.juego_activo and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Procesar frame
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                # Dibujar landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                        
                        # Detectar gesto
                        gesto = self.detectar_gesto(hand_landmarks)
                        if gesto:
                            self.gesto_detectado_actual = gesto
                            
                            # Verificar si es correcto
                            if gesto == self.gesto_actual and not self.gesto_reconocido:
                                self.puntuacion += 10
                                self.gesto_reconocido = True
                                
                                # Mostrar feedback en el frame
                                cv2.putText(frame, "¡CORRECTO! +10", (50, 50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    self.gesto_detectado_actual = None
                
                # Añadir frame a la cola (sin bloquear)
                if not self.frame_queue.full():
                    try:
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass
                
                # Controlar FPS
                time.sleep(1/30)  # ~30 FPS
    
    def actualizar_video(self):
        """Actualiza el video en la GUI"""
        if self.juego_activo:
            try:
                # Obtener frame más reciente
                frame = None
                while not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                
                if frame is not None:
                    # Redimensionar frame
                    frame_resized = cv2.resize(frame, (640, 480))
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    
                    # Convertir a ImageTk
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    # Actualizar label
                    self.video_label.configure(image=imgtk, text="")
                    self.video_label.image = imgtk  # Mantener referencia
                    
                    # Actualizar estado del gesto
                    if self.gesto_detectado_actual:
                        color = "#27ae60" if self.gesto_detectado_actual == self.gesto_actual else "#e74c3c"
                        self.estado_label.config(text=f"🎯 {self.gesto_detectado_actual}", fg=color)
                    else:
                        self.estado_label.config(text="🔍 Buscando gesto...", fg="#95a5a6")
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error actualizando video: {e}")
            
            # Programar siguiente actualización
            self.update_video_job = self.root.after(33, self.actualizar_video)  # ~30 FPS
    
    def pausar_juego(self):
        self.juego_activo = False
        self.iniciar_btn.config(state='normal')
        self.pausar_btn.config(state='disabled')
        
        # Cancelar actualización de video
        if self.update_video_job:
            self.root.after_cancel(self.update_video_job)
        
        # Liberar cámara
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Limpiar cola
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        # Mostrar mensaje en video
        self.video_label.configure(image="", text="Juego pausado\nPresiona 'Iniciar' para continuar", 
                                  font=('Arial', 16), fg='#ecf0f1')
    
    def reiniciar_juego(self):
        self.puntuacion = 0
        self.gesto_actual = random.choice(self.gestos_disponibles)
        self.tiempo_inicio = time.time()
        self.gesto_reconocido = False
        self.gesto_detectado_actual = None
        self.actualizar_labels()
    
    def actualizar_interfaz(self):
        if self.juego_activo:
            # Calcular tiempo restante
            tiempo_transcurrido = time.time() - self.tiempo_inicio
            tiempo_restante = max(0, self.tiempo_limite - tiempo_transcurrido)
            
            # Verificar si se acabó el tiempo
            if tiempo_restante <= 0:
                if not self.gesto_reconocido:
                    self.puntuacion = max(0, self.puntuacion - 5)
                
                # Cambiar a nuevo gesto
                self.gesto_actual = random.choice(self.gestos_disponibles)
                self.tiempo_inicio = time.time()
                self.gesto_reconocido = False
                self.gesto_detectado_actual = None
            
            self.actualizar_labels()
            
            # Continuar actualizando
            self.root.after(100, self.actualizar_interfaz)
    
    def actualizar_labels(self):
        tiempo_transcurrido = time.time() - self.tiempo_inicio
        tiempo_restante = max(0, self.tiempo_limite - tiempo_transcurrido)
        
        self.gesto_label.config(text=f"Gesto: {self.gesto_actual}")
        self.emoji_label.config(text=self.obtener_emoji(self.gesto_actual))
        self.tiempo_label.config(text=f"Tiempo: {tiempo_restante:.1f}s")
        self.puntuacion_label.config(text=f"Puntuación: {self.puntuacion}")
        
        # Cambiar color del tiempo
        if tiempo_restante > 2:
            self.tiempo_label.config(fg='#27ae60')
        elif tiempo_restante > 1:
            self.tiempo_label.config(fg='#f39c12')
        else:
            self.tiempo_label.config(fg='#e74c3c')
    
    def cerrar_aplicacion(self):
        self.juego_activo = False
        
        # Cancelar trabajos pendientes
        if self.update_video_job:
            self.root.after_cancel(self.update_video_job)
        
        # Liberar recursos
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandLanguageGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.cerrar_aplicacion)
    root.mainloop()