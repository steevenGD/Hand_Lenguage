import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
import random
import time
import threading
from PIL import Image, ImageTk
import queue
import numpy as np
import tensorflow as tf

class HandLanguageGUI:
    def __init__(self, root):
        import tensorflow as tf
        import numpy as np
        self.modelo_lstm = tf.keras.models.load_model('modelo_gestos_lstm.keras')
        self.labels_lstm = np.load('labels_lstm.npy')
        self.sequence_length = 60
        self.landmark_buffer = []

        self.root = root
        self.root.title("Lenguaje de Señas ASL")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')

        # Variables del juego
        self.gesto_actual = None
        self.puntuacion = 0
        self.tiempo_inicio = time.time()
        self.tiempo_limite = 8 
        self.gesto_reconocido = False
        self.juego_activo = False

        # Variables modo práctica
        self.practica_activa = False
        self.gesto_practica = None

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = None

        # Variables para evitar parpadeo
        self.frame_queue = queue.Queue(maxsize=2)
        self.gesto_detectado_actual = None
        self.update_video_job = None

        self.crear_pantalla_inicio()

    def crear_pantalla_inicio(self):
        self.limpiar_ventana()
        frame_inicio = tk.Frame(self.root, bg='#2c3e50')
        frame_inicio.pack(expand=True)
        tk.Label(frame_inicio, text="Bienvenido a Lenguaje de Señas ASL", font=('Arial', 22, 'bold'), fg='#ecf0f1', bg='#2c3e50').pack(pady=30)
        btn_jugar = tk.Button(frame_inicio, text="Jugar", font=('Arial', 16, 'bold'), bg='#27ae60', fg='white', width=15, command=self.iniciar_juego)
        btn_jugar.pack(pady=15)
        btn_practicar = tk.Button(frame_inicio, text="Practicar", font=('Arial', 16, 'bold'), bg='#3498db', fg='white', width=15, command=self.iniciar_practica)
        btn_practicar.pack(pady=15)

    def limpiar_ventana(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def iniciar_practica(self):
        self.practica_activa = True
        self.limpiar_ventana()
        frame_practica = tk.Frame(self.root, bg='#2c3e50')
        frame_practica.pack(expand=True, fill='both')

        tk.Label(frame_practica, text="Modo Práctica", font=('Arial', 20, 'bold'), fg='#ecf0f1', bg='#2c3e50').pack(pady=10)

        # Selección de palabra antes de mostrar práctica
        select_frame = tk.Frame(frame_practica, bg='#2c3e50')
        select_frame.pack(pady=10)
        tk.Label(select_frame, text="Selecciona una palabra para practicar:", font=('Arial', 14, 'bold'), fg='#ecf0f1', bg='#2c3e50').pack(side='left', padx=5)
        self.combo_palabras = ttk.Combobox(select_frame, values=list(self.labels_lstm), font=('Arial', 12), state='readonly')
        self.combo_palabras.pack(side='left', padx=5)
        self.combo_palabras.bind("<<ComboboxSelected>>", self.seleccionar_gesto_practica)

        # Frame principal de práctica (imagen y cámara)
        self.frame_practica_main = tk.Frame(frame_practica, bg='#2c3e50')
        self.frame_practica_main.pack(expand=True, fill='both', padx=20, pady=10)

        # Palabra seleccionada arriba centrada
        self.label_palabra_practica = tk.Label(self.frame_practica_main, text="", font=('Arial', 22, 'bold'), fg='#e74c3c', bg='#2c3e50')
        self.label_palabra_practica.pack(pady=10)

        # Contenedor horizontal para imagen y cámara
        contenedor = tk.Frame(self.frame_practica_main, bg='#2c3e50')
        contenedor.pack(expand=True, fill='both')

        # Imagen del gesto (de momento en blanco)
        img_frame = tk.Frame(contenedor, bg='#34495e', relief='raised', bd=2)
        img_frame.pack(side='left', expand=True, fill='both', padx=(0, 10), ipadx=10, ipady=10)
        self.img_gesto_label = tk.Label(img_frame, bg='#ecf0f1')
        self.img_gesto_label.pack(expand=True, fill='both', padx=10, pady=10)

        # Cámara para practicar (más grande)
        cam_frame = tk.Frame(contenedor, bg='#34495e', relief='raised', bd=2)
        cam_frame.pack(side='right', expand=True, fill='both', padx=(10, 0), ipadx=10, ipady=10)
        self.video_label_practica = tk.Label(cam_frame, bg='#34495e', text="Selecciona una palabra", font=('Arial', 16), fg='#ecf0f1')
        self.video_label_practica.pack(expand=True, fill='both', padx=10, pady=10)

        # Botón para volver
        btn_volver = tk.Button(self.frame_practica_main, text="Volver", font=('Arial', 12, 'bold'), bg='#e74c3c', fg='white', command=self.crear_pantalla_inicio)
        btn_volver.pack(pady=10)

        # Iniciar cámara para práctica
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la cámara")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.video_thread_practica = threading.Thread(target=self.capturar_video_practica)
        self.video_thread_practica.daemon = True
        self.video_thread_practica.start()
        self.actualizar_video_practica()

    def seleccionar_gesto_practica(self, event):
        self.gesto_practica = self.combo_palabras.get()
        self.label_palabra_practica.config(text=f"{self.gesto_practica}")
        # Cargar la imagen correspondiente al gesto seleccionado
        try:
            img_path = f"imagenes_gestos/{self.gesto_practica.lower()}.png"
            img = Image.open(img_path).resize((640, 480))
        except Exception:
            # Si no existe la imagen, muestra una en blanco
            img = Image.new('RGB', (640, 480), color='white')
        imgtk = ImageTk.PhotoImage(image=img)
        self.img_gesto_label.configure(image=imgtk)
        self.img_gesto_label.image = imgtk

    def capturar_video_practica(self):
        feedback_timer = None
        feedback_text = None
        feedback_color = (0, 255, 0)
        last_feedback = None
        
        feedback_duration = 2.0  # Segundos que se muestra el feedback
        with self.mp_hands.Hands(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            max_num_hands=2,
            static_image_mode=False
        ) as hands:
            while self.practica_activa and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                manos_landmarks = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        manos_landmarks.append(hand_landmarks)
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                now = time.time()
                # Solo predecir si no estamos mostrando feedback
                if not (feedback_text == "CORRECTO" and feedback_timer and (now - feedback_timer < feedback_duration)):
                    gesto_detectado = self.detectar_gesto_lstm(manos_landmarks)
                    print(f"[DEBUG] Gesto detectado: {gesto_detectado}")
                    nuevo_feedback = None
                    nuevo_color = None
                    if self.gesto_practica:
                        if gesto_detectado == self.gesto_practica:
                            nuevo_feedback = "CORRECTO"
                            nuevo_color = (0, 255, 0)
                        elif gesto_detectado is None:
                            nuevo_feedback = "NO DETECTADO"
                            nuevo_color = (128, 128, 128)
                        else:
                            nuevo_feedback = "INCORRECTO"
                            nuevo_color = (0, 0, 255)
                    # Solo actualiza feedback y timer si el feedback cambia
                    if nuevo_feedback != last_feedback:
                        feedback_text = nuevo_feedback
                        feedback_color = nuevo_color
                        feedback_timer = now
                        last_feedback = nuevo_feedback
                # Mostrar feedback durante feedback_duration segundos
                if feedback_text and feedback_timer and (now - feedback_timer < feedback_duration):
                    cv2.putText(frame, feedback_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2)
                    if feedback_text == "CORRECTO" and (now - feedback_timer >= feedback_duration - 0.05):
                        self.landmark_buffer = []  # Limpiar buffer justo al terminar el feedback
                elif feedback_text and feedback_timer and (now - feedback_timer >= feedback_duration):
                    feedback_text = None
                    feedback_timer = None
                    last_feedback = None
                if not hasattr(self, 'frame_queue_practica'):
                    self.frame_queue_practica = queue.Queue(maxsize=2)
                if not self.frame_queue_practica.full():
                    try:
                        self.frame_queue_practica.put_nowait(frame)
                    except:
                        pass
                time.sleep(1/30)

    def actualizar_video_practica(self):
        if self.practica_activa:
            try:
                frame = None
                while hasattr(self, 'frame_queue_practica') and not self.frame_queue_practica.empty():
                    frame = self.frame_queue_practica.get_nowait()
                if frame is not None:
                    frame_resized = cv2.resize(frame, (640, 480))
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb).resize((640, 480))
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label_practica.configure(image=imgtk, text="")
                    self.video_label_practica.image = imgtk
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error actualizando video práctica: {e}")
            self.root.after(33, self.actualizar_video_practica)
        
    def crear_interfaz(self):
        # Título principal
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.pack(pady=10)
        
        tk.Label(title_frame, text="Lenguaje de Señas ASL", 
                font=('Arial', 20, 'bold'), fg='#ecf0f1', bg='#2c3e50').pack()
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Frame izquierdo - Video
        video_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        video_frame.pack(side='left', expand=True, fill='both', padx=(0, 10))
        
        tk.Label(video_frame, text="Camara", font=('Arial', 14, 'bold'), 
                fg='#ecf0f1', bg='#34495e').pack(pady=5)
        
        self.video_label = tk.Label(video_frame, bg='#34495e', text="Presiona 'Iniciar Juego'", 
                    font=('Arial', 16), fg='#ecf0f1')
        self.video_label.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Frame derecho - Controles
        control_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        control_frame.pack(side='right', fill='y', padx=(10, 0))
        control_frame.configure(width=350)
        
        # Información del juego
        info_frame = tk.Frame(control_frame, bg='#34495e')
        info_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(info_frame, text="Informacion del Juego", 
                font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#34495e').pack(pady=5)
        
        # Palabra objetivo
        self.gesto_label = tk.Label(info_frame, text=f"Palabra: {self.gesto_actual}", 
                                   font=('Arial', 18, 'bold'), fg='#e74c3c', bg='#34495e')
        self.gesto_label.pack(pady=5)
        
        # Descripción del gesto
        self.descripcion_label = tk.Label(info_frame, 
                                         text=self.obtener_descripcion_gesto(self.gesto_actual), 
                                         font=('Arial', 10), fg='#f39c12', bg='#34495e',
                                         wraplength=280, justify='center')
        self.descripcion_label.pack(pady=5)
        
        # Tiempo restante
        self.tiempo_label = tk.Label(info_frame, text="Tiempo: 8.0s", 
                                    font=('Arial', 14), fg='#27ae60', bg='#34495e')
        self.tiempo_label.pack(pady=5)
        
        # Puntuación
        self.puntuacion_label = tk.Label(info_frame, text="Puntuacion: 0", 
                                        font=('Arial', 14, 'bold'), fg='#3498db', bg='#34495e')
        self.puntuacion_label.pack(pady=5)
        
        # Estado del gesto
        self.estado_label = tk.Label(info_frame, text="Buscando gesto...", 
                                    font=('Arial', 11), fg='#95a5a6', bg='#34495e')
        self.estado_label.pack(pady=8)
        
        # Botones
        button_frame = tk.Frame(control_frame, bg='#34495e')
        button_frame.pack(fill='x', padx=10, pady=15)
        
        self.iniciar_btn = tk.Button(button_frame, text="Iniciar Juego", 
                                    command=self.iniciar_juego, font=('Arial', 12, 'bold'),
                                    bg='#27ae60', fg='white', relief='raised', bd=2)
        self.iniciar_btn.pack(fill='x', pady=3)
        
        self.pausar_btn = tk.Button(button_frame, text="Pausar", 
                                   command=self.pausar_juego, font=('Arial', 12, 'bold'),
                                   bg='#f39c12', fg='white', relief='raised', bd=2, state='disabled')
        self.pausar_btn.pack(fill='x', pady=3)
        
        self.reiniciar_btn = tk.Button(button_frame, text="Reiniciar", 
                                      command=self.reiniciar_juego, font=('Arial', 12, 'bold'),
                                      bg='#e74c3c', fg='white', relief='raised', bd=2)
        self.reiniciar_btn.pack(fill='x', pady=3)
        
        # Guía de gestos
        guia_frame = tk.Frame(control_frame, bg='#34495e')
        guia_frame.pack(fill='x', padx=10, pady=10)
        tk.Label(guia_frame, text="Palabras ASL Disponibles", 
                font=('Arial', 11, 'bold'), fg='#ecf0f1', bg='#34495e').pack()
        
        # Crear scrollable frame
        canvas = tk.Canvas(guia_frame, bg='#34495e', height=180, highlightthickness=0)
        scrollbar = ttk.Scrollbar(guia_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#34495e')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Lista de palabras ASL
        for palabra in self.labels_lstm:
            frame_palabra = tk.Frame(scrollable_frame, bg='#34495e')
            frame_palabra.pack(fill='x', pady=1)
            
            descripcion = self.obtener_descripcion_gesto(palabra)
            tk.Label(frame_palabra, text=f"{palabra}: {descripcion}", 
                    font=('Arial', 8), fg='#bdc3c7', bg='#34495e', 
                    anchor='w', wraplength=300).pack(fill='x')
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Consejos
        consejos_frame = tk.Frame(control_frame, bg='#34495e')
        consejos_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(consejos_frame, text="Consejos:", 
                font=('Arial', 10, 'bold'), fg='#ecf0f1', bg='#34495e').pack()
        
        consejos = [
            "• Mantén la mano centrada y visible",
            "• Haz gestos claros y definidos",
            "• Mantén la posición por 2-3 segundos",
            "• Buena iluminación es importante"
        ]
        
        for consejo in consejos:
            tk.Label(consejos_frame, text=consejo, font=('Arial', 8), 
                    fg='#bdc3c7', bg='#34495e', anchor='w').pack(fill='x', pady=1)
    
    def iniciar_juego(self):
        self.juego_activo = True
        self.practica_activa = False
        self.limpiar_ventana()
        self.crear_interfaz()
        self.iniciar_btn.config(state='disabled')
        self.pausar_btn.config(state='normal')
        # Elegir palabra objetivo al iniciar
        self.gesto_actual = random.choice(self.labels_lstm)
        self.tiempo_inicio = time.time()
        self.gesto_reconocido = False
        self.gesto_detectado_actual = None
        # Configurar cámara
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la cámara")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Iniciar threads
        self.video_thread = threading.Thread(target=self.capturar_video)
        self.video_thread.daemon = True
        self.video_thread.start()
        self.actualizar_video()
        self.actualizar_interfaz()
    
    def capturar_video(self):
        """Captura video y detecta gestos ASL (soporta gestos de dos manos)"""
        with self.mp_hands.Hands(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            max_num_hands=2,
            static_image_mode=False
        ) as hands:
            feedback_timer = None
            feedback_duration = 2.0  # segundos
            feedback_active = False
            while self.juego_activo and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                manos_landmarks = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        manos_landmarks.append(hand_landmarks)
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                now = time.time()
                # Si estamos mostrando feedback de correcto, solo mostrar el mensaje y esperar
                if feedback_active and feedback_timer and (now - feedback_timer < feedback_duration):
                    cv2.putText(frame, "¡CORRECTO! +20", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif feedback_active and feedback_timer and (now - feedback_timer >= feedback_duration):
                    # Termina feedback, pasa a siguiente palabra
                    self.gesto_actual = random.choice(self.labels_lstm)
                    self.tiempo_inicio = time.time()
                    self.gesto_reconocido = False
                    self.gesto_detectado_actual = None
                    feedback_active = False
                    feedback_timer = None
                else:
                    # Detectar gesto usando LSTM (igual que en práctica)
                    gesto = self.detectar_gesto_lstm(manos_landmarks)
                    if gesto:
                        self.gesto_detectado_actual = gesto
                        if gesto == self.gesto_actual and not self.gesto_reconocido:
                            self.puntuacion += 20
                            self.gesto_reconocido = True
                            feedback_active = True
                            feedback_timer = now
                            cv2.putText(frame, "¡CORRECTO! +20", (50, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            self.estado_label.config(text=f"Detectado: {self.gesto_detectado_actual}", fg="#27ae60")
                    else:
                        self.gesto_detectado_actual = None
                # Mostrar palabra objetivo
                cv2.putText(frame, f"Palabra: {self.gesto_actual}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if not self.frame_queue.full():
                    try:
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass
                time.sleep(1/30)
    
    def actualizar_video(self):
        """Actualiza el video en la GUI"""
        if self.juego_activo:
            try:
                frame = None
                while not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                
                if frame is not None:
                    frame_resized = cv2.resize(frame, (640, 480))
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    self.video_label.configure(image=imgtk, text="")
                    self.video_label.image = imgtk
                    
                    if self.gesto_detectado_actual:
                        color = "#27ae60" if self.gesto_detectado_actual == self.gesto_actual else "#e74c3c"
                        self.estado_label.config(text=f"Detectado: {self.gesto_detectado_actual}", fg=color)
                    else:
                        self.estado_label.config(text="Buscando gesto...", fg="#95a5a6")
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error actualizando video: {e}")
            
            self.update_video_job = self.root.after(33, self.actualizar_video)
    
    def pausar_juego(self):
        self.juego_activo = False
        self.iniciar_btn.config(state='normal')
        self.pausar_btn.config(state='disabled')
        
        if self.update_video_job:
            self.root.after_cancel(self.update_video_job)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        self.video_label.configure(image="", text="Juego pausado\nPresiona 'Iniciar' para continuar", 
                                  font=('Arial', 16), fg='#ecf0f1')
    
    def reiniciar_juego(self):
        self.puntuacion = 0
        self.gesto_actual = random.choice(self.labels_lstm)
        self.tiempo_inicio = time.time()
        self.gesto_reconocido = False
        self.gesto_detectado_actual = None
        self.actualizar_labels()
    
    def actualizar_interfaz(self):
        if self.juego_activo:
            tiempo_transcurrido = time.time() - self.tiempo_inicio
            tiempo_restante = max(0, self.tiempo_limite - tiempo_transcurrido)
            if tiempo_restante <= 0:
                if not self.gesto_reconocido:
                    self.puntuacion = max(0, self.puntuacion - 10)
                self.gesto_actual = random.choice(self.labels_lstm)
                self.tiempo_inicio = time.time()
                self.gesto_reconocido = False
                self.gesto_detectado_actual = None
            self.actualizar_labels()
            self.root.after(100, self.actualizar_interfaz)
    
    def actualizar_labels(self):
        tiempo_transcurrido = time.time() - self.tiempo_inicio
        tiempo_restante = max(0, self.tiempo_limite - tiempo_transcurrido)
        self.gesto_label.config(text=f"Palabra: {self.gesto_actual}")
        self.descripcion_label.config(text=self.obtener_descripcion_gesto(self.gesto_actual))
        self.tiempo_label.config(text=f"Tiempo: {tiempo_restante:.1f}s")
        self.puntuacion_label.config(text=f"Puntuacion: {self.puntuacion}")
        if tiempo_restante > 4:
            self.tiempo_label.config(fg='#27ae60')
        elif tiempo_restante > 2:
            self.tiempo_label.config(fg='#f39c12')
        else:
            self.tiempo_label.config(fg='#e74c3c')
    
    def cerrar_aplicacion(self):
        self.juego_activo = False
        
        if self.update_video_job:
            self.root.after_cancel(self.update_video_job)
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.root.destroy()

    def obtener_descripcion_gesto(self, gesto):
        return f"Gesto: {gesto}"

    def detectar_gesto_lstm(self, manos_landmarks):
        n_landmarks = 21
        n_features = 3
        total_features = n_landmarks * n_features * 2  # 126
        # Solo predecir si hay al menos una mano detectada
        if len(manos_landmarks) == 0:
            self.landmark_buffer = []  # Limpiar buffer si no hay mano
            return None
        row = []
        for lm in manos_landmarks[0].landmark:
            row.extend([lm.x, lm.y, lm.z])
        if len(manos_landmarks) > 1:
            for lm in manos_landmarks[1].landmark:
                row.extend([lm.x, lm.y, lm.z])
        else:
            row.extend([0.0] * (n_landmarks * n_features))
        # Asegura que el vector siempre tenga 126 valores
        if len(row) != total_features:
            row = row[:total_features] if len(row) > total_features else row + [0.0] * (total_features - len(row))
        self.landmark_buffer.append(row)
        if len(self.landmark_buffer) > self.sequence_length:
            self.landmark_buffer.pop(0)
        if len(self.landmark_buffer) == self.sequence_length:
            sequence_np = np.array(self.landmark_buffer).reshape(1, self.sequence_length, total_features)
            pred = self.modelo_lstm.predict(sequence_np, verbose=0)
            idx = np.argmax(pred)
            label = self.labels_lstm[idx]
            return label
        return None

if __name__ == "__main__":
    root = tk.Tk()
    app = HandLanguageGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.cerrar_aplicacion)
    root.mainloop()