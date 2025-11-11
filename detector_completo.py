import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
import time
import os
import csv

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Detecção Completa - Intelbras + YOLOv8")
        self.root.geometry("700x400")
        self.root.resizable(False, False)

        # Centraliza a janela
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 700) // 2
        y = (screen_height - 400) // 2
        self.root.geometry(f"700x400+{x}+{y}")

        # Configurações padrão
        self.camera_ip = "192.168.1.120"
        self.camera_user = "admin"
        self.camera_password = "admin1234"
        self.camera_port = "554"
        self.camera_url = f"rtsp://{self.camera_user}:{self.camera_password}@{self.camera_ip}:{self.camera_port}/cam/realmonitor?channel=1&subtype=0"

        # Variáveis de controle
        self.cap = None
        self.is_running = False
        self.frame_counter = 0
        self.last_detection = None
        self.lock = threading.Lock()

        # Contadores de veículos e pessoas
        self.counts = {
            'car': 0,
            'motorcycle': 0,
            'bus': 0,
            'truck': 0,
            'adult': 0,      # pessoa adulta
            'child': 0,      # pessoa criança
            'dog': 0         # cachorro
        }

        # Controle de logs
        self.log_dir = "logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.current_log_file = self.get_new_log_filename()
        self.log_file_size = 0
        self.max_file_size = 1024 * 1024  # 1 MB

        # Cria o arquivo CSV com cabeçalho
        self.create_log_file(self.current_log_file)

        # Frame principal
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame do vídeo
        self.video_frame = tk.Label(self.main_frame, bg="black", text="Câmera não iniciada", fg="white", font=("Arial", 12))
        self.video_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5)

        # Frame lateral
        self.side_frame = tk.Frame(self.main_frame, width=300, bg="lightgray")
        self.side_frame.grid(row=0, column=1, sticky="ns", padx=5, pady=5)
        self.side_frame.grid_propagate(False)

        # Rótulos de contagem de veículos
        self.car_label = tk.Label(
            self.side_frame,
            text="Carros: 0",
            font=("Arial", 12),
            bg="lightgray"
        )
        self.car_label.pack(pady=3)

        self.moto_label = tk.Label(
            self.side_frame,
            text="Motos: 0",
            font=("Arial", 12),
            bg="lightgray"
        )
        self.moto_label.pack(pady=3)

        self.bus_label = tk.Label(
            self.side_frame,
            text="Ônibus: 0",
            font=("Arial", 12),
            bg="lightgray"
        )
        self.bus_label.pack(pady=3)

        self.truck_label = tk.Label(
            self.side_frame,
            text="Caminhões: 0",
            font=("Arial", 12),
            bg="lightgray"
        )
        self.truck_label.pack(pady=3)

        # Rótulos de contagem de pessoas e animais
        self.adult_label = tk.Label(
            self.side_frame,
            text="Adultos: 0",
            font=("Arial", 12),
            bg="lightgray"
        )
        self.adult_label.pack(pady=3)

        self.child_label = tk.Label(
            self.side_frame,
            text="Crianças: 0",
            font=("Arial", 12),
            bg="lightgray"
        )
        self.child_label.pack(pady=3)

        self.dog_label = tk.Label(
            self.side_frame,
            text="Cachorros: 0",
            font=("Arial", 12),
            bg="lightgray"
        )
        self.dog_label.pack(pady=3)

        # Botão de iniciar/parar
        self.toggle_button = tk.Button(
            self.side_frame,
            text="Iniciar Detecção",
            command=self.toggle_detection,
            bg="green",
            fg="white",
            font=("Arial", 12)
        )
        self.toggle_button.pack(pady=10)

        # Botão de configurações
        self.config_button = tk.Button(
            self.side_frame,
            text="Configurações",
            command=self.open_config,
            bg="orange",
            fg="white",
            font=("Arial", 12)
        )
        self.config_button.pack(pady=5)

        # Botão de fechar
        self.close_button = tk.Button(
            self.side_frame,
            text="Fechar",
            command=self.close_app,
            bg="red",
            fg="white",
            font=("Arial", 12)
        )
        self.close_button.pack(pady=10)

        # Carrega o modelo YOLO
        self.model = YOLO('yolo11n.pt')

        # Classes detectáveis (IDs do COCO dataset)
        self.detection_classes = {
            0: 'person',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            18: 'dog'
        }

    def get_new_log_filename(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_dir, f"log_completo_{timestamp}.csv")

    def create_log_file(self, filename):
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Data/Hora", "Tipo", "Quantidade"])
        self.log_file_size = os.path.getsize(filename)

    def log_detection(self, obj_type, count):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = [timestamp, obj_type, count]

        if self.log_file_size >= self.max_file_size:
            self.current_log_file = self.get_new_log_filename()
            self.create_log_file(self.current_log_file)

        with open(self.current_log_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(log_entry)

        self.log_file_size = os.path.getsize(self.current_log_file)

    def open_config(self):
        config_window = tk.Toplevel(self.root)
        config_window.title("Configurações da Câmera")
        config_window.geometry("300x250")
        config_window.resizable(False, False)

        config_window.transient(self.root)
        config_window.grab_set()

        tk.Label(config_window, text="IP da Câmera:", anchor="w").pack(fill="x", padx=10, pady=5)
        ip_entry = tk.Entry(config_window)
        ip_entry.pack(padx=10, fill="x")
        ip_entry.insert(0, self.camera_ip)

        tk.Label(config_window, text="Usuário:", anchor="w").pack(fill="x", padx=10, pady=5)
        user_entry = tk.Entry(config_window)
        user_entry.pack(padx=10, fill="x")
        user_entry.insert(0, self.camera_user)

        tk.Label(config_window, text="Senha:", anchor="w").pack(fill="x", padx=10, pady=5)
        password_entry = tk.Entry(config_window, show="*")
        password_entry.pack(padx=10, fill="x")
        password_entry.insert(0, self.camera_password)

        tk.Label(config_window, text="Porta:", anchor="w").pack(fill="x", padx=10, pady=5)
        port_entry = tk.Entry(config_window)
        port_entry.pack(padx=10, fill="x")
        port_entry.insert(0, self.camera_port)

        def save_config():
            self.camera_ip = ip_entry.get().strip()
            self.camera_user = user_entry.get().strip()
            self.camera_password = password_entry.get().strip()
            self.camera_port = port_entry.get().strip()
            self.camera_url = f"rtsp://{self.camera_user}:{self.camera_password}@{self.camera_ip}:{self.camera_port}/cam/realmonitor?channel=1&subtype=0"
            messagebox.showinfo("Sucesso", "Configurações da câmera salvas com sucesso!")
            config_window.destroy()

        button_frame = tk.Frame(config_window)
        button_frame.pack(pady=15)
        tk.Button(button_frame, text="Salvar", command=save_config, bg="green", fg="white").pack()

    def toggle_detection(self):
        if not self.is_running:
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.camera_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        
        if not self.cap.isOpened():
            messagebox.showerror("Erro de Conexão", f"Não foi possível conectar à câmera.\nVerifique IP, porta, login e senha.\nURL tentada: {self.camera_url}")
            self.video_frame.config(text="❌ Erro de conexão com a câmera", fg="red", bg="black")
            return
        
        ret, test_frame = self.cap.read()
        if not ret:
            messagebox.showerror("Erro de Fluxo", f"Conexão com a câmera estabelecida, mas não é possível ler o fluxo de vídeo.\nVerifique as credenciais ou a URL.")
            self.cap.release()
            self.video_frame.config(text="❌ Erro ao ler o fluxo da câmera", fg="red", bg="black")
            return
        
        self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        
        self.counts = {k: 0 for k in self.counts}
        self.update_labels()
        
        self.is_running = True
        self.toggle_button.config(text="Parar Detecção", bg="red")
        self.video_frame.config(text="", bg="black")
        self.update_frame()

    def stop_detection(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False
        self.toggle_button.config(text="Iniciar Detecção", bg="green")
        self.video_frame.config(text="Câmera parada", fg="red", bg="black", font=("Arial", 12))

    def update_labels(self):
        self.car_label.config(text=f"Carros: {self.counts['car']}")
        self.moto_label.config(text=f"Motos: {self.counts['motorcycle']}")
        self.bus_label.config(text=f"Ônibus: {self.counts['bus']}")
        self.truck_label.config(text=f"Caminhões: {self.counts['truck']}")
        self.adult_label.config(text=f"Adultos: {self.counts['adult']}")
        self.child_label.config(text=f"Crianças: {self.counts['child']}")
        self.dog_label.config(text=f"Cachorros: {self.counts['dog']}")

    def detect_async(self, frame):
        def run():
            # Detecta todas as classes relevantes
            results = self.model(frame, classes=[0, 2, 3, 5, 7, 18], conf=0.4)
            with self.lock:
                self.last_detection = results

        thread = threading.Thread(target=run)
        thread.start()

    def update_frame(self):
        if not self.is_running:
            return
        
        if self.cap is None:
            self.video_frame.config(text="❌ Câmera não está aberta", fg="red", bg="black")
            return
            
        ret, frame = self.cap.read()
        if ret:
            frame_display = cv2.resize(frame, (320, 240))

            self.frame_counter += 1
            if self.frame_counter % 3 == 0:
                frame_detect = cv2.resize(frame, (160, 120))
                self.detect_async(frame_detect)

            results = None
            with self.lock:
                results = self.last_detection

            # Reinicia contadores
            temp_counts = {k: 0 for k in self.counts}

            if results is not None and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    class_name = self.detection_classes.get(cls_id)
                    if class_name:
                        # Diferencia pessoa adulta de criança
                        if class_name == 'person':
                            bbox = box.xyxy[0]
                            bbox_height = bbox[3] - bbox[1]
                            # Estimativa: bounding box > 40 (na resolução de detecção) = adulto
                            if bbox_height > 40:
                                temp_counts['adult'] += 1
                            else:
                                temp_counts['child'] += 1
                        else:
                            temp_counts[class_name] += 1

            # Atualiza contadores globais
            self.counts = temp_counts
            self.update_labels()

            # Salva log apenas se houver detecções
            for obj_type, count in self.counts.items():
                if count > 0:
                    self.log_detection(obj_type, count)

            # Desenha bounding boxes no frame de exibição
            if results is not None and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    class_name = self.detection_classes.get(cls_id)
                    if class_name:
                        bbox = box.xyxy[0]
                        x1, y1, x2, y2 = map(int, bbox)
                        x1_disp = int(x1 * 2)
                        y1_disp = int(y1 * 2)
                        x2_disp = int(x2 * 2)
                        y2_disp = int(y2 * 2)

                        # Cores por tipo
                        color = (0, 255, 0)  # verde
                        label = class_name
                        if class_name == 'person':
                            bbox_height = bbox[3] - bbox[1]
                            if bbox_height > 40:
                                color = (0, 255, 255)  # amarelo
                                label = 'adult'
                            else:
                                color = (255, 0, 255)  # rosa
                                label = 'child'
                        elif class_name == 'motorcycle':
                            color = (255, 0, 0)  # azul
                        elif class_name == 'bus':
                            color = (0, 0, 255)  # vermelho
                        elif class_name == 'truck':
                            color = (255, 255, 0)  # ciano
                        elif class_name == 'dog':
                            color = (0, 255, 128)  # verde claro

                        cv2.rectangle(frame_display, (x1_disp, y1_disp), (x2_disp, y2_disp), color, 1)
                        cv2.putText(
                            frame_display,
                            label,
                            (x1_disp, y1_disp - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            color,
                            1
                        )

            img = Image.fromarray(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk, text="", bg="black")
        else:
            print("⚠️ Erro ao ler o frame da câmera IP.")
            self.video_frame.configure(image=None)
            self.video_frame.config(text="❌ Erro ao ler o fluxo da câmera", fg="red", bg="black")

        if self.is_running:
            self.root.after(10, self.update_frame)

    def close_app(self):
        self.stop_detection()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()