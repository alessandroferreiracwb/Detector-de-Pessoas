import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
import time

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Detecção de Pessoas - Intelbras + YOLOv8")
        self.root.geometry("600x400")
        self.root.resizable(False, False)

        # Centraliza a janela
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 600) // 2
        y = (screen_height - 400) // 2
        self.root.geometry(f"600x400+{x}+{y}")

        # Configurações padrão
        self.camera_ip = "192.168.1.120"
        self.camera_user = "admin"
        self.camera_password = "admin1234"
        self.camera_port = "554"
        self.camera_url = f"rtsp://{self.camera_user}:{self.camera_password}@{self.camera_ip}:{self.camera_port}/cam/realmonitor?channel=1&subtype=0"

        # Configurações de calibração
        self.REFERENCE_HEIGHT = 2.70
        self.CALIBRATION_DISTANCE = 2.5
        self.REFERENCE_PIXELS = 80

        # Variáveis de controle
        self.cap = None
        self.is_running = False
        self.frame_counter = 0
        self.last_detection = None
        self.lock = threading.Lock()

        # Frame principal
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame do vídeo
        self.video_frame = tk.Label(self.main_frame, bg="black", text="Câmera não iniciada", fg="white", font=("Arial", 12))
        self.video_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5)

        # Frame lateral
        self.side_frame = tk.Frame(self.main_frame, width=200, bg="lightgray")
        self.side_frame.grid(row=0, column=1, sticky="ns", padx=5, pady=5)
        self.side_frame.grid_propagate(False)

        # Rótulo de contagem de pessoas
        self.count_label = tk.Label(
            self.side_frame,
            text="Pessoas detectadas: 0",
            font=("Arial", 12),
            bg="lightgray"
        )
        self.count_label.pack(pady=10)

        # Texto: "Pessoa mais próxima:"
        self.closest_title = tk.Label(
            self.side_frame,
            text="Pessoa mais próxima:",
            font=("Arial", 12),
            bg="lightgray"
        )
        self.closest_title.pack(pady=(20, 0))

        # Valor da distância — embaixo, maior e destacado
        self.closest_value = tk.Label(
            self.side_frame,
            text="-- m",
            font=("Arial", 24, "bold"),
            bg="lightgray",
            fg="blue"
        )
        self.closest_value.pack(pady=5)

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

        # Botão de calibrar
        self.calibrate_button = tk.Button(
            self.side_frame,
            text="Calibrar",
            command=self.open_calibrate,
            bg="blue",
            fg="white",
            font=("Arial", 12)
        )
        self.calibrate_button.pack(pady=5)

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

        # Carrega o modelo
        self.model = YOLO('yolo11n.pt')

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

    def open_calibrate(self):
        if not self.is_running:
            messagebox.showwarning("Aviso", "A câmera deve estar rodando para calibrar.")
            return

        cal_window = tk.Toplevel(self.root)
        cal_window.title("Calibração")
        cal_window.geometry("300x200")
        cal_window.resizable(False, False)

        cal_window.transient(self.root)
        cal_window.grab_set()

        tk.Label(cal_window, text="Altura da pessoa (m):", anchor="w").pack(fill="x", padx=10, pady=5)
        height_entry = tk.Entry(cal_window)
        height_entry.pack(padx=10, fill="x")
        height_entry.insert(0, str(self.REFERENCE_HEIGHT))

        tk.Label(cal_window, text="Distância de calibração (m):", anchor="w").pack(fill="x", padx=10, pady=5)
        distance_entry = tk.Entry(cal_window)
        distance_entry.pack(padx=10, fill="x")
        distance_entry.insert(0, str(self.CALIBRATION_DISTANCE))

        def calibrate_now():
            try:
                new_height = float(height_entry.get())
                new_distance = float(distance_entry.get())
                self.REFERENCE_HEIGHT = new_height
                self.CALIBRATION_DISTANCE = new_distance
            except ValueError:
                messagebox.showerror("Erro", "Por favor, insira valores numéricos válidos.")
                return

            ret, frame = self.cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (160, 120))
                results = self.model(frame_resized, classes=0, conf=0.5)

                if results[0].boxes is not None:
                    largest_box = max(results[0].boxes, key=lambda box: (box.xyxy[0][3] - box.xyxy[0][1]))
                    bbox_height = (largest_box.xyxy[0][3] - largest_box.xyxy[0][1]).item()
                    self.REFERENCE_PIXELS = bbox_height
                    print(f"REFERÊNCIA ATUALIZADA: {self.REFERENCE_PIXELS:.2f} pixels (a {self.CALIBRATION_DISTANCE}m com altura {self.REFERENCE_HEIGHT}m)")
                    messagebox.showinfo(
                        "Calibrado com Sucesso",
                        f"Referência definida com base em:\n"
                        f"Altura da pessoa: {self.REFERENCE_HEIGHT} m\n"
                        f"Distância: {self.CALIBRATION_DISTANCE} m\n"
                        f"Altura no frame: {self.REFERENCE_PIXELS:.2f} pixels"
                    )
                else:
                    messagebox.showwarning("Aviso", "Nenhuma pessoa detectada para calibrar.")
            else:
                messagebox.showerror("Erro", "Não foi possível ler o frame da câmera.")

        tk.Button(cal_window, text="Calibrar", command=calibrate_now, bg="blue", fg="white").pack(pady=10)

    def toggle_detection(self):
        if not self.is_running:
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        if self.cap is not None:
            self.cap.release()
        
        # Tenta abrir a câmera
        self.cap = cv2.VideoCapture(self.camera_url)
        
        # Define timeouts e buffers para evitar travamentos
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        
        # Verifica se a câmera foi aberta com sucesso
        if not self.cap.isOpened():
            messagebox.showerror("Erro de Conexão", f"Não foi possível conectar à câmera.\nVerifique IP, porta, login e senha.\nURL tentada: {self.camera_url}")
            self.video_frame.config(text="❌ Erro de conexão com a câmera", fg="red", bg="black")
            return
        
        # Tenta ler um frame para verificar se a conexão é válida
        ret, test_frame = self.cap.read()
        if not ret:
            messagebox.showerror("Erro de Fluxo", f"Conexão com a câmera estabelecida, mas não é possível ler o fluxo de vídeo.\nVerifique as credenciais ou a URL.")
            self.cap.release()
            self.video_frame.config(text="❌ Erro ao ler o fluxo da câmera", fg="red", bg="black")
            return
        
        # Se tudo estiver OK, libera o frame de teste e inicia o loop
        self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        
        self.is_running = True
        self.toggle_button.config(text="Parar Detecção", bg="red")
        self.video_frame.config(text="", bg="black")  # Limpa mensagem de erro
        self.update_frame()

    def stop_detection(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False
        self.toggle_button.config(text="Iniciar Detecção", bg="green")
        self.video_frame.config(text="Câmera parada", fg="red", bg="black", font=("Arial", 12))

    def estimate_distance(self, bbox_height):
        distance = (self.REFERENCE_PIXELS * self.CALIBRATION_DISTANCE) / bbox_height
        return round(distance, 2)

    def detect_async(self, frame):
        def run():
            results = self.model(frame, classes=0, conf=0.4)
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

            people_count = 0
            distances = []

            if results is not None and results[0].boxes is not None:
                people_count = len(results[0].boxes)
                for box in results[0].boxes:
                    bbox = box.xyxy[0]
                    x1, y1, x2, y2 = map(int, bbox)
                    bbox_height = bbox[3] - bbox[1]
                    distance = self.estimate_distance(bbox_height.item())
                    distances.append(distance)

                    x1_disp = int(x1 * 2)
                    y1_disp = int(y1 * 2)
                    x2_disp = int(x2 * 2)
                    y2_disp = int(y2 * 2)

                    cv2.rectangle(frame_display, (x1_disp, y1_disp), (x2_disp, y2_disp), (0, 255, 0), 1)
                    cv2.putText(
                        frame_display,
                        f'{distance:.2f}m',
                        (x1_disp, y1_disp - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 0),
                        1
                    )

            self.count_label.config(text=f"Pessoas detectadas: {people_count}")
            closest = min(distances) if distances else None
            if closest is not None:
                self.closest_value.config(text=f"{closest:.2f} m")
            else:
                self.closest_value.config(text="-- m")

            img = Image.fromarray(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk, text="", bg="black")
        else:
            # Erro ao ler frame - pode ser perda de conexão
            print("⚠️ Erro ao ler o frame da câmera IP. Verifique a conexão.")
            self.video_frame.configure(image=None)
            self.video_frame.config(text="❌ Erro ao ler o fluxo da câmera", fg="red", bg="black")
            # Opcional: tentar reconectar automaticamente
            # self.stop_detection()

        if self.is_running:
            self.root.after(10, self.update_frame)

    def close_app(self):
        self.stop_detection()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()