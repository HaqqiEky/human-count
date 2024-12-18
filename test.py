import cv2
import threading
import queue
from ultralytics import YOLO
import tkinter as tk

# Muat model 
model = YOLO('best_s_3.pt')

# Queue untuk menyimpan frame yang akan diproses
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

# Variable global untuk jumlah objek
detected_count = 0
terminal_status = "Normal"  # Status terminal

# Fungsi untuk menangkap frame dari webcam
def capture_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if not frame_queue.full():
            frame_queue.put(frame)

    cap.release()

# Fungsi untuk memproses frame menggunakan model
def process_frames():
    global detected_count, terminal_status
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Gunakan GPU untuk prediksi
            results = model(frame, conf=0.8, iou=0.3, device="cuda" if model.device.type == "cuda" else "cpu")
            
            # Hitung jumlah objek "person" yang terdeteksi
            detected_count = sum(1 for obj in results[0].boxes if obj.cls == 0)  # Asumsi class 0 adalah "person"

            # Update status terminal
            terminal_status = "Full" if detected_count >= 2 else "Normal"

            # Anotasi frame
            annotated_frame = results[0].plot()

            if not result_queue.full():
                result_queue.put(annotated_frame)

# Fungsi untuk memperbarui UI
def update_ui():
    label_count.config(text=f"Detected Persons: {detected_count}")
    label_status.config(text=f"Terminal Status: {terminal_status}")
    root.after(100, update_ui)  # Perbarui setiap 100 ms

# Jalankan thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
process_thread = threading.Thread(target=process_frames, daemon=True)

capture_thread.start()
process_thread.start()

# Buat UI menggunakan Tkinter
root = tk.Tk()
root.title("Object Detection - YOLOv8")

label_count = tk.Label(root, text="Detected Persons: 0", font=("Arial", 16))
label_count.pack(pady=10)

label_status = tk.Label(root, text="Terminal Status: Normal", font=("Arial", 16), fg="green")
label_status.pack(pady=10)

# Loop untuk menampilkan hasil webcam
def display_frame():
    if not result_queue.empty():
        annotated_frame = result_queue.get()
        cv2.imshow('Webcam - YOLOv8', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        root.quit()  

    root.after(10, display_frame)  # Panggil fungsi ini lagi setelah 10 ms

# Panggil fungsi UI
update_ui()
display_frame()
root.mainloop()

cv2.destroyAllWindows()