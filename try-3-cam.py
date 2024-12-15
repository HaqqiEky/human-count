import cv2
import threading
import queue
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk

# Muat model YOLOv8
model = YOLO('model/best.pt')

# Cek kamera yang tersedia
def check_available_cameras():
    available_cameras = []
    for i in range(5):  # Cek 5 indeks pertama
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} is available.")
            available_cameras.append(i)
        cap.release()
    return available_cameras

# Cek kamera yang tersedia
camera_indices = check_available_cameras()

if not camera_indices:
    print("No available cameras.")
else:
    print(f"Available cameras: {camera_indices}")

    # Mencoba untuk menggunakan kamera yang terdeteksi
    for idx in camera_indices:
        print(f"Trying to access Camera {idx}")
        cap = cv2.VideoCapture(idx)
        
        # Cek apakah kamera berhasil diakses
        if cap.isOpened():
            print(f"Successfully opened Camera {idx}")
            # Tambahkan kode untuk menampilkan atau menggunakan kamera di sini
            cap.release()
        else:
            print(f"Failed to open Camera {idx}")


# Buat queue untuk setiap kamera
frame_queues = [queue.Queue(maxsize=1) for _ in camera_indices]
result_queues = [queue.Queue(maxsize=1) for _ in camera_indices]

# Deteksi jumlah objek dan status terminal
detected_count = [0] * len(camera_indices)
terminal_status = ["Normal"] * len(camera_indices)

# Fungsi untuk menangkap frame dari kamera
def capture_frames(camera_index, queue_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Camera {camera_index} cannot be accessed.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if not frame_queues[queue_index].full():
            frame_queues[queue_index].put(frame)

    cap.release()

# Fungsi untuk memproses frame menggunakan model YOLO
def process_frames(queue_index):
    global detected_count, terminal_status
    while True:
        if not frame_queues[queue_index].empty():
            frame = frame_queues[queue_index].get()

            results = model.predict(frame, conf=0.2, device='cuda')
            
            # Hitung jumlah "person" (asumsi class 0 adalah "person")
            detected_count[queue_index] = sum(1 for obj in results[0].boxes if obj.cls == 0)

            # Update status terminal
            terminal_status[queue_index] = "Full" if detected_count[queue_index] >= 2 else "Normal"

            # Anotasi frame
            annotated_frame = results[0].plot()

            if not result_queues[queue_index].full():
                result_queues[queue_index].put(annotated_frame)

# Fungsi untuk memperbarui UI
def update_ui():
    for idx in range(len(camera_indices)):
        labels_count[idx].config(text=f"Detected Persons (Cam {idx + 1}): {detected_count[idx]}")
        labels_status[idx].config(text=f"Terminal Status (Cam {idx + 1}): {terminal_status[idx]}")
    root.after(100, update_ui)

# Fungsi untuk menampilkan frame di UI
def display_frames():
    for idx, result_queue in enumerate(result_queues):
        if not result_queue.empty():
            annotated_frame = result_queue.get()

            # Konversi frame dari BGR ke RGB
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((400, 300))  # Sesuaikan ukuran frame
            imgtk = ImageTk.PhotoImage(image=img)

            canvas[idx].delete("all")  # Bersihkan canvas sebelumnya
            canvas[idx].create_image(0, 0, image=imgtk, anchor=tk.NW)

            # Simpan referensi untuk mencegah garbage collection
            canvas[idx].image = imgtk

    root.after(10, display_frames)

# Jalankan thread untuk setiap kamera
capture_threads = []
process_threads = []

for idx, cam_idx in enumerate(camera_indices):
    capture_thread = threading.Thread(target=capture_frames, args=(cam_idx, idx), daemon=True)
    process_thread = threading.Thread(target=process_frames, args=(idx,), daemon=True)
    capture_threads.append(capture_thread)
    process_threads.append(process_thread)

for thread in capture_threads + process_threads:
    thread.start()

# Buat UI menggunakan Tkinter
root = tk.Tk()
root.title("Object Detection - YOLOv8")

labels_count = []
labels_status = []
canvas = []

for idx in range(len(camera_indices)):
    frame = tk.Frame(root, bd=2, relief=tk.SOLID)
    frame.pack(pady=10)

    label_count = tk.Label(frame, text=f"Detected Persons (Cam {idx + 1}): 0", font=("Arial", 14))
    label_count.pack()

    label_status = tk.Label(frame, text="Terminal Status: Normal", font=("Arial", 14), fg="green")
    label_status.pack()

    labels_count.append(label_count)
    labels_status.append(label_status)

    canvas_frame = tk.Canvas(frame, width=400, height=300, bg="black")
    canvas_frame.pack()
    canvas.append(canvas_frame)

# Panggil fungsi UI
update_ui()
display_frames()
root.mainloop()

cv2.destroyAllWindows()
