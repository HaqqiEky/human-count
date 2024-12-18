import cv2
import threading
import queue
import numpy as np
from collections import deque
from ultralytics import YOLO
import tkinter as tk

# Muat model YOLO
model = YOLO('best_s_3.pt')

# Queue untuk frame dan hasil deteksi
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

# Antrian stabilisasi untuk setiap classno
class_queues = {
    0: deque(maxlen=5),  # Person
    1: deque(maxlen=5)   # Phone
}

# Variabel global untuk jumlah objek
person_count = 0
phone_count = 0

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

# Fungsi untuk memproses frame menggunakan YOLO
def process_frames():
    global person_count, phone_count
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Gunakan model YOLO untuk deteksi
            results = model(frame, conf=0.4, iou=0.3, device="cuda" if model.device.type == "cuda" else "cpu")
            boxes = results[0].boxes
            detections = boxes.xyxy.cpu().numpy() if len(boxes) > 0 else []
            classes = boxes.cls.cpu().numpy() if len(boxes) > 0 else []

            # Hitung jumlah objek per kelas
            person_count = np.sum(classes == 0)
            phone_count = np.sum(classes == 1)

            # Stabilisasi deteksi
            class_detections = {0: [], 1: []}
            for box, cls in zip(detections, classes):
                cls = int(cls)
                if cls in class_detections:
                    class_detections[cls].append(box)

            # Tambahkan hasil ke antrian untuk stabilisasi
            for cls, dets in class_detections.items():
                if len(dets) > 0:
                    class_queues[cls].append(np.array(dets))
                else:
                    class_queues[cls].append(np.array([]))

            # Stabilkan bounding boxes
            annotated_frame = frame.copy()
            for cls, queue in class_queues.items():
                if len(queue) > 1 and all(len(det) > 0 for det in queue):
                    max_boxes = min(len(det) for det in queue)
                    filtered_detections = [det[:max_boxes] for det in queue]
                    stabilized_detections = np.mean(filtered_detections, axis=0)
                else:
                    stabilized_detections = queue[-1] if len(queue[-1]) > 0 else []

                # Anotasi frame
                for box in stabilized_detections:
                    x1, y1, x2, y2 = map(int, box[:4])
                    color = (0, 255, 0) if cls == 0 else (255, 0, 0)
                    label = "Person" if cls == 0 else "Phone"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if not result_queue.full():
                result_queue.put(annotated_frame)

# Fungsi untuk memperbarui UI
def update_ui():
    label_person.config(text=f"Detected Persons: {person_count}")
    label_phone.config(text=f"Detected Phones: {phone_count}")
    root.after(100, update_ui)

# Fungsi untuk menampilkan frame
def display_frame():
    if not result_queue.empty():
        frame = result_queue.get()
        cv2.imshow("YOLO Detection with Stabilization", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        root.quit()
    root.after(10, display_frame)

# Jalankan thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
process_thread = threading.Thread(target=process_frames, daemon=True)

capture_thread.start()
process_thread.start()

# UI dengan Tkinter
root = tk.Tk()
root.title("YOLO Detection - Real-Time Count")

label_person = tk.Label(root, text="Detected Persons: 0", font=("Arial", 16))
label_person.pack(pady=10)

label_phone = tk.Label(root, text="Detected Phones: 0", font=("Arial", 16))
label_phone.pack(pady=10)

# Update UI dan tampilkan frame
update_ui()
display_frame()

root.mainloop()
cv2.destroyAllWindows()
