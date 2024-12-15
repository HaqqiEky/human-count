import cv2
import threading
import queue
import tensorflow as tf
import numpy as np
import tkinter as tk

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model/best_float32.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
    
print("Input details:")
print(input_details)
print("\nOutput details:")
print(output_details)


# Fungsi untuk memproses frame menggunakan model TFLite
def process_frames():
    global detected_count, terminal_status
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Preprocess frame for TFLite model
            input_shape = input_details[0]['shape'][1:3]  # Input size (e.g., 640x640)
            frame_height, frame_width, _ = frame.shape
            resized_frame = cv2.resize(frame, (input_shape[1], input_shape[0]))  # Resize to 640x640
            input_data = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
            input_data = np.float32(input_data) / 255.0  # Normalize to [0, 1]

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # Get detection results
            output_data = interpreter.get_tensor(output_details[0]['index'])  # Output tensor
            output_data = np.squeeze(output_data)  # Remove batch dimension

            # Parse detections
            boxes = output_data[:4, :]  # Bounding box coordinates
            scores = output_data[4, :]  # Confidence scores

            # Filter valid detections
            threshold = 0.4
            valid_detections = scores > threshold
            boxes = boxes[:, valid_detections]
            scores = scores[valid_detections]

            # Update detected count
            detected_count = boxes.shape[1]
            terminal_status = "Full" if detected_count >= 2 else "Normal"

            # Annotate frame
            for i in range(boxes.shape[1]):
                # Convert bounding box format
                x_center, y_center, width, height = boxes[:, i]
                x_min = (x_center - width / 2) * input_shape[1]  # Scale to model input size
                y_min = (y_center - height / 2) * input_shape[0]
                x_max = (x_center + width / 2) * input_shape[1]
                y_max = (y_center + height / 2) * input_shape[0]

                # Scale bounding box to original frame size
                x_min = int(x_min * (frame_width / input_shape[1]))
                x_max = int(x_max * (frame_width / input_shape[1]))
                y_min = int(y_min * (frame_height / input_shape[0]))
                y_max = int(y_max * (frame_height / input_shape[0]))

                # Clip coordinates to frame size
                x_min = max(0, min(frame_width, x_min))
                y_min = max(0, min(frame_height, y_min))
                x_max = max(0, min(frame_width, x_max))
                y_max = max(0, min(frame_height, y_max))

                # Draw bounding box and confidence score
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{scores[i]:.2f}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if not result_queue.full():
                result_queue.put(frame)


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
root.title("Object Detection - TFLite")

label_count = tk.Label(root, text="Detected Persons: 0", font=("Arial", 16))
label_count.pack(pady=10)

label_status = tk.Label(root, text="Terminal Status: Normal", font=("Arial", 16), fg="green")
label_status.pack(pady=10)

# Loop untuk menampilkan hasil webcam
def display_frame():
    if not result_queue.empty():
        annotated_frame = result_queue.get()
        cv2.imshow('Webcam - TFLite', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        root.quit()  # Tutup aplikasi saat 'q' ditekan

    root.after(10, display_frame)  # Panggil fungsi ini lagi setelah 10 ms

# Panggil fungsi UI
update_ui()
display_frame()
root.mainloop()

cv2.destroyAllWindows()
