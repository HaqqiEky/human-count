import cv2

# Check available camera devices
for i in range(5):  # Check first 5 indices (adjust range if needed)
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available.")
    cap.release()
