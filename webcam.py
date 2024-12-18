from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('best_s_3.pt')

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop for real-time detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    results = model(frame, conf=0.4)
    annotated_frame = results[0].plot()

    cv2.imshow('YOLO Real-Time Detection with CUDA', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()