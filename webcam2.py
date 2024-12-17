from ultralytics import YOLO
import cv2
from collections import deque
import numpy as np

# Load the YOLO model
model = YOLO('best_s_2.pt')

# Initialize webcam
cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define queues for storing detections for each class
class_queues = {
    0: deque(maxlen=5),  # Class 0 (e.g., person)
    1: deque(maxlen=5)   # Class 1 (e.g., car)
}

# Main loop for real-time detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Resize frame to reduce processing time
    frame = cv2.resize(frame, (640, 480))

    # Apply Gaussian blur to reduce noise
    frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Perform YOLO detection with adjusted confidence and IoU thresholds
    results = model(frame_blurred, conf=0.3, iou=0.3)

    # Extract bounding boxes and classes
    boxes = results[0].boxes
    detections = boxes.xyxy.cpu().numpy() if len(boxes) > 0 else np.array([])
    classes = boxes.cls.cpu().numpy() if len(boxes) > 0 else np.array([])

    # Separate detections by class
    class_detections = {0: [], 1: []}
    for box, cls in zip(detections, classes):
        cls = int(cls)  # Convert class to integer
        if cls in class_detections:
            class_detections[cls].append(box)

    # Add detections to their respective class queues
    for cls, dets in class_detections.items():
        if len(dets) > 0:
            class_queues[cls].append(np.array(dets))
        else:
            class_queues[cls].append(np.array([]))

    # Annotate the frame with stabilized bounding boxes
    annotated_frame = frame.copy()
    for cls, queue in class_queues.items():
        # Stabilize detections for the class
        if len(queue) > 1 and all(len(det) > 0 for det in queue):
            # Ensure all previous detections have the same shape
            max_boxes = min(len(det) for det in queue)
            filtered_detections = [det[:max_boxes] for det in queue]
            stabilized_detections = np.mean(filtered_detections, axis=0)
        else:
            stabilized_detections = queue[-1] if len(queue[-1]) > 0 else []

        # Draw bounding boxes for the class
        for box in stabilized_detections:
            x1, y1, x2, y2 = map(int, box[:4])
            color = (0, 255, 0) if cls == 0 else (255, 0, 0)  # Green for class 0, Blue for class 1
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = "Person" if cls == 0 else "Phone"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the annotated frame
    cv2.imshow('YOLO Real-Time Detection with Stabilization', annotated_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()