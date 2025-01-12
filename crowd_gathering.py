from ultralytics import YOLO
import cv2
import numpy as np

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # Use yolov8s.pt or yolov8m.pt for better accuracy

# Video path
video_path = "crowd_video.mp4"
cap = cv2.VideoCapture(video_path)

# Output video settings
output_path = "crowd_gathering.mp4"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Line position for counting
count_line_position = 400
offset = 6
counter = 0

# Function to detect gatherings
def detect_gatherings(boxes, threshold=50):
    centers = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in boxes]
    gatherings = []
    for i, c1 in enumerate(centers):
        for j, c2 in enumerate(centers):
            if i != j:
                dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                if dist < threshold:
                    gatherings.append((c1, c2))
    return gatherings

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict using YOLOv8
    results = model(frame)

    # Extract person detections
    person_boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates
        cls = int(box.cls[0])  # Class ID
        if cls == 0:  # Class 0 corresponds to "person"
            person_boxes.append((x1, y1, x2, y2))
            label = f"Person {box.conf[0]:.2f}"  # Confidence
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Counting logic
    detect = []
    for (x1, y1, x2, y2) in person_boxes:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        detect.append((cx, cy))
        if cy < (count_line_position + offset) and cy > (count_line_position - offset):
            counter += 1

    # Draw counting line
    cv2.line(frame, (25, count_line_position), (frame_width - 25, count_line_position), (255, 127, 0), 3)
    cv2.putText(frame, f"Human Count (Cross Line): {counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 0), 2)

    # Detect gatherings
    gatherings = detect_gatherings(person_boxes)
    for g in gatherings:
        c1, c2 = g
        cv2.circle(frame, c1, 10, (0, 0, 255), -1)
        cv2.circle(frame, c2, 10, (0, 0, 255), -1)
        cv2.line(frame, c1, c2, (255, 0, 0), 2)
        midpoint = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
        cv2.putText(frame, "Gathering!", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Show total detections in the frame
    cv2.putText(frame, f"Total People in Frame: {len(person_boxes)}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Human Detection", frame)
    out.write(frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
