import cv2
from ultralytics import YOLO
import numpy as np

# Load the trained YOLOv8 model
model = YOLO('best.pt')  # Replace with your trained weights file

# Open the video file
input_video_path = 'crowded (1).mp4'  # Replace with your video file path
cap = cv2.VideoCapture(input_video_path)

# Get video properties for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second

# Create VideoWriter object to save the output video
output_video_path = 'output_video.mp4'  # Path for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize variables for speed estimation
previous_positions = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Iterate over detected objects
    current_positions = {}
    for result in results:
        boxes = result.boxes  # Accessing the boxes attribute
        for box in boxes:
            # Unpack box attributes
            x1, y1, x2, y2 = box.xyxy[0]  # Get bounding box coordinates directly
            conf = box.conf[0]  # Confidence score
            cls = box.cls[0]    # Class ID

            # Convert to int for drawing
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            object_id = int(cls)  # Assuming class ID as the object ID

            # Calculate center position of the bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_positions[object_id] = (center_x, center_y)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {object_id} Conf: {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Estimate speed
    for object_id, current_position in current_positions.items():
        if object_id in previous_positions:
            previous_position = previous_positions[object_id]
            # Calculate distance traveled (Euclidean distance)
            distance = np.linalg.norm(np.array(current_position) - np.array(previous_position))
            speed = distance * fps  # Speed in pixels per second

            # Display speed on frame
            cv2.putText(frame, f'Speed: {speed:.2f} px/s', (10, 30 + object_id * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update previous positions
    previous_positions = current_positions

    # Write the frame to the output video
    out.write(frame)

    # Show the frame with detections and speed
    cv2.imshow('YOLOv8 Speed Estimation', frame)

    # Break on 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()