import cv2
from ultralytics import YOLO
import numpy as np

# Load the trained YOLOv8 model
model = YOLO('best.pt')  # Replace with your trained weights file

# Open the video file
input_video_path = '1338598-hd_1920_1080_30fps.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(input_video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second

# Create VideoWriter object to save the output video
output_video_path = 'output8_video.mp4'  # Path for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize variables for speed estimation
previous_positions = {}
speed_threshold_mps = 4.17  # Speed threshold in meters per second (15 km/h)
pixel_to_meter_ratio = 0.05  # Conversion factor (1 pixel = 0.05 meters)
speeds = []  # List to hold individual speeds for median calculation

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
            distance_pixels = np.linalg.norm(np.array(current_position) - np.array(previous_position))
            speed_mps = (distance_pixels * pixel_to_meter_ratio) * fps  # Speed in meters per second

            # Store the speed for median calculation
            speeds.append(speed_mps)

            # Display speed on frame in m/s
            cv2.putText(frame, f'Speed: {speed_mps:.2f} m/s', (10, 30 + object_id * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update previous positions
    previous_positions = current_positions

    # Calculate median speed if any speeds are recorded
    if speeds:
        median_speed = np.median(speeds)

        # Check against median speed threshold and print alert
        if median_speed > speed_threshold_mps:
            print(f'Alert: High median speed! Median Speed: {median_speed:.2f} m/s')

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