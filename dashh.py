from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import streamlit as st
import numpy as np
import tempfile
from collections import deque, defaultdict
import os
import time
from scipy.spatial import distance

# Load the pre-trained YOLO model
model = YOLO("yolov8n.pt")

# Streamlit dashboard setup
st.set_page_config(page_title="Crowd Control Dashboard", layout="wide")
st.title("Crowd Control Dashboard")

# Sidebar for uploading video
with st.sidebar:
    uploaded_video = st.file_uploader("Upload a video for crowd analysis", type=["mp4", "mov", "avi"])

# Real-time placeholders for charts and metrics
object_counts = []
occupancy_percentages = []
rate_of_change_history = []

# Placeholder for alert message
alert_placeholder = st.empty()

if uploaded_video is not None:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        video_path = temp_video.name

    # Ensure the video file exists
    if not os.path.exists(video_path):
        st.error("Error: Video file not found. Please try uploading again.")
        st.stop()

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Tracking history for speed calculations
    history = deque(maxlen=5000)  # Increased history to accumulate all positions
    object_speeds = defaultdict(lambda: deque(maxlen=5))  # Store speeds for each object
    previous_positions = {}  # Store previous frame positions for speed calculations
    crowd_alert_triggered = False

    # Set up the dashboard layout
    col1, col2 = st.columns(2)
    total_crowd_count = 0

    # Placeholder for metrics
    avg_speed_placeholder = col2.empty()
    crowd_count_placeholder = col1.empty()

    # Column for video
    video_col = st.columns(1)[0]
    video_placeholder = video_col.empty()

    # Set up placeholders for real-time charts next to each other
    chart_col1, chart_col2, chart_col3 = st.columns(3)
    with chart_col1:
        density_chart_placeholder = st.empty()
    with chart_col2:
        rate_of_change_chart_placeholder = st.empty()
    with chart_col3:
        occupancy_chart_placeholder = st.empty()
    summary_metrics_placeholder = st.empty()

    # Initialize lists to store data for plots
    object_counts = []
    occupancy_percentages = []

    # Adding a new column for the X-Y plane representation
    xy_plane_col = st.columns(1)[0]
    xy_plane_placeholder = xy_plane_col.empty()

# Process the video in real-time
frame_skip = 5
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Perform inference with YOLO model every nth frame
    if frame_count % frame_skip == 0:
        results = model.predict(frame, show=False)
        if results is None or len(results) == 0:
            continue  # Skip if no detections

        frame_data = results[0]

        # Annotate detections on the frame
        annotated_frame = frame_data.plot(labels=False)
        boxes = frame_data.boxes.xyxy

        # Get crowd count and density
        crowd_count = len(boxes)
        object_counts.append(crowd_count)
        crowd_density = crowd_count / (width * height / 1e6)
        occupancy_percentage = (crowd_count / (width * height / 1e6)) * 100
        occupancy_percentages.append(occupancy_percentage)

        # Track detected positions for speed
        current_positions = {}
        positions = []
        x_coords = []
        y_coords = []

        for idx, box in enumerate(boxes):
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            positions.append((int(x_center), int(y_center)))
            current_positions[idx] = (int(x_center), int(y_center))  # Using loop index as a unique identifier
            x_coords.append(int(x_center))
            y_coords.append(int(y_center))

            # Calculate speed if previous position exists
            if idx in previous_positions:
                dist = distance.euclidean(previous_positions[idx], (x_center, y_center))
                object_speeds[idx].append(dist)

        # Update previous positions
        previous_positions = current_positions
        history.extend(positions)

        # Update video in real-time
        video_placeholder.image(annotated_frame, channels="BGR")

        # Update metrics in real-time
        avg_speed = np.mean([np.mean(speeds) for speeds in object_speeds.values() if len(speeds) > 0]) if object_speeds else 0
        crowd_count_placeholder.metric("Current Crowd Count", crowd_count)
        avg_speed_placeholder.metric("Average Speed", f"{avg_speed:.2f} px/frame")

        # Update X-Y plane with current positions using hexbin
        # Update X-Y plane with current positions using hexbin
        if len(x_coords) > 0:
           plt.clf()  # Clear the previous plot
           plt.figure(figsize=(6, 6))

          # Calculate density values using hexbin
           hexbin_plot = plt.hexbin(x_coords, y_coords, gridsize=15, cmap='Reds', mincnt=1, vmin=0, vmax=5)

          # Adjust vmin and vmax based on your desired density range
           plt.colorbar(hexbin_plot, label='Density')
           plt.xlim(0, width)
           plt.ylim(0, height)  # Set the limits to match the correct orientation
           plt.title('Hexbin Representation of Object Centers')
           plt.xlabel('X Coordinate')
           plt.ylabel('Y Coordinate')
           xy_plane_placeholder.pyplot(plt)



        # Update charts in real-time
        if len(object_counts) > 1:
            rate_of_change = np.diff(object_counts)

            # Update Crowd Density over Time chart
            plt.clf()  # Clear previous plot
            plt.figure(figsize=(10, 6))
            plt.plot(object_counts, label='Object Count per Frame')
            plt.title('Crowd Density over Time')
            plt.xlabel('Frame Number')
            plt.ylabel('Object Count')
            plt.legend()
            plt.grid(True)
            density_chart_placeholder.pyplot(plt)

            # Update Rate of Change in Crowd Density chart
            plt.clf()  # Clear previous plot
            plt.figure(figsize=(10, 6))
            plt.plot(rate_of_change, label='Rate of Change in Object Count', color='orange')
            plt.title('Rate of Change in Crowd Density')
            plt.xlabel('Frame Number')
            plt.ylabel('Rate of Change')
            plt.legend()
            plt.grid(True)
            rate_of_change_chart_placeholder.pyplot(plt)

            # Update Occupancy Percentage over Time chart
            plt.clf()  # Clear previous plot
            plt.figure(figsize=(10, 6))
            plt.plot(occupancy_percentages, label='Occupancy Percentage', color='green')
            plt.title('Occupancy Percentage over Time')
            plt.xlabel('Frame Number')
            plt.ylabel('Occupancy (%)')
            plt.legend()
            plt.grid(True)
            occupancy_chart_placeholder.pyplot(plt)

        # Update summary metrics in real-time
        if len(object_counts) > 1:
            average_object_count = np.mean(object_counts)
            average_occupancy = np.mean(occupancy_percentages)
            summary_metrics_placeholder.write(f'Average Object Count (Crowd Density): {average_object_count:.2f}')
            summary_metrics_placeholder.write(f'Average Occupancy Percentage: {average_occupancy:.2f}%')
            summary_metrics_placeholder.write(f'Maximum Object Count: {max(object_counts)}')
            summary_metrics_placeholder.write(f'Minimum Object Count: {min(object_counts)}')

    # Pause to simulate real-time FPS
    time.sleep(1/fps)

# Release resources after processing all frames
cap.release()
