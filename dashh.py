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
import seaborn as sns

# Set Seaborn's style globally for all plots
sns.set(style="whitegrid")

# Load the pre-trained YOLO model
model = YOLO("yolov8n.pt")

# Streamlit dashboard setup
st.set_page_config(page_title="Crowd Control Dashboard", layout="wide")
st.title("Crowd Control Dashboard")

# Sidebar for uploading video and density threshold
with st.sidebar:
    uploaded_video = st.file_uploader("Upload a video for crowd analysis", type=["mp4", "mov", "avi"])
    density_threshold = st.slider("Set Density Threshold (per hexagon)", min_value=1, max_value=10, value=1)

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
    history = deque(maxlen=5000)
    object_speeds = defaultdict(lambda: deque(maxlen=5))
    previous_positions = {}
    crowd_alert_triggered = False

    # Set up the dashboard layout
    col1, col2 = st.columns(2)
    total_crowd_count = 0

    # Placeholder for metrics
    avg_speed_placeholder = col2.empty()
    crowd_count_placeholder = col1.empty()

    # Columns for video and hexbin plot
    video_col, hexbin_col = st.columns(2)
    video_placeholder = video_col.empty()
    hexbin_placeholder = hexbin_col.empty()

    # Set up placeholders for real-time charts below the video and hexbin plot
    chart_col1, chart_col2, chart_col3 = st.columns(3)
    with chart_col1:
        rate_of_change_chart_placeholder = st.empty()
    with chart_col2:
        occupancy_chart_placeholder = st.empty()
    with chart_col3:
        summary_metrics_placeholder = st.empty()

    # Initialize lists to store data for plots
    object_counts = []
    occupancy_percentages = []

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
                continue

            frame_data = results[0]
            annotated_frame = frame_data.plot(labels=False)
            boxes = frame_data.boxes.xyxy

            # Get crowd count and density
            crowd_count = len(boxes)
            object_counts.append(crowd_count)
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
                current_positions[idx] = (int(x_center), int(y_center))
                x_coords.append(int(x_center))
                y_coords.append(int(y_center))

                if idx in previous_positions:
                    dist = distance.euclidean(previous_positions[idx], (x_center, y_center))
                    object_speeds[idx].append(dist)

            previous_positions = current_positions
            history.extend(positions)

            # Display the video frame
            video_placeholder.image(annotated_frame, channels="BGR")

            # Hexbin plot for density
            if len(x_coords) > 0:
                plt.clf()
                plt.figure(figsize=(8, 6))
                hexbin_plot = plt.hexbin(x_coords, height - np.array(y_coords), gridsize=10, mincnt=1, vmin=1, vmax=10, cmap='Reds')
                plt.colorbar(hexbin_plot, label='Density')
                plt.title('Hexbin Density Plot of Object Centers')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.xlim(0, width)
                plt.ylim(0, height)
                hexbin_placeholder.pyplot(plt)

                current_density = np.max(hexbin_plot.get_array())
                if current_density > density_threshold:
                    alert_placeholder.warning(f"Alert: Density exceeded threshold! Current Density: {current_density} objects/hexagon")

            if len(object_counts) > 1:
                rate_of_change = np.diff(object_counts)

                plt.clf()
                plt.figure(figsize=(10, 6))
                plt.plot(rate_of_change, label='Rate of Change in Object Count', color='orange')
                plt.title('Rate of Change in Crowd Density')
                plt.xlabel('Frame Number')
                plt.ylabel('Rate of Change')
                plt.legend()
                plt.grid(True)
                rate_of_change_chart_placeholder.pyplot(plt)

                plt.clf()
                plt.figure(figsize=(10, 6))
                plt.plot(occupancy_percentages, label='Occupancy Percentage', color='green')
                plt.title('Occupancy Percentage over Time')
                plt.xlabel('Frame Number')
                plt.ylabel('Occupancy (%)')
                plt.legend()
                plt.grid(True)
                occupancy_chart_placeholder.pyplot(plt)

                if len(object_counts) > 1:
                    average_object_count = np.mean(object_counts)
                    average_occupancy = np.mean(occupancy_percentages)
                    summary_metrics_placeholder.write(f'Average Object Count (Crowd Density): {average_object_count:.2f}')
                    summary_metrics_placeholder.write(f'Average Occupancy Percentage: {average_occupancy:.2f}%')
                    summary_metrics_placeholder.write(f'Maximum Object Count: {max(object_counts)}')
                    summary_metrics_placeholder.write(f'Minimum Object Count: {min(object_counts)}')

        time.sleep(1/fps)

    cap.release()
