from ultralytics import YOLO
import streamlit as st
import cv2
import tempfile
from collections import deque, defaultdict
import os
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Load the pre-trained YOLO model
model = YOLO("best.pt")

# Streamlit dashboard setup
st.set_page_config(page_title="Crowd Control Dashboard", layout="wide")
st.title("Crowd Control Dashboard")

# Sidebar for uploading video and hexbin threshold slider
with st.sidebar:
    uploaded_video = st.file_uploader("Upload a video for crowd analysis", type=["mp4", "mov", "avi"])
    density_threshold = st.slider("Set Density Threshold (per hexagon)", min_value=1, max_value=10, value=1)
    speed_threshold = st.slider("Set Speed Threshold (m/s)", min_value=0.0, max_value=20.0, value=5.0)  # Speed threshold slider

# Real-time placeholders for charts and metrics
object_counts = []
occupancy_percentages = []
rate_of_change_history = []

# Placeholder for alert messages
alert_placeholder = st.empty()
hexbin_alert_placeholder = st.empty()
speed_alert_placeholder = st.empty()  # Placeholder for speed alerts

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
    object_speeds = defaultdict(lambda: deque(maxlen=5))  # Store speeds for each object
    previous_positions = {}  # Store previous frame positions for speed calculations
    crowd_alert_triggered = False

    # Set up the dashboard layout
    col1, col2 = st.columns(2)
    median_speed_placeholder = col2.empty()
    crowd_count_placeholder = col1.empty()

    # Columns for video and hexbin plot
    video_col, hexbin_col = st.columns(2)
    video_placeholder = video_col.empty()
    hexbin_placeholder = hexbin_col.empty()

    # Set up placeholders for real-time charts and metrics
    chart_col1, chart_col2, chart_col3 = st.columns(3)
    with chart_col1:
        density_chart_placeholder = st.empty()
    with chart_col2:
        rate_of_change_chart_placeholder = st.empty()
    with chart_col3:
        occupancy_chart_placeholder = st.empty()

    # Summary metrics placeholders
    summary_metrics_placeholder = st.empty()
    
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
            frame_data = results[0]

            # Annotate detections on the frame
            annotated_frame = frame_data.plot(labels=False)
            boxes = frame_data.boxes.xyxy

            # Get crowd count and density
            crowd_count = len(boxes)
            object_counts.append(crowd_count)

            # Calculate total area of all detected boxes
            total_box_area = sum((box[2] - box[0]) * (box[3] - box[1]) for box in boxes)  # width * height for each box
            frame_area = width * height  # Total area of the frame

            # Calculate occupancy percentage as the sum of all boxes area in the frame
            occupancy_percentage = (total_box_area / frame_area) * 100 if frame_area > 0 else 0
            occupancy_percentages.append(occupancy_percentage)

            # Trigger an alert if the crowd count exceeds a threshold
            if crowd_count > 50 and not crowd_alert_triggered:  # Set a threshold for crowd count
                alert_placeholder.warning("⚠️ Alert: High crowd density detected! Please take necessary action.")
                crowd_alert_triggered = True

            # Track detected positions for hexbin and speed
            current_positions = {}
            x_coords = []  # Reset x_coords for the current frame
            y_coords = []  # Reset y_coords for the current frame

            for idx, box in enumerate(boxes):
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                x_coords.append(x_center)
                y_coords.append(y_center)
                current_positions[idx] = (int(x_center), int(y_center))  # Using loop index as a unique identifier

                # Calculate speed if previous position exists
                if idx in previous_positions:
                    dist = distance.euclidean(previous_positions[idx], (x_center, y_center))
                    speed_mps = (dist * (1 / fps))  # Convert to m/s
                    object_speeds[idx].append(speed_mps)

            # Update previous positions
            previous_positions = current_positions

            # Update video placeholders
            video_placeholder.image(annotated_frame, channels="BGR")

            # Hexbin plot for current frame
            if len(x_coords) > 0:
                plt.clf()  # Clear the previous plot
                plt.figure(figsize=(6, 2.6))
                hexbin_plot = plt.hexbin(x_coords, height - np.array(y_coords), gridsize=6, mincnt=1, vmin=0, vmax=10, cmap='Blues')  # Invert Y-axis
                plt.colorbar(hexbin_plot, label='Density')
                plt.title('Hexbin Density Plot of Object Centers')

                plt.xlim(0, width)
                plt.ylim(0, height)  # Set Y-axis to normal
                hexbin_placeholder.pyplot(plt)

                # Calculate the maximum density in hexbin for the current frame
                current_density = np.max(hexbin_plot.get_array())

                # Check if the current density exceeds the threshold
                if current_density > density_threshold:
                    hexbin_alert_placeholder.warning(f"Alert: Density exceeded threshold! Current Density: {current_density:.2f} objects/hexagon")

            # Update metrics in real-time
            if object_speeds:
                all_speeds = [speed for speeds in object_speeds.values() for speed in speeds if len(speeds) > 0]  # Flatten the list
                median_speed = np.median(all_speeds) if all_speeds else 0  # Calculate median speed
            else:
                median_speed = 0

            # Display median speed
            median_speed_placeholder.metric("Median Speed", f"{median_speed:.2f} m/s")

            # Check if median speed exceeds the threshold
            if median_speed > speed_threshold:
                speed_alert_placeholder.warning(f"⚠️ Alert: Median speed exceeded threshold! Current Median Speed: {median_speed:.2f} m/s")

            crowd_count_placeholder.metric("Current Crowd Count", crowd_count)

            # Update charts in real-time
            if len(object_counts) > 1:
                rate_of_change = np.diff(object_counts)

                # Update Crowd Density over Time chart
                density_chart_placeholder.plotly_chart(
                    go.Figure(data=go.Scatter(
                        x=list(range(len(object_counts))),
                        y=object_counts,
                        mode='lines',
                        name='Object Count per Frame'
                    )).update_layout(
                        title='Crowd Density over Time',
                        xaxis_title='Frame Number',
                        yaxis_title='Object Count',
                        showlegend=False
                    ),
                    use_container_width=True
                )

                # Update Rate of Change in Crowd Density chart
                rate_of_change_chart_placeholder.plotly_chart(
                    go.Figure(data=go.Scatter(
                        x=list(range(len(rate_of_change))),
                        y=rate_of_change,
                        mode='lines',
                        name='Rate of Change in Object Count',
                        line=dict(color='orange')
                    )).update_layout(
                        title='Rate of Change in Crowd Density',
                        xaxis_title='Frame Number',
                        yaxis_title='Rate of Change',
                        showlegend=False
                    ),
                    use_container_width=True
                )

                # Update Occupancy Percentage over Time chart
                occupancy_chart_placeholder.plotly_chart(
                    go.Figure(data=go.Scatter(
                        x=list(range(len(occupancy_percentages))),
                        y=occupancy_percentages,
                        mode='lines',
                        name='Occupancy Percentage',
                        line=dict(color='green')
                    )).update_layout(
                        title='Occupancy Percentage over Time',
                        xaxis_title='Frame Number',
                        yaxis_title='Occupancy (%)',
                        showlegend=False
                    ),
                    use_container_width=True
                )

            # Update summary metrics in real-time
            with summary_metrics_placeholder.container():
                with st.expander("Summary Metrics", expanded=True):
                    # Create columns for layout
                    col1, col2, col3 = st.columns(3)

                    # Display average object count
                    with col1:
                        average_object_count = np.mean(object_counts) if object_counts else 0
                        st.metric(label="Average Object Count", value=f"{average_object_count:.2f}", delta_color="normal")
                        st.write("📊 This represents the average density of the crowd.")

                    # Display maximum object count
                    with col2:
                        max_object_count = max(object_counts) if object_counts else 0
                        st.metric(label="Maximum Object Count", value=f"{max_object_count}", delta_color="normal")
                        st.write("⚠️ Indicates the highest density detected.")

                    # Display minimum object count
                    with col3:
                        min_object_count = min(object_counts) if object_counts else 0
                        st.metric(label="Minimum Object Count", value=f"{min_object_count}", delta_color="normal")
                        st.write("🔍 Shows the lowest density detected.")

    # Release the video capture object
    cap.release()

    # Reset the alert trigger for future analysis
    crowd_alert_triggered = False
else:
    st.info("Please upload a video file to start the analysis.")