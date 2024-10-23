# ذَروة (Peak)

<img src="logo.png" alt="Project Logo" width="1000"/> <!-- Adjust width as needed -->


## Overview

a computer vision project that provides a real-time crowd control analysis dashboard utilizing the YOLOv8-seg model for object detection. The system tracks people’s movement and analyzes crowd density, speed, and occupancy metrics from uploaded videos **SAHI** (Slicing Aided Hyper Inference) model was used to optimize inference results, ensuring improved detection accuracy and efficiency.

### Key Features

- **Object Detection**: Utilizes the YOLOv8-seg model to detect people in video frames.
- **Speed Calculation**: Tracks the movement of individuals and calculates their speed.
- **Hexbin Density Plot**: Displays the density of people in the frame as a hexbin plot.
- **Real-time Alerts**: Triggers alerts if the density or speed exceeds user-defined thresholds.
- **Real-time Charts**: Includes visualizations for crowd density, rate of change in density, and occupancy over time.
- **Crowd Count**: Real-time calculation of the number of people in each frame.
- **Occupancy Percentage**: Calculates the percentage of the frame occupied by people.

 <!-- Replace with the actual path to your logo image -->

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/NadaQQ/CV-Crowd-Management.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have a YOLO model weights file (`best.pt`).

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Dependencies

This project requires the following dependencies:

- `ultralytics`: YOLO model for object detection
- `streamlit`: Framework for building the dashboard
- `opencv-python`: For video processing
- `plotly`: For creating interactive plots
- `matplotlib`: For hexbin plots
- `scipy`: For calculating the Euclidean distance in speed analysis
- `numpy`: For numerical operations

## How to Use

1. Upload a video file via the dashboard.
2. Adjust the density and speed thresholds using the sidebar.
3. View real-time crowd analysis.

## Results


