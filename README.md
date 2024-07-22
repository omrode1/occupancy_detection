# Operator Occupancy Module

## Overview

The Operator Occupancy Module is designed to detect and monitor the occupancy of operators within specified regions of interest (ROI) using multiple camera feeds. The system initializes multiple cameras simultaneously, records the in and out times of operators, and stores this data in a continuous database. During each shift, the recorded data is summarized and stored in a separate database. The summarized data is then visualized using a dedicated dashboard.

## Features

- **Multiple Camera Initialization:** Simultaneously initializes and manages multiple camera feeds.
- **ROI-Based Detection:** Detects operator occupancy within specified regions of interest (ROI).
- **Continuous Data Recording:** Records in and out times of operators in a continuous database.
- **Shift-Based Summarization:** Summarizes data at the end of each shift and stores it in a summarized database.
- **Dashboard Visualization:** Provides a dashboard for visualizing summarized data and monitoring operator occupancy.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/occupancy_detection.git
    cd occupancy_detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Configure the system settings by editing the `config.json` file:
    ```json
    {
      "camera_feeds": ["rtsp://camera1_url", "rtsp://camera2_url"],
      "roi_values": {
        "camera1": {"x": 0, "y": 0, "width": 100, "height": 200},
        "camera2": {"x": 50, "y": 50, "width": 150, "height": 250}
      },
      "continuous_db": "continuous_database.db",
      "summarized_db": "summarized_database.db",
      "shift_duration": "8 hours",
      "dashboard_port": 8080
    }
    ```

## Usage

1. **Start the Occupancy Detection Module:**
    ```bash
    python main.py
    ```

2. **Access the Dashboard:**
   Open your web browser and navigate to `http://localhost:8080` to view the dashboard and monitor operator occupancy.

## Database Structure

### Continuous Database

- **Table Name:** `occupancy_records`
- **Columns:**
  - `id`: Integer, Primary Key
  - `camera_id`: Text
  - `operator_id`: Text
  - `in_time`: Timestamp
  - `out_time`: Timestamp

### Summarized Database

- **Table Name:** `shift_summaries`
- **Columns:**
  - `id`: Integer, Primary Key
  - `camera_id`: Text
  - `operator_id`: Text
  - `shift_start`: Timestamp
  - `shift_end`: Timestamp
  - `total_time`: Integer (in minutes)

## Configuration

The `config.json` file allows customization of various settings such as camera feeds, ROI values, database paths, shift duration, and the dashboard port. Ensure that the configuration file is properly set up before running the module.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


