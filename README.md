# AI-Powered Border Surveillance System

An autonomous surveillance system that detects, tracks, and verifies threats in real-time using multimodal AI. This system provides comprehensive monitoring of border areas with intelligent threat detection capabilities.

## Features

- **Intrusion Detection**: Identifies unauthorized personnel, vehicles, drones, and fence tampering
- **Behavior Analysis**: Flags suspicious activities (loitering, crawling) using temporal models
- **Edge Deployment**: Operates offline on low-power devices (NVIDIA Jetson) for remote areas
- **Alert System**: Triggers instant notifications via dashboards or SMS with GPS coordinates
- **Multi-Camera Support**: Manage and monitor multiple camera feeds simultaneously
- **Video Analysis**: Upload and analyze pre-recorded videos for retrospective threat assessment
- **Interactive Dashboard**: Real-time statistics and alert history visualization

## Project Structure

```
Border-Surveillance-System/
├── src/               # Source code for surveillance modules
│   ├── detection/     # Object detection and behavior analysis modules
│   ├── visualization/ # Visualization utilities
│   └── run.py         # Main entry point for standalone mode
├── models/            # Pre-trained AI models
├── config/            # Configuration files
├── utils/             # Utility functions and helpers
├── logs/              # System logs
├── output/            # Detection output and captured images
├── main.py            # Interactive GUI application
├── run.sh             # Execution script with environment setup
├── requirements.txt   # Dependencies
└── .env               # Environment configuration
```

## Requirements

The system requires the following major dependencies:
- Python 3.8+
- PyTorch 1.9.0+
- OpenCV 4.5.0+
- Ultralytics (YOLOv8)
- FastAPI (for dashboard backend)
- Additional dependencies listed in `requirements.txt`

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/Border-Surveillance-System.git
   cd Border-Surveillance-System
   ```

2. Set up the environment and install dependencies using the provided script:
   ```
   chmod +x run.sh
   ./run.sh
   ```
   
   This script will:
   - Create a virtual environment (if not already present)
   - Install all required dependencies
   - Create necessary directories
   - Set up the configuration file

3. Configure settings in `.env` for your specific deployment needs.

## Running the System

### GUI Mode (Interactive)

To run the system with the full graphical interface:

```
python main.py
```

This launches the interactive dashboard where you can:
- Add and manage camera feeds
- Start/stop surveillance
- Upload and analyze videos
- View detection statistics and alerts

### Command-Line Mode

For headless operation or server deployment:

```
./run.sh
```

Optional arguments:
- `-s, --source`: Specify video source (camera index, RTSP URL, or file path)
- `-o, --output`: Set output directory for recordings
- `--no-display`: Run without displaying video feed (for headless servers)
- `--dashboard-host`: Set dashboard API host (default: 0.0.0.0)
- `--dashboard-port`: Set dashboard API port (default: 8000)
- `--dashboard-only`: Run only the dashboard component
- `--surveillance-only`: Run only the surveillance component

Example:
```
./run.sh --source rtsp://camera_url --no-display
```

## Dashboard Access

When running in command-line mode, the dashboard is accessible at:
```
http://<host>:8000
```

## Deployment

The system is designed to run on edge devices such as NVIDIA Jetson or similar hardware with GPU capabilities for real-time inference. For production deployment, consider:

- Setting up as a system service for automatic startup
- Configuring remote monitoring capabilities
- Implementing data retention policies for captured footage

## License

MIT 