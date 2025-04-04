# AI-Powered Border Surveillance System

An autonomous surveillance system that detects, tracks, and verifies threats in real-time using multimodal AI.

## Features

- **Intrusion Detection**: Identifies unauthorized personnel, vehicles, drones, and fence tampering
- **Behavior Analysis**: Flags suspicious activities (loitering, crawling) using temporal models
- **Edge Deployment**: Operates offline on low-power devices (NVIDIA Jetson) for remote areas
- **Alert System**: Triggers instant notifications via dashboards or SMS with GPS coordinates

## Project Structure

```
Border-Surveillance-System/
├── src/               # Source code
├── models/            # Pre-trained AI models
├── config/            # Configuration files
├── utils/             # Utility functions
├── tests/             # Test scripts
└── docs/              # Documentation
```

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure settings in `config/settings.py`
4. Run the system: `python src/main.py`

## Deployment

The system is designed to run on edge devices such as NVIDIA Jetson or similar hardware with GPU capabilities for real-time inference.

## License

MIT 