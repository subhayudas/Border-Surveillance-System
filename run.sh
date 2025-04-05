#!/bin/bash

# Border Surveillance System Runner Script

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install or update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if GPS setup is needed
if [ "$1" == "--setup-gps" ]; then
    echo "Setting up GPS daemon..."
    bash utils/install_gpsd.sh
    exit 0
fi

# Check for any arguments
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Border Surveillance System"
    echo "Usage:"
    echo "  ./run.sh                  - Run the system normally"
    echo "  ./run.sh --setup-gps      - Install and configure GPS daemon"
    echo "  ./run.sh --no-gps         - Run with GPS disabled"
    echo "  ./run.sh --debug          - Run in debug mode (more verbose output)"
    echo "  ./run.sh --help           - Show this help message"
    exit 0
fi

# Set environment variables based on arguments
if [ "$1" == "--no-gps" ]; then
    export GPS_ENABLED=false
    echo "Running with GPS disabled..."
fi

if [ "$1" == "--debug" ]; then
    export LOG_LEVEL=DEBUG
    echo "Running in debug mode..."
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Create output directory if it doesn't exist
mkdir -p output

# Run the application
echo "Starting Border Surveillance System..."
python main.py

# Deactivate virtual environment when done
deactivate 