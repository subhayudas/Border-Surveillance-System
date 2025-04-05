#!/bin/bash

# Script to install and configure GPS daemon for Border Surveillance System
# This will enable real-time location tracking for the map feature

echo "Installing GPS daemon (gpsd) and related utilities..."

# Check OS type and install dependencies
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux installation
    if [ -f /etc/debian_version ]; then
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y gpsd gpsd-clients python3-gps
    elif [ -f /etc/redhat-release ]; then
        # RHEL/CentOS/Fedora
        sudo yum install -y gpsd gpsd-clients python-gps
    else
        echo "Unsupported Linux distribution. Please install gpsd manually."
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS installation using Homebrew
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install gpsd
else
    echo "Unsupported operating system: $OSTYPE"
    echo "Please install gpsd manually for your system."
    exit 1
fi

echo "GPS daemon installed successfully."

# Configuration
echo "Configuring gpsd..."

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Create or update gpsd configuration
    sudo tee /etc/default/gpsd > /dev/null <<EOT
# Default settings for the gpsd init script and the hotplug wrapper.

# Start the gpsd daemon automatically at boot time
START_DAEMON="true"

# Use USB GPS device
DEVICES="/dev/ttyUSB0"

# Other options you want to pass to gpsd
GPSD_OPTIONS="-n"
EOT

    # Restart gpsd service
    sudo systemctl restart gpsd
    echo "GPSD configured to use /dev/ttyUSB0 (default USB GPS device)"
    echo "If your GPS device is connected to a different port, edit /etc/default/gpsd"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "On macOS, you will need to start gpsd manually when needed:"
    echo "  gpsd -N -D 5 /dev/tty.usbserial-*"
    echo "Replace /dev/tty.usbserial-* with your actual GPS device."
fi

echo ""
echo "To verify GPS connection, run one of these commands:"
echo "  cgps -s     (console client)"
echo "  xgps        (graphical client, if installed)"
echo ""
echo "GPS installation and setup complete!"
echo "Note: You may need to connect a GPS receiver to your device for this to work."
echo "      If running without GPS hardware, the system will use the default coordinates." 