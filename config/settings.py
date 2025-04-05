import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Camera and video settings
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", 0)  # Default to webcam (0)
# Fixed resolution for consistent aspect ratio
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DISPLAY_WIDTH = 640  # Fixed display width
DISPLAY_HEIGHT = 480  # Fixed display height
FPS = int(os.getenv("FPS", 20))

# Detection settings
DETECTION_THRESHOLD = 0.65  # Default detection confidence threshold

# Expect people in surveillance footage (used for adaptive thresholds)
EXPECT_PEOPLE = True

# Classes of interest for the surveillance system
CLASSES_OF_INTEREST = [
    'person', 
    'car', 
    'truck', 
    'bicycle', 
    'motorcycle', 
    'bus', 
    'backpack', 
    'suitcase',
    'bottle',
    'knife',
    'cell phone',
    'umbrella',
    'handbag',
    'tie',
    'frisbee',
    'sports ball',
    'rifle',
    'handgun',
    'shotgun',
    'bazooka',
    'weapon',
    'grenade'
]

# Border crossing detection settings
BORDER_LINES = [
    {
        'id': 'main_border',
        'points': [(0, FRAME_HEIGHT // 2), (FRAME_WIDTH, FRAME_HEIGHT // 2)],
        'direction': 'both'  # 'north_to_south', 'south_to_north', or 'both'
    },
    # Example of a diagonal border
    {
        'id': 'northeast_border',
        'points': [(0, FRAME_HEIGHT), (FRAME_WIDTH, 0)],
        'direction': 'north_to_south'
    }
]

# Fence tampering detection settings
FENCE_REGIONS = []  # List of polygons defining fence areas
TAMPERING_SENSITIVITY = 0.1

# Behavior analysis settings
SUSPICIOUS_BEHAVIORS = {
    'loitering': {
        'time_threshold': 30.0,  # seconds
        'area_threshold': 0.05,  # fraction of frame
    },
    'crawling': {
        'height_ratio_threshold': 0.8,  # height/width ratio
    }
}

# Alert settings
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", 60))  # seconds between identical alerts
SMS_ALERTS_ENABLED = os.getenv("SMS_ALERTS_ENABLED", "false").lower() == "true"
DASHBOARD_ALERTS_ENABLED = os.getenv("DASHBOARD_ALERTS_ENABLED", "true").lower() == "true"

# Twilio settings for SMS alerts
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
ALERT_RECIPIENT_NUMBERS = os.getenv("ALERT_RECIPIENT_NUMBERS", "").split(",")

# MQTT settings for alert distribution
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "border/alerts")

# Geographical settings
GPS_ENABLED = os.getenv("GPS_ENABLED", "false").lower() == "true"
DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", 0.0))
DEFAULT_LON = float(os.getenv("DEFAULT_LON", 0.0))

# Performance settings for edge deployment
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
MODEL_PRECISION = os.getenv("MODEL_PRECISION", "fp16")  # fp32, fp16, or int8
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 1))

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.path.join(BASE_DIR, "logs", "surveillance.log")

# Suspicious object definitions with confidence adjustments
SUSPICIOUS_OBJECTS = {
    'person': {
        'priority': 'high',
        'confidence_adjustment': 1.0,  # No adjustment for person class
    },
    'backpack': {
        'priority': 'medium',
        'confidence_adjustment': 0.9,  # Slightly reduce confidence for backpacks
    },
    # Weapon definitions with priority and confidence adjustments
    'rifle': {
        'priority': 'critical',
        'confidence_adjustment': 1.2,  # Increase confidence for rifles
    },
    'handgun': {
        'priority': 'critical',
        'confidence_adjustment': 1.2,  # Increase confidence for handguns
    },
    'shotgun': {
        'priority': 'critical', 
        'confidence_adjustment': 1.2,  # Increase confidence for shotguns
    },
    'bazooka': {
        'priority': 'critical',
        'confidence_adjustment': 1.3,  # Significant increase for bazookas
    }, 
    'knife': {
        'priority': 'high',
        'confidence_adjustment': 1.1,  # Slight increase for knives
    },
    'grenade': {
        'priority': 'critical',
        'confidence_adjustment': 1.3,  # Significant increase for grenades
    }
}