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
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 640))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 480))
FPS = int(os.getenv("FPS", 20))

# Detection settings
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", 0.65))
CLASSES_OF_INTEREST = [
    "person", 
    "car", 
    "truck", 
    "motorcycle", 
    "bicycle", 
    "drone",
    "airplane",
    "backpack",
    "suitcase",
    "handbag",
    "umbrella",
    "cell phone"
]

# Fence tampering detection settings
FENCE_REGIONS = []  # List of polygons defining fence areas
TAMPERING_SENSITIVITY = float(os.getenv("TAMPERING_SENSITIVITY", 0.3))

# Behavior analysis settings
SUSPICIOUS_BEHAVIORS = {
    "loitering": {
        "time_threshold": int(os.getenv("LOITERING_TIME", 30)),  # in seconds
        "area_threshold": float(os.getenv("LOITERING_AREA", 0.2))  # in ratio of frame
    },
    "crawling": {
        "height_ratio_threshold": float(os.getenv("CRAWLING_RATIO", 0.5))  # height/width ratio
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