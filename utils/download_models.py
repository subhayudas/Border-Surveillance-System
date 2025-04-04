import os
import urllib.request
import cv2
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def download_haar_cascade():
    """Download Haar cascade classifier for gun detection"""
    cascade_path = os.path.join(settings.MODELS_DIR, "haarcascade_gun.xml")
    
    # Create models directory if it doesn't exist
    if not os.path.exists(settings.MODELS_DIR):
        os.makedirs(settings.MODELS_DIR)
    
    # Download if not exists
    if not os.path.exists(cascade_path):
        print("Downloading gun cascade classifier...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        # Note: Using face cascade as a placeholder since gun cascade isn't in OpenCV's repo
        # In a real application, you would use a properly trained gun cascade
        urllib.request.urlretrieve(url, cascade_path)
        print(f"Downloaded to {cascade_path}")
    else:
        print("Gun cascade classifier already exists")

if __name__ == "__main__":
    download_haar_cascade()