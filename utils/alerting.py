import time
import json
import threading
from datetime import datetime
import paho.mqtt.client as mqtt
from twilio.rest import Client

from config import settings
from utils.logger import logger

class AlertManager:
    def __init__(self):
        self.last_alerts = {}  # To track alert cooldowns
        self._lock = threading.Lock()  # Thread safety for alert tracking
        
        # Initialize Twilio client if SMS alerts are enabled
        self.twilio_client = None
        if settings.SMS_ALERTS_ENABLED and settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN:
            try:
                self.twilio_client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
                logger.info("Twilio client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Twilio client: {str(e)}")
        
        # Initialize MQTT client if dashboard alerts are enabled
        self.mqtt_client = None
        if settings.DASHBOARD_ALERTS_ENABLED:
            try:
                self.mqtt_client = mqtt.Client()
                self.mqtt_client.connect(settings.MQTT_BROKER, settings.MQTT_PORT, 60)
                self.mqtt_client.loop_start()
                logger.info("MQTT client connected successfully")
            except Exception as e:
                logger.error(f"Failed to connect to MQTT broker: {str(e)}")
    
    def send_alert(self, alert_type, message, location=None, image_path=None, confidence=None):
        """
        Send an alert through configured channels
        
        Args:
            alert_type (str): Type of alert (intrusion, tampering, etc.)
            message (str): Alert message
            location (tuple, optional): GPS coordinates (lat, lon)
            image_path (str, optional): Path to snapshot image
            confidence (float, optional): Detection confidence score
        """
        # Check cooldown period for this alert type
        current_time = time.time()
        with self._lock:
            if alert_type in self.last_alerts:
                last_time = self.last_alerts[alert_type]
                if current_time - last_time < settings.ALERT_COOLDOWN:
                    return  # Skip this alert, still in cooldown
            
            # Update last alert time
            self.last_alerts[alert_type] = current_time
        
        # Format location string if available
        location_str = f"{location[0]:.6f}, {location[1]:.6f}" if location else "Unknown"
        
        # Create alert data structure
        alert_data = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "location": location_str,
            "confidence": confidence if confidence is not None else 0.0,
            "image": image_path
        }
        
        # Send SMS alert if enabled
        if settings.SMS_ALERTS_ENABLED and self.twilio_client and settings.TWILIO_PHONE_NUMBER:
            self._send_sms_alert(alert_data)
        
        # Send dashboard alert if enabled
        if settings.DASHBOARD_ALERTS_ENABLED and self.mqtt_client:
            self._send_dashboard_alert(alert_data)
    
    def _send_sms_alert(self, alert_data):
        """Send SMS notification using Twilio"""
        sms_message = (
            f"ALERT: {alert_data['type']}\n"
            f"{alert_data['message']}\n"
            f"Location: {alert_data['location']}\n"
            f"Time: {alert_data['timestamp']}"
        )
        
        for recipient in settings.ALERT_RECIPIENT_NUMBERS:
            if recipient:
                try:
                    self.twilio_client.messages.create(
                        body=sms_message,
                        from_=settings.TWILIO_PHONE_NUMBER,
                        to=recipient
                    )
                    logger.info(f"SMS alert sent to {recipient}")
                except Exception as e:
                    logger.error(f"Failed to send SMS alert to {recipient}: {str(e)}")
    
    def _send_dashboard_alert(self, alert_data):
        """Send alert to dashboard via MQTT"""
        try:
            # Convert alert data to JSON and publish to MQTT topic
            json_data = json.dumps(alert_data)
            self.mqtt_client.publish(settings.MQTT_TOPIC, json_data)
            logger.info(f"Dashboard alert published to {settings.MQTT_TOPIC}")
        except Exception as e:
            logger.error(f"Failed to publish dashboard alert: {str(e)}")
    
    def __del__(self):
        """Clean up resources on destruction"""
        if self.mqtt_client:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except:
                pass

# Create a global alert manager instance
alert_manager = AlertManager() 