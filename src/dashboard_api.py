#!/usr/bin/env python3
"""
Border Surveillance System - Dashboard API Server

Provides a REST API and websocket server for the surveillance dashboard
"""

import os
import sys
import json
import time
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import paho.mqtt.client as mqtt
from typing import List, Dict, Any, Optional
import threading
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from utils.logger import logger

# Initialize FastAPI app
app = FastAPI(
    title="Border Surveillance API",
    description="API server for the Border Surveillance System",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize alert storage
alerts = []
alert_lock = threading.Lock()
MAX_ALERTS = 1000  # Maximum number of alerts to store

# Websocket connections
active_connections: List[WebSocket] = []

# MQTT client for receiving alerts
mqtt_client = None

class AlertManager:
    """Manages alert storage and distribution"""
    
    def add_alert(self, alert_data: Dict[str, Any]) -> None:
        """Add a new alert to the storage"""
        global alerts
        
        with alert_lock:
            # Add timestamp if not present
            if "timestamp" not in alert_data:
                alert_data["timestamp"] = datetime.now().isoformat()
            
            # Add alert ID if not present
            if "id" not in alert_data:
                alert_data["id"] = str(int(time.time() * 1000))
            
            # Add alert to storage, limit size
            alerts.append(alert_data)
            if len(alerts) > MAX_ALERTS:
                alerts = alerts[-MAX_ALERTS:]
        
        # Notify all connected websocket clients
        asyncio.create_task(broadcast_alert(alert_data))

alert_manager = AlertManager()

async def broadcast_alert(alert: Dict[str, Any]) -> None:
    """Send alert to all connected websocket clients"""
    for connection in active_connections:
        try:
            await connection.send_json(alert)
        except Exception as e:
            logger.error(f"Error broadcasting alert: {str(e)}")

def on_mqtt_message(client, userdata, message):
    """Handle incoming MQTT messages"""
    try:
        payload = message.payload.decode("utf-8")
        alert_data = json.loads(payload)
        alert_manager.add_alert(alert_data)
        logger.info(f"Received alert via MQTT: {alert_data.get('type', 'unknown')}")
    except Exception as e:
        logger.error(f"Error processing MQTT message: {str(e)}")

def setup_mqtt():
    """Initialize MQTT client for alert subscription"""
    global mqtt_client
    
    try:
        mqtt_client = mqtt.Client()
        mqtt_client.on_message = on_mqtt_message
        mqtt_client.connect(settings.MQTT_BROKER, settings.MQTT_PORT, 60)
        mqtt_client.subscribe(settings.MQTT_TOPIC)
        mqtt_client.loop_start()
        logger.info(f"MQTT client connected to {settings.MQTT_BROKER} and subscribed to {settings.MQTT_TOPIC}")
    except Exception as e:
        logger.error(f"Failed to initialize MQTT client: {str(e)}")
        mqtt_client = None

# API routes
@app.get("/")
async def root():
    """API root endpoint"""
    return {"message": "Border Surveillance System API"}

@app.get("/api/alerts")
async def get_alerts(
    limit: int = 100, 
    offset: int = 0,
    alert_type: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    """Get alert history with optional filtering"""
    global alerts
    
    with alert_lock:
        filtered_alerts = alerts
        
        # Filter by alert type if specified
        if alert_type:
            filtered_alerts = [a for a in filtered_alerts if a.get("type") == alert_type]
        
        # Filter by time range if specified
        if start_time:
            try:
                start = datetime.fromisoformat(start_time)
                filtered_alerts = [
                    a for a in filtered_alerts 
                    if datetime.fromisoformat(a.get("timestamp", "")) >= start
                ]
            except (ValueError, TypeError):
                pass
        
        if end_time:
            try:
                end = datetime.fromisoformat(end_time)
                filtered_alerts = [
                    a for a in filtered_alerts 
                    if datetime.fromisoformat(a.get("timestamp", "")) <= end
                ]
            except (ValueError, TypeError):
                pass
        
        # Apply pagination
        paginated = filtered_alerts[offset:offset+limit]
        
        return {
            "total": len(filtered_alerts),
            "alerts": paginated
        }

@app.get("/api/alerts/recent")
async def get_recent_alerts(limit: int = 10):
    """Get the most recent alerts"""
    global alerts
    
    with alert_lock:
        recent = alerts[-limit:] if alerts else []
        return recent

@app.get("/api/alerts/{alert_id}")
async def get_alert(alert_id: str):
    """Get a specific alert by ID"""
    global alerts
    
    with alert_lock:
        for alert in alerts:
            if alert.get("id") == alert_id:
                return alert
    
    raise HTTPException(status_code=404, detail="Alert not found")

@app.get("/api/snapshots/{filename}")
async def get_snapshot(filename: str):
    """Serve snapshot image files"""
    snapshots_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
        "snapshots"
    )
    
    file_path = os.path.join(snapshots_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Snapshot not found")
    
    return FileResponse(file_path)

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    global alerts
    
    with alert_lock:
        # Count alerts by type
        alert_counts = {}
        for alert in alerts:
            alert_type = alert.get("type", "unknown")
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        
        # Get alerts for the last 24 hours
        now = datetime.now()
        day_ago = now - timedelta(days=1)
        
        alerts_24h = [
            a for a in alerts
            if datetime.fromisoformat(a.get("timestamp", "")) >= day_ago
        ]
        
        return {
            "total_alerts": len(alerts),
            "alerts_by_type": alert_counts,
            "alerts_last_24h": len(alerts_24h),
            "system_uptime": "Unknown",  # Would come from main system
            "active_connections": len(active_connections)
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Websocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send the most recent alerts on connection
        with alert_lock:
            recent_alerts = alerts[-10:] if alerts else []
        
        for alert in recent_alerts:
            await websocket.send_json(alert)
        
        # Keep connection alive and handle incoming messages
        while True:
            # We don't expect client messages, but need to keep the connection alive
            await websocket.receive_text()
    
    except Exception as e:
        logger.info(f"Websocket connection closed: {str(e)}")
    
    finally:
        # Remove connection on disconnect
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    setup_mqtt()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

def run_server(host="0.0.0.0", port=8000):
    """Run the API server"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server() 