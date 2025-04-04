#!/usr/bin/env python3
"""
Border Surveillance System - Main Application

This is the main entry point for the border surveillance system that:
1. Captures video from camera or file
2. Detects objects using AI models
3. Analyzes behaviors and detects fence tampering
4. Generates alerts for suspicious activities
5. Displays and optionally records the processed video
"""

import os
import sys
import cv2
import time
import argparse
import threading
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.detection import ObjectDetector, BehaviorAnalyzer, FenceTamperingDetector
from src.visualization import Visualizer
from utils.logger import logger, SurveillanceLogger
from utils.alerting import alert_manager

class SurveillanceSystem:
    """Main surveillance system class that coordinates all components"""
    
    def __init__(self, video_source=None, output_path=None, show_display=True):
        """
        Initialize the surveillance system
        
        Args:
            video_source: Camera index or video file path
            output_path: Path to save output video
            show_display: Whether to show live display
        """
        # Create output directory if needed
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Create snapshots directory
        self.snapshots_dir = os.path.join(self.output_dir, "snapshots")
        if not os.path.exists(self.snapshots_dir):
            os.makedirs(self.snapshots_dir)
        
        # Initialize video source
        self.video_source = video_source if video_source is not None else settings.VIDEO_SOURCE
        self.cap = self._init_video_capture(self.video_source)
        
        # Initialize video writer if output path provided
        self.output_path = output_path
        self.writer = None
        
        # Initialize detection and analysis components
        self.object_detector = ObjectDetector()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.fence_tampering_detector = FenceTamperingDetector()
        self.visualizer = Visualizer()
        
        # Set up display option
        self.show_display = show_display
        
        # Initialize logging
        self.logger_instance = SurveillanceLogger()
        
        # Initialize system state
        self.running = False
        self.frame_count = 0
        self.last_snapshot_time = 0
        
        logger.info("Surveillance system initialized successfully")
    
    def _init_video_capture(self, source):
        """Initialize video capture from source"""
        try:
            # Try to interpret source as an integer (camera index)
            cap = cv2.VideoCapture(int(source))
        except (ValueError, TypeError):
            # If not an integer, treat as a file path
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, settings.FPS)
        
        return cap
    
    def _init_video_writer(self, frame_size):
        """Initialize video writer with timestamp-based filename"""
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            if os.path.isdir(self.output_path):
                output_file = os.path.join(self.output_path, f"surveillance_{timestamp}.avi")
            else:
                output_file = self.output_path
            
            self.writer = cv2.VideoWriter(
                output_file,
                fourcc,
                settings.FPS,
                frame_size
            )
            logger.info(f"Recording video to: {output_file}")
    
    def _save_snapshot(self, frame, alert):
        """Save a snapshot image when an alert is triggered"""
        current_time = time.time()
        
        # Rate limit snapshots (max 1 per second)
        if current_time - self.last_snapshot_time < 1.0:
            return
        
        # Create annotated snapshot
        snapshot = self.visualizer.create_snapshot(frame, alert)
        
        # Save to file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        alert_type = alert['type'].replace(' ', '_')
        filename = f"{alert_type}_{timestamp}.jpg"
        filepath = os.path.join(self.snapshots_dir, filename)
        
        cv2.imwrite(filepath, snapshot)
        logger.info(f"Saved alert snapshot to {filepath}")
        
        # Update snapshot timestamp
        self.last_snapshot_time = current_time
        
        return filepath
    
    def _process_frame(self, frame):
        """Process a single frame through the detection and analysis pipeline"""
        # Make a working copy of the frame
        processed_frame = frame.copy()
        
        # Object detection
        detections = self.object_detector.detect(frame)
        
        # Behavior analysis
        behavior_alerts = self.behavior_analyzer.update(detections, frame)
        
        # Fence tampering detection
        tampering_alerts = self.fence_tampering_detector.detect(frame)
        
        # Combine all alerts
        all_alerts = behavior_alerts + tampering_alerts
        
        # Visualize results
        processed_frame = self.visualizer.draw_detections(processed_frame, detections)
        processed_frame = self.visualizer.draw_alerts(processed_frame, all_alerts)
        processed_frame = self.visualizer.draw_fence_regions(processed_frame, settings.FENCE_REGIONS)
        processed_frame = self.visualizer.add_info_overlay(processed_frame, len(detections))
        
        # Process alerts (send notifications, save snapshots)
        for alert in all_alerts:
            # Save snapshot
            snapshot_path = self._save_snapshot(frame, alert)
            
            # Get GPS coordinates if available
            location = None
            if settings.GPS_ENABLED:
                location = (settings.DEFAULT_LAT, settings.DEFAULT_LON)
            
            # Send alert notification
            alert_manager.send_alert(
                alert_type=alert['type'],
                message=alert['message'],
                location=location,
                image_path=snapshot_path,
                confidence=alert.get('confidence', 0.0)
            )
            
            # Log the alert
            self.logger_instance.log_alert(alert['type'], alert['message'])
        
        return processed_frame, detections, all_alerts
    
    def run(self):
        """Main processing loop"""
        self.running = True
        logger.info("Starting surveillance system")
        
        # Wait for first frame to initialize writer
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from video source")
            return
        
        # Initialize video writer if needed
        if self.output_path and self.writer is None:
            self._init_video_writer(frame.shape[:2][::-1])  # (width, height)
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("End of video stream reached")
                    break
                
                # Process frame
                processed_frame, detections, alerts = self._process_frame(frame)
                
                # Write to output if enabled
                if self.writer:
                    self.writer.write(processed_frame)
                
                # Display if enabled
                if self.show_display:
                    cv2.imshow("Border Surveillance", processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            logger.info("Surveillance system interrupted by user")
        
        except Exception as e:
            logger.error(f"Error in surveillance system: {str(e)}")
        
        finally:
            # Clean up resources
            if self.cap:
                self.cap.release()
            
            if self.writer:
                self.writer.release()
            
            if self.show_display:
                cv2.destroyAllWindows()
            
            logger.info(f"Surveillance system stopped after processing {self.frame_count} frames")
    
    def stop(self):
        """Stop the surveillance system"""
        self.running = False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Border Surveillance System")
    
    parser.add_argument(
        "-s", "--source",
        help="Video source (camera index or file path)",
        default=settings.VIDEO_SOURCE
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output video file or directory",
        default=None
    )
    
    parser.add_argument(
        "--no-display",
        help="Disable the video display window",
        action="store_true"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and run the surveillance system
    system = SurveillanceSystem(
        video_source=args.source,
        output_path=args.output,
        show_display=not args.no_display
    )
    
    system.run() 