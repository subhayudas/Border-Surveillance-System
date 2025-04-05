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
# Update the imports at the top of the file
from src.detection import ObjectDetector, BehaviorAnalyzer, FenceTamperingDetector, BorderCrossingDetector, WeaponDetector
from src.visualization import Visualizer, UI
from utils.logger import logger, SurveillanceLogger
from utils.alerting import alert_manager

class SurveillanceSystem:
    """Main surveillance system class that coordinates all components"""
    
    # Then update the SurveillanceSystem.__init__ method to add the border crossing detector
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
        self.weapon_detector = WeaponDetector()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.fence_tampering_detector = FenceTamperingDetector()
        self.border_crossing_detector = BorderCrossingDetector()
        self.visualizer = Visualizer()
        
        # Set up display option
        self.show_display = show_display
        
        # Initialize UI if display is enabled
        self.ui = UI(self.change_video_source) if show_display else None
        
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
        
        # Object detection - general objects
        general_detections = self.object_detector.detect(frame)
        
        # Weapon detection - specialized detector
        weapon_detections = self.weapon_detector.detect(frame)
        
        # Combine detections from both detectors
        # Prefer weapon detector results for weapon classes
        weapon_classes = ['rifle', 'bazooka', 'shotgun', 'handgun', 'knife', 'grenade', 'weapon', 'gun']
        weapon_bboxes = []
        
        # Get all weapon bounding boxes for overlap filtering
        for det in weapon_detections:
            x1, y1, x2, y2 = det[:4]
            weapon_bboxes.append((x1, y1, x2, y2))
        
        # Filter general detections to remove overlapping with weapon detections
        filtered_general_detections = []
        for det in general_detections:
            x1, y1, x2, y2, conf, class_id, class_name = det
            
            # Skip if this is a weapon class from general detector (prefer specialized)
            if class_name in weapon_classes:
                continue
                
            # Check if this detection overlaps with any weapon detection
            overlap = False
            for wx1, wy1, wx2, wy2 in weapon_bboxes:
                # Check for overlap
                if (x1 < wx2 and x2 > wx1 and y1 < wy2 and y2 > wy1):
                    # Calculate overlap area
                    x_overlap = min(x2, wx2) - max(x1, wx1)
                    y_overlap = min(y2, wy2) - max(y1, wy1)
                    overlap_area = x_overlap * y_overlap
                    det_area = (x2 - x1) * (y2 - y1)
                    
                    # If significant overlap, skip this general detection
                    if overlap_area > 0.5 * det_area:
                        overlap = True
                        break
            
            if not overlap:
                filtered_general_detections.append(det)
        
        # Combine detections
        all_detections = filtered_general_detections + weapon_detections
        
        # Behavior analysis
        behavior_alerts = self.behavior_analyzer.update(all_detections, frame)
        
        # Fence tampering detection
        tampering_alerts = self.fence_tampering_detector.detect(frame)
        
        # Border crossing detection
        border_crossing_alerts = self.border_crossing_detector.detect(all_detections, frame)
        
        # Generate weapon alerts
        weapon_alerts = self._generate_weapon_alerts(weapon_detections)
        
        # Combine all alerts
        all_alerts = behavior_alerts + tampering_alerts + weapon_alerts + border_crossing_alerts
        
        # Visualize results
        processed_frame = self.visualizer.draw_detections(processed_frame, all_detections)
        processed_frame = self.visualizer.draw_alerts(processed_frame, all_alerts)
        processed_frame = self.visualizer.draw_fence_regions(processed_frame, settings.FENCE_REGIONS)
        processed_frame = self.visualizer.draw_border_lines(processed_frame, settings.BORDER_LINES)
        processed_frame = self.visualizer.add_info_overlay(processed_frame, len(all_detections))
        
        # Add UI elements if enabled
        if self.show_display and self.ui:
            processed_frame = self.ui.draw_buttons(processed_frame)
        
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
        
        return processed_frame, all_detections, all_alerts
    
    def _generate_weapon_alerts(self, weapon_detections):
        """Generate alerts for detected weapons"""
        alerts = []
        
        # Alert thresholds for different weapon types
        threshold_modifiers = {
            'rifle': 0.45,
            'bazooka': 0.4,  # Lower threshold for more dangerous weapons
            'shotgun': 0.45,
            'handgun': 0.5,
            'knife': 0.55,
            'grenade': 0.4,
            'weapon': 0.6
        }
        
        # Process each weapon detection
        for detection in weapon_detections:
            x1, y1, x2, y2, confidence, class_id, class_name = detection
            
            # Get threshold for this weapon type
            base_threshold = settings.DETECTION_THRESHOLD
            weapon_threshold = base_threshold * threshold_modifiers.get(class_name, 1.0)
            
            # Create alert if confidence exceeds the threshold
            if confidence >= weapon_threshold:
                alert = {
                    'type': f'weapon_{class_name}',
                    'message': f"Detected {class_name.upper()} with confidence {confidence:.2f}",
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'class_name': class_name
                }
                alerts.append(alert)
        
        return alerts
    
    # Then update the run method to include border crossing detection
    def run(self):
        """Main processing loop"""
        self.running = True
        logger.info("Starting surveillance system")
        
        # Create a named window for mouse callbacks if we're showing the display
        if self.show_display:
            cv2.namedWindow("Border Surveillance")
            if self.ui:
                cv2.setMouseCallback("Border Surveillance", self.ui.handle_mouse_event)
        
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
                    # If we're using a file source, try to loop back to beginning
                    if not isinstance(self.video_source, int) and self.ui and self.ui.video_source_type == "file":
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to start
                        ret, frame = self.cap.read()
                        if not ret:
                            break  # Really can't read, so break
                    else:
                        break
                
                # Process frame using our main processing pipeline
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
                    elif key == ord('s'):  # 's' key to toggle video source
                        if self.ui:
                            self.ui.toggle_video_source()
                
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

    def change_video_source(self, new_source):
        """Change the video source while the system is running"""
        logger.info(f"Changing video source to: {new_source}")
        
        # Close the current video source
        if self.cap is not None:
            self.cap.release()
        
        # Initialize the new video source
        self.video_source = new_source
        self.cap = self._init_video_capture(new_source)
        
        # Reset tracking variables
        self.frame_count = 0
        self.last_snapshot_time = 0
        
        # Reset all detectors to ensure clean state
        self.object_detector = ObjectDetector()
        self.weapon_detector = WeaponDetector()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.fence_tampering_detector = FenceTamperingDetector()
        self.border_crossing_detector = BorderCrossingDetector()
        
        # Reset any video writer if it exists
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            
        logger.info(f"Video source changed successfully to {new_source}")


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