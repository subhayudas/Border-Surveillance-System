import cv2
import numpy as np
import time
from datetime import datetime

class Visualizer:
    """Handles visualization of detections and alerts on video frames"""
    
    def __init__(self):
        self.colors = {
            # People and vehicles
            'person': (0, 255, 0),      # Green
            'car': (0, 165, 255),       # Orange
            'truck': (0, 0, 255),       # Red
            'motorcycle': (255, 0, 0),  # Blue
            'bicycle': (255, 0, 255),   # Purple
            
            # Items
            'drone': (255, 255, 0),     # Cyan
            'backpack': (128, 0, 128),  # Purple
            'suitcase': (165, 42, 42),  # Brown
            'handbag': (255, 128, 0),   # Light blue
            'umbrella': (0, 128, 128),  # Teal
            'cell phone': (0, 0, 128),  # Dark blue
            
            # Weapons
            'rifle': (0, 0, 255),       # Red
            'bazooka': (0, 0, 150),     # Dark red
            'shotgun': (0, 50, 255),    # Red-blue
            'handgun': (0, 100, 255),   # Blue-red
            'knife': (255, 0, 100),     # Purple-red
            'grenade': (0, 50, 50),     # Dark teal
            'weapon': (100, 0, 100),    # Dark purple
            'gun': (0, 100, 255),       # For compatibility with older detections
            
            # Alerts
            'fence_tampering': (0, 0, 255),     # Red
            'loitering': (0, 255, 255),         # Yellow
            'crawling': (255, 0, 255),          # Magenta
            'abandoned_item': (255, 0, 0),      # Red
            'weapon_rifle': (0, 0, 255),        # Red
            'weapon_bazooka': (0, 0, 150),      # Dark red
            'weapon_shotgun': (0, 50, 255),     # Red-blue
            'weapon_handgun': (0, 100, 255),    # Blue-red
            'weapon_knife': (255, 0, 100),      # Purple-red
            'weapon_grenade': (0, 50, 50),      # Dark teal
            'weapon_weapon': (100, 0, 100),     # Dark purple
            'default': (200, 200, 200)          # Gray
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        
        # Track FPS
        self.prev_frame_time = 0
        self.fps = 0
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels for detections
        
        Args:
            frame: OpenCV image to draw on
            detections: List of [x1, y1, x2, y2, confidence, class_id, class_name]
            
        Returns:
            frame: The image with detections drawn
        """
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id, class_name = detection
            
            # Get color for this class
            color = self.colors.get(class_name, self.colors['default'])
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label with class name and confidence
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate label background size
            (label_width, label_height), _ = cv2.getTextSize(
                label, self.font, self.font_scale, self.font_thickness
            )
            
            # Draw label background
            cv2.rectangle(
                frame, 
                (x1, y1 - label_height - 5), 
                (x1 + label_width + 5, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, 
                label, 
                (x1 + 3, y1 - 4), 
                self.font, 
                self.font_scale, 
                (255, 255, 255), 
                self.font_thickness
            )
        
        return frame
    
    def draw_alerts(self, frame, alerts):
        """
        Draw alert indicators on the frame
        
        Args:
            frame: OpenCV image to draw on
            alerts: List of alert dictionaries
            
        Returns:
            frame: The image with alerts drawn
        """
        for alert in alerts:
            alert_type = alert['type']
            bbox = alert.get('bbox', None)
            
            # Get color for this alert type
            color = self.colors.get(alert_type, self.colors['default'])
            
            # If we have a bounding box, draw it with thicker lines
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Add flashing effect for active alerts by drawing a second rectangle
                if int(time.time() * 2) % 2 == 0:  # Flash twice per second
                    cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), color, 2)
                
                # Draw alert type
                label_bg_height = 30
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_bg_height),
                    (x2, y1),
                    color,
                    -1
                )
                
                # Format alert text - for weapon alerts, extract the weapon type
                if alert_type.startswith('weapon_'):
                    weapon_type = alert_type.split('_')[1].upper()
                    alert_text = f"WEAPON: {weapon_type}"
                else:
                    alert_text = alert_type.upper()
                
                # Calculate text size and position
                text_size, _ = cv2.getTextSize(
                    alert_text, self.font, self.font_scale * 1.2, self.font_thickness * 2
                )
                text_x = x1 + (x2 - x1 - text_size[0]) // 2
                text_y = y1 - 10
                
                cv2.putText(
                    frame,
                    alert_text,
                    (text_x, text_y),
                    self.font,
                    self.font_scale * 1.2,
                    (255, 255, 255),
                    self.font_thickness * 2
                )
                
                # Special handling for weapons
                if alert_type.startswith('weapon_'):
                    # Draw a danger icon or warning pattern around the weapon
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    radius = max(x2 - x1, y2 - y1) // 2
                    
                    # Draw danger icon (concentric circles in warning color)
                    cv2.circle(frame, (center_x, center_y), radius, color, 2)
                    cv2.circle(frame, (center_x, center_y), radius * 3 // 4, color, 1)
                    cv2.circle(frame, (center_x, center_y), radius // 2, color, 1)
                    
                    # Draw X in the center
                    cv2.line(frame, 
                            (center_x - radius//3, center_y - radius//3),
                            (center_x + radius//3, center_y + radius//3),
                            color, 2)
                    cv2.line(frame, 
                            (center_x - radius//3, center_y + radius//3),
                            (center_x + radius//3, center_y - radius//3),
                            color, 2)
                
                # Special handling for drones - draw warning zone
                elif alert_type == 'drone':
                    # Draw a warning circle around the drone
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    radius = max(x2 - x1, y2 - y1)
                    cv2.circle(frame, (center_x, center_y), radius, color, 2)
                    cv2.circle(frame, (center_x, center_y), radius + 10, color, 1)
                
                # Special handling for abandoned items
                elif alert_type == 'abandoned_item':
                    # Draw a dashed warning box around the item
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    radius = max(x2 - x1, y2 - y1) // 2
                    for i in range(0, 360, 30):  # Draw dashed line
                        angle = i * np.pi / 180
                        start_x = int(center_x + (radius + 5) * np.cos(angle))
                        start_y = int(center_y + (radius + 5) * np.sin(angle))
                        end_x = int(center_x + (radius + 5) * np.cos(angle + np.pi/6))
                        end_y = int(center_y + (radius + 5) * np.sin(angle + np.pi/6))
                        cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
        
        return frame
    
    def add_info_overlay(self, frame, num_detections=0):
        """
        Add information overlay with timestamp, FPS, etc.
        
        Args:
            frame: OpenCV image to draw on
            num_detections: Number of detections in the frame
            
        Returns:
            frame: The image with info overlay
        """
        # Calculate FPS
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = current_time
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Draw black background for overlay
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, height - 40), (width, height), (0, 0, 0), -1)
        
        # Draw timestamp
        cv2.putText(
            frame,
            timestamp,
            (10, height - 15),
            self.font,
            self.font_scale * 1.2,
            (255, 255, 255),
            self.font_thickness
        )
        
        # Draw FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (width - 120, height - 15),
            self.font,
            self.font_scale * 1.2,
            (255, 255, 255),
            self.font_thickness
        )
        
        # Draw detection count
        count_text = f"Detections: {num_detections}"
        cv2.putText(
            frame,
            count_text,
            (width - 250, height - 15),
            self.font,
            self.font_scale * 1.2,
            (255, 255, 255),
            self.font_thickness
        )
        
        return frame
    
    def draw_fence_regions(self, frame, fence_regions):
        """
        Draw fence region outlines
        
        Args:
            frame: OpenCV image to draw on
            fence_regions: List of polygon coordinates
            
        Returns:
            frame: The image with fence regions drawn
        """
        for i, region in enumerate(fence_regions):
            if region:
                points = np.array(region, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], True, (0, 255, 255), 2)
                
                # Add region label
                centroid_x = sum(p[0] for p in region) // len(region)
                centroid_y = sum(p[1] for p in region) // len(region)
                
                cv2.putText(
                    frame,
                    f"Fence {i+1}",
                    (centroid_x, centroid_y),
                    self.font,
                    self.font_scale,
                    (0, 255, 255),
                    self.font_thickness
                )
        
        return frame
    
    def create_snapshot(self, frame, alert=None):
        """
        Create a snapshot image with alert details for saving
        
        Args:
            frame: OpenCV image to create snapshot from
            alert: Alert information to include
            
        Returns:
            snapshot: Annotated snapshot image
        """
        snapshot = frame.copy()
        
        # Add timestamp and large alert banner
        if alert:
            # Add red banner at the top
            height, width = snapshot.shape[:2]
            cv2.rectangle(snapshot, (0, 0), (width, 60), (0, 0, 255), -1)
            
            # Add alert text
            cv2.putText(
                snapshot,
                f"ALERT: {alert['type'].upper()} - {alert['message']}",
                (10, 40),
                self.font,
                1.2,
                (255, 255, 255),
                2
            )
            
            # Add timestamp at the bottom
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                snapshot,
                timestamp,
                (10, height - 10),
                self.font,
                0.8,
                (255, 255, 255),
                1
            )
        
        return snapshot
    
    def draw_border_lines(self, frame, border_lines):
        """
        Draw border lines on the frame
        
        Args:
            frame: OpenCV image to draw on
            border_lines: List of border line definitions
            
        Returns:
            frame: The image with border lines drawn
        """
        for border in border_lines:
            points = border['points']
            border_id = border.get('id', 'default')
            direction = border.get('direction', 'both')
            
            # Draw the border line
            cv2.line(
                frame,
                (points[0][0], points[0][1]),
                (points[1][0], points[1][1]),
                (0, 0, 255),  # Red color for borders
                2,
                cv2.LINE_AA
            )
            
            # Add border label
            mid_x = (points[0][0] + points[1][0]) // 2
            mid_y = (points[0][1] + points[1][1]) // 2
            
            # Add a small offset to the label position
            label_offset_x = 5
            label_offset_y = -10
            
            # Create label text
            label = f"Border: {border_id}"
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(
                label, self.font, self.font_scale, self.font_thickness
            )
            
            cv2.rectangle(
                frame,
                (mid_x + label_offset_x, mid_y + label_offset_y - label_height - 5),
                (mid_x + label_offset_x + label_width + 5, mid_y + label_offset_y),
                (0, 0, 255),
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (mid_x + label_offset_x + 3, mid_y + label_offset_y - 4),
                self.font,
                self.font_scale,
                (255, 255, 255),
                self.font_thickness
            )
            
            # Draw direction indicators
            if direction == 'north_to_south' or direction == 'both':
                # Draw arrow pointing south
                cv2.arrowedLine(
                    frame,
                    (mid_x + 20, mid_y - 10),
                    (mid_x + 20, mid_y + 10),
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                    tipLength=0.3
                )
            
            if direction == 'south_to_north' or direction == 'both':
                # Draw arrow pointing north
                cv2.arrowedLine(
                    frame,
                    (mid_x - 20, mid_y + 10),
                    (mid_x - 20, mid_y - 10),
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                    tipLength=0.3
                )
        
        return frame