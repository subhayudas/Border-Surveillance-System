import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

from config import settings
from utils.logger import logger
from utils.alerting import alert_manager

class ObjectDetector:
    """YOLOv8-based object detection for border surveillance"""
    
    def __init__(self):
        # Load YOLO model
        model_path = os.path.join(settings.MODELS_DIR, "yolov8n.pt")
        if not os.path.exists(model_path):
            logger.info(f"Model not found at {model_path}, downloading YOLOv8n...")
            self.model = YOLO("yolov8n.pt")
            # Save the model for future use
            if not os.path.exists(settings.MODELS_DIR):
                os.makedirs(settings.MODELS_DIR)
            self.model.save(model_path)
        else:
            logger.info(f"Loading model from {model_path}")
            self.model = YOLO(model_path)
        
        # Set device
        self.device = "cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Set detection threshold
        self.threshold = settings.DETECTION_THRESHOLD
        
        # Initialize trackers for behavior analysis
        self.tracked_objects = {}  # Format: {id: {first_seen: timestamp, positions: [positions], etc}}
    
    def detect(self, frame):
        """
        Detect objects in a frame
        
        Args:
            frame: OpenCV image (numpy array)
            
        Returns:
            list: List of detections in format [x1, y1, x2, y2, confidence, class_id]
        """
        # Run YOLOv8 inference on the frame
        results = self.model(frame, conf=self.threshold)
        
        # Extract predictions
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates, confidence and class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # Only keep classes of interest
                if class_name in settings.CLASSES_OF_INTEREST:
                    detections.append([int(x1), int(y1), int(x2), int(y2), confidence, class_id, class_name])
        
        return detections

class BehaviorAnalyzer:
    """Analyzes object behaviors to detect suspicious activities"""
    
    def __init__(self):
        self.tracked_objects = {}  # Format: {id: {first_seen: timestamp, positions: [(x,y)], etc}}
        self.next_id = 0
        self.loitering_alerts = set()  # IDs of objects already alerted for loitering
    
    def update(self, detections, frame):
        """
        Update tracking for behavior analysis
        
        Args:
            detections: List of detections [x1, y1, x2, y2, conf, class_id, class_name]
            frame: Current video frame
            
        Returns:
            list: Alerts triggered by behaviors
        """
        current_time = time.time()
        frame_height, frame_width = frame.shape[:2]
        alerts = []
        
        # Match detections with tracked objects or create new tracks
        matched_ids = set()
        
        for detection in detections:
            x1, y1, x2, y2, conf, class_id, class_name = detection
            
            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Calculate box dimensions
            width = x2 - x1
            height = y2 - y1
            
            # Try to match with existing tracked objects
            matched = False
            best_match_id = None
            best_match_distance = float('inf')
            
            for obj_id, obj_data in self.tracked_objects.items():
                if obj_data['class_name'] != class_name:
                    continue
                
                if len(obj_data['positions']) == 0:
                    continue
                    
                # Get the last known position
                last_x, last_y = obj_data['positions'][-1]
                
                # Calculate Euclidean distance
                distance = ((last_x - center_x) ** 2 + (last_y - center_y) ** 2) ** 0.5
                
                # If distance is below threshold, consider it a match
                if distance < 50 and distance < best_match_distance:  # 50 pixels threshold
                    best_match_id = obj_id
                    best_match_distance = distance
                    matched = True
            
            if matched:
                # Update existing tracked object
                obj_id = best_match_id
                self.tracked_objects[obj_id]['positions'].append((center_x, center_y))
                self.tracked_objects[obj_id]['last_seen'] = current_time
                self.tracked_objects[obj_id]['bbox'] = (x1, y1, x2, y2)
                self.tracked_objects[obj_id]['confidence'] = conf
                matched_ids.add(obj_id)
            else:
                # Create new tracked object
                self.tracked_objects[self.next_id] = {
                    'class_name': class_name,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'positions': [(center_x, center_y)],
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                }
                matched_ids.add(self.next_id)
                self.next_id += 1
        
        # Remove old tracks
        ids_to_remove = []
        for obj_id in self.tracked_objects:
            if current_time - self.tracked_objects[obj_id]['last_seen'] > 5.0:  # 5 seconds timeout
                ids_to_remove.append(obj_id)
        
        for obj_id in ids_to_remove:
            self.tracked_objects.pop(obj_id, None)
        
        # Analyze behaviors
        for obj_id in matched_ids:
            obj_data = self.tracked_objects[obj_id]
            
            # Check for loitering
            if obj_data['class_name'] == 'person' and obj_id not in self.loitering_alerts:
                duration = current_time - obj_data['first_seen']
                if duration > settings.SUSPICIOUS_BEHAVIORS['loitering']['time_threshold']:
                    # Calculate the bounding area of all positions
                    positions = obj_data['positions']
                    if len(positions) > 10:  # Need enough points for meaningful analysis
                        x_coords = [p[0] for p in positions]
                        y_coords = [p[1] for p in positions]
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)
                        
                        # Calculate area ratio compared to whole frame
                        area_ratio = ((max_x - min_x) * (max_y - min_y)) / (frame_width * frame_height)
                        
                        if area_ratio < settings.SUSPICIOUS_BEHAVIORS['loitering']['area_threshold']:
                            # Person has been in a small area for too long
                            alert = {
                                'type': 'loitering',
                                'message': f"Person loitering detected for {int(duration)} seconds",
                                'confidence': obj_data['confidence'],
                                'bbox': obj_data['bbox']
                            }
                            alerts.append(alert)
                            self.loitering_alerts.add(obj_id)  # Mark as alerted
            
            # Check for crawling behavior
            if obj_data['class_name'] == 'person':
                x1, y1, x2, y2 = obj_data['bbox']
                height = y2 - y1
                width = x2 - x1
                height_width_ratio = height / width if width > 0 else 0
                
                if height_width_ratio < settings.SUSPICIOUS_BEHAVIORS['crawling']['height_ratio_threshold']:
                    alert = {
                        'type': 'crawling',
                        'message': "Person crawling detected",
                        'confidence': obj_data['confidence'],
                        'bbox': obj_data['bbox']
                    }
                    alerts.append(alert)
        
        return alerts

class FenceTamperingDetector:
    """Detects tampering with fences and perimeter barriers"""
    
    def __init__(self):
        self.previous_frame = None
        self.fence_masks = []
        
        # Create fence region masks from settings
        for fence_region in settings.FENCE_REGIONS:
            if fence_region:
                mask = np.zeros((settings.FRAME_HEIGHT, settings.FRAME_WIDTH), dtype=np.uint8)
                points = np.array(fence_region, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [points], 255)
                self.fence_masks.append(mask)
    
    def detect(self, frame):
        """
        Detect tampering with fence regions
        
        Args:
            frame: Current video frame
            
        Returns:
            list: Fence tampering alerts if detected
        """
        if not self.fence_masks:
            return []  # No fence regions defined
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return []
        
        # Calculate absolute difference between current and previous frame
        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate the thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        alerts = []
        
        # Check each fence region for significant changes
        for i, mask in enumerate(self.fence_masks):
            # Apply mask to get changes only in fence region
            masked_thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
            
            # Calculate the percentage of changed pixels in the fence region
            non_zero = cv2.countNonZero(masked_thresh)
            total = cv2.countNonZero(mask)
            change_percent = non_zero / total if total > 0 else 0
            
            if change_percent > settings.TAMPERING_SENSITIVITY:
                alert = {
                    'type': 'fence_tampering',
                    'message': f"Fence tampering detected in region {i+1}",
                    'confidence': change_percent,
                    'region_id': i
                }
                alerts.append(alert)
        
        # Update previous frame
        self.previous_frame = gray
        
        return alerts

class ItemDetector:
    """Specialized detector for items like cell phones and bags"""
    
    def __init__(self):
        # Load YOLO model - we'll use the same model but focus on item classes
        model_path = os.path.join(settings.MODELS_DIR, "yolov8n.pt")
        if not os.path.exists(model_path):
            logger.info(f"Model not found at {model_path}, downloading YOLOv8n...")
            self.model = YOLO("yolov8n.pt")
            # Save the model for future use
            if not os.path.exists(settings.MODELS_DIR):
                os.makedirs(settings.MODELS_DIR)
            self.model.save(model_path)
        else:
            logger.info(f"Loading model from {model_path}")
            self.model = YOLO(model_path)
        
        # Set device
        self.device = "cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Set threshold for detection
        self.threshold = settings.DETECTION_THRESHOLD - 0.1
        
        # Define item classes from COCO dataset
        # COCO class indices: cell phone = 67, backpack = 27, handbag = 31, suitcase = 33
        self.item_classes = {
            'cell phone': 67,     # cell phone in COCO
            'backpack': 27,       # backpack in COCO
            'handbag': 31,        # handbag in COCO
            'suitcase': 33        # suitcase in COCO
        }
        
        logger.info("Item detector initialized for cell phones and bags detection")
    
    def detect(self, frame):
        """
        Detect items (cell phones and bags) in a frame
        
        Args:
            frame: OpenCV image (numpy array)
            
        Returns:
            list: List of item detections in format [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        detections = []
        
        # YOLO detection
        results = self.model(frame, conf=self.threshold)
        
        # Extract predictions
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates, confidence and class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Check if this is an item class we're interested in
                item_name = None
                for name, coco_id in self.item_classes.items():
                    if class_id == coco_id:
                        item_name = name
                        break
                
                if item_name:
                    detections.append([
                        int(x1), int(y1), int(x2), int(y2), 
                        confidence, class_id, item_name
                    ])
        
        return detections

# Add after the ItemDetector class

class BorderCrossingDetector:
    """Detects unauthorized border crossings"""
    
    def __init__(self):
        # Initialize border line definitions
        self.border_lines = []
        
        # Load border lines from settings
        if hasattr(settings, 'BORDER_LINES') and settings.BORDER_LINES:
            self.border_lines = settings.BORDER_LINES
        else:
            # Default border line (horizontal line in the middle of the frame)
            self.border_lines = [
                {
                    'id': 'default',
                    'points': [(0, settings.FRAME_HEIGHT // 2), 
                              (settings.FRAME_WIDTH, settings.FRAME_HEIGHT // 2)],
                    'direction': 'both'  # 'north_to_south', 'south_to_north', or 'both'
                }
            ]
        
        # Track objects that have crossed borders
        self.tracked_objects = {}  # {object_id: {'position': (x,y), 'crossed': False, 'direction': None}}
        self.next_id = 0
        
        # Store recent crossings to avoid duplicate alerts
        self.recent_crossings = {}  # {object_id: timestamp}
        self.crossing_cooldown = 5  # seconds
        
        logger.info(f"Border crossing detector initialized with {len(self.border_lines)} border lines")
    
    def detect(self, detections, frame):
        """
        Detect border crossings based on object movements
        
        Args:
            detections: List of detections [x1, y1, x2, y2, confidence, class_id, class_name]
            frame: Current video frame
            
        Returns:
            list: Border crossing alerts if detected
        """
        current_time = time.time()
        frame_height, frame_width = frame.shape[:2]
        
        # Current detections by ID for tracking
        current_detections = {}
        
        # Process each detection (focus on people, vehicles)
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id, class_name = detection
            
            # Only track people and vehicles for border crossing
            if class_name not in ['person', 'car', 'truck', 'motorcycle', 'bicycle']:
                continue
                
            # Calculate center point of the object
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Try to match with existing tracked objects
            matched_id = None
            
            for obj_id, obj_data in self.tracked_objects.items():
                prev_x, prev_y = obj_data['position']
                
                # Simple distance-based matching
                distance = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
                
                if distance < 50:  # Threshold for matching
                    matched_id = obj_id
                    break
            
            # If no match found, create new tracked object
            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1
                self.tracked_objects[matched_id] = {
                    'position': (center_x, center_y),
                    'crossed': False,
                    'direction': None,
                    'class': class_name,
                    'first_seen': current_time
                }
            
            # Update the tracked object's position
            current_detections[matched_id] = {
                'position': (center_x, center_y),
                'crossed': self.tracked_objects[matched_id]['crossed'],
                'direction': self.tracked_objects[matched_id]['direction'],
                'class': class_name,
                'first_seen': self.tracked_objects[matched_id]['first_seen']
            }
        
        # Check for border crossings
        alerts = []
        
        for obj_id, obj_data in current_detections.items():
            current_pos = obj_data['position']
            
            # Skip if this object has recently triggered an alert
            if obj_id in self.recent_crossings and current_time - self.recent_crossings[obj_id] < self.crossing_cooldown:
                continue
            
            # Get previous position if available
            prev_pos = self.tracked_objects.get(obj_id, {}).get('position')
            
            if prev_pos:
                # Check each border line
                for border in self.border_lines:
                    border_points = border['points']
                    border_direction = border.get('direction', 'both')
                    
                    # Check if the object crossed this border
                    if self._line_crossing(prev_pos, current_pos, border_points):
                        # Determine crossing direction
                        crossing_direction = self._determine_crossing_direction(prev_pos, current_pos, border_points)
                        
                        # Check if this direction should trigger an alert
                        if (border_direction == 'both' or 
                            (border_direction == 'north_to_south' and crossing_direction == 'north_to_south') or
                            (border_direction == 'south_to_north' and crossing_direction == 'south_to_north')):
                            
                            # Create alert
                            alert = {
                                'type': 'border_crossing',
                                'message': f"{obj_data['class'].capitalize()} crossed border {border.get('id', 'default')} ({crossing_direction})",
                                'confidence': 0.9,
                                'bbox': [x1, y1, x2, y2] if 'x1' in locals() else None,
                                'position': current_pos,
                                'border_id': border.get('id', 'default'),
                                'direction': crossing_direction,
                                'object_class': obj_data['class'],
                                'critical': True
                            }
                            alerts.append(alert)
                            
                            # Update object status
                            current_detections[obj_id]['crossed'] = True
                            current_detections[obj_id]['direction'] = crossing_direction
                            
                            # Add to recent crossings
                            self.recent_crossings[obj_id] = current_time
        
        # Update tracked objects
        self.tracked_objects = current_detections
        
        # Clean up old tracked objects
        self._cleanup_old_tracks(current_time)
        
        return alerts
    
    def _line_crossing(self, p1, p2, line):
        """Check if a moving object (from p1 to p2) crosses a line"""
        x1, y1 = p1
        x2, y2 = p2
        line_x1, line_y1 = line[0]
        line_x2, line_y2 = line[1]
        
        # Line intersection formula
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        
        a = (x1, y1)
        b = (x2, y2)
        c = (line_x1, line_y1)
        d = (line_x2, line_y2)
        
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)
    
    def _determine_crossing_direction(self, p1, p2, line):
        """Determine the direction of crossing (north_to_south or south_to_north)"""
        x1, y1 = p1
        x2, y2 = p2
        line_x1, line_y1 = line[0]
        line_x2, line_y2 = line[1]
        
        # For a horizontal line
        if abs(line_y1 - line_y2) < abs(line_x1 - line_x2):
            if y2 > y1:
                return "north_to_south"
            else:
                return "south_to_north"
        # For a vertical line
        else:
            if x2 > x1:
                return "west_to_east"
            else:
                return "east_to_west"
    
    def _cleanup_old_tracks(self, current_time, max_age=10):
        """Remove tracks that haven't been updated recently"""
        ids_to_remove = []
        
        for obj_id, obj_data in self.tracked_objects.items():
            if current_time - obj_data.get('first_seen', 0) > max_age:
                ids_to_remove.append(obj_id)
        
        for obj_id in ids_to_remove:
            del self.tracked_objects[obj_id]
        
        # Also clean up recent crossings
        crossings_to_remove = []
        for obj_id, timestamp in self.recent_crossings.items():
            if current_time - timestamp > self.crossing_cooldown:
                crossings_to_remove.append(obj_id)
        
        for obj_id in crossings_to_remove:
            del self.recent_crossings[obj_id]