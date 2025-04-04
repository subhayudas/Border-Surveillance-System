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
                
                # Adjust confidence based on the class for more accurate classification
                adjusted_confidence = confidence
                
                # Boost confidence for non-person objects to prevent misclassification
                if class_name in ["backpack", "drone", "suitcase", "handbag"]:
                    adjusted_confidence = min(confidence * 1.15, 1.0)  # Boost by 15% but cap at 1.0
                
                # Only keep classes of interest with sufficient confidence
                if class_name in settings.CLASSES_OF_INTEREST and adjusted_confidence >= self.threshold:
                    # Store the adjusted confidence for display
                    detections.append([int(x1), int(y1), int(x2), int(y2), adjusted_confidence, class_id, class_name])
        
        # Apply additional logic to resolve overlapping detections
        # If a backpack/drone/suitcase significantly overlaps with a person, it may be being carried
        filtered_detections = self._filter_overlapping_detections(detections)
        
        return filtered_detections
        
    def _filter_overlapping_detections(self, detections):
        """
        Filter out cases where objects are misidentified
        
        Args:
            detections: List of [x1, y1, x2, y2, confidence, class_id, class_name]
            
        Returns:
            list: Filtered list of detections
        """
        if len(detections) <= 1:
            return detections
            
        filtered = []
        items_to_check = []
        
        # First pass - separate persons from items
        for detection in detections:
            if detection[6] in ["backpack", "drone", "suitcase", "handbag", "umbrella", "cell phone"]:
                items_to_check.append(detection)
            else:
                filtered.append(detection)
        
        # Second pass - check if items are overlapping with persons
        for item in items_to_check:
            item_x1, item_y1, item_x2, item_y2 = item[0:4]
            item_area = (item_x2 - item_x1) * (item_y2 - item_y1)
            is_overlapping = False
            
            for person in filtered:
                if person[6] != "person":
                    continue
                    
                person_x1, person_y1, person_x2, person_y2 = person[0:4]
                
                # Calculate intersection
                x_left = max(item_x1, person_x1)
                y_top = max(item_y1, person_y1)
                x_right = min(item_x2, person_x2)
                y_bottom = min(item_y2, person_y2)
                
                # Check if there is an intersection
                if x_right < x_left or y_bottom < y_top:
                    continue
                    
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                overlap_ratio = intersection_area / item_area
                
                # If more than 70% of the item is inside the person, it's being carried
                if overlap_ratio > 0.7:
                    is_overlapping = True
                    break
            
            # If the item is not significantly overlapping with a person, add it
            if not is_overlapping:
                filtered.append(item)
                
        return filtered

class BehaviorAnalyzer:
    """Analyzes object behaviors to detect suspicious activities"""
    
    def __init__(self):
        self.tracked_objects = {}  # Format: {id: {first_seen: timestamp, positions: [(x,y)], etc}}
        self.next_id = 0
        self.loitering_alerts = set()  # IDs of objects already alerted for loitering
        self.drone_alerts = set()  # IDs of drones already alerted
    
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
            
            # Check for drone presence
            if obj_data['class_name'] == 'drone' and obj_id not in self.drone_alerts:
                # Create an alert for drone detected
                alert = {
                    'type': 'drone',
                    'message': "Suspicious drone detected",
                    'confidence': obj_data['confidence'],
                    'bbox': obj_data['bbox']
                }
                alerts.append(alert)
                self.drone_alerts.add(obj_id)  # Mark as alerted
                
            # Check for abandoned items (backpack, suitcase, etc.)
            if obj_data['class_name'] in ['backpack', 'suitcase', 'handbag'] and len(obj_data['positions']) > 15:
                # Check if the item hasn't moved significantly
                positions = obj_data['positions']
                x_coords = [p[0] for p in positions[-15:]]  # Last 15 positions
                y_coords = [p[1] for p in positions[-15:]]
                
                x_movement = max(x_coords) - min(x_coords)
                y_movement = max(y_coords) - min(y_coords)
                
                if x_movement < 10 and y_movement < 10:  # Very little movement
                    alert = {
                        'type': 'abandoned_item',
                        'message': f"Abandoned {obj_data['class_name']} detected",
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