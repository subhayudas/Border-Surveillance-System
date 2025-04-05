import os
import cv2
import numpy as np
import time
try:
    from ultralytics import YOLO
except ImportError:
    # Handle the import error gracefully
    YOLO = None
    print("Warning: ultralytics import failed, using dummy YOLO class")
    class YOLO:
        """Dummy YOLO class for when ultralytics is not available"""
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return []
        def to(self, *args, **kwargs):
            return self

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
        self.threshold = settings.DETECTION_THRESHOLD  # General threshold
        # Person detection needs a lower threshold for better recall
        self.person_threshold = max(0.25, self.threshold - 0.2)  # Even lower threshold for person
        
        # Dictionary of class-specific thresholds
        self.class_thresholds = {
            'person': self.person_threshold,
            'car': max(0.3, self.threshold - 0.1),
            'truck': max(0.3, self.threshold - 0.1),
            'motorcycle': max(0.35, self.threshold - 0.1),
            'bicycle': max(0.35, self.threshold - 0.1)
        }
        
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
        # Run YOLOv8 inference on the frame with the lowest threshold
        # to ensure we catch all potential objects, especially people
        min_threshold = min(self.class_thresholds.values())
        results = self.model(frame, conf=min_threshold)
        
        # Extract predictions
        detections = []
        
        # Process each detection with class-specific thresholds
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates, confidence and class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # Apply class-specific threshold
                # For person class, use a lower threshold to improve recall
                # For other classes, use the base threshold
                class_threshold = self.class_thresholds.get(class_name, self.threshold)
                
                # Only keep classes of interest with sufficient confidence
                if class_name in settings.CLASSES_OF_INTEREST and confidence >= class_threshold:
                    # For person class, boost the confidence a bit to emphasize it
                    if class_name == 'person':
                        confidence = min(confidence * 1.05, 1.0)
                    
                    # Add detection to list
                    detections.append([
                        int(x1), int(y1), int(x2), int(y2), 
                        confidence, class_id, class_name
                    ])
        
        # Special filtering for person class to reduce false negatives
        person_detections = [d for d in detections if d[6] == 'person']
        
        # If we have no person detections but should expect some (e.g., in a surveillance context),
        # try running with an even lower threshold as a fallback
        if not person_detections and settings.EXPECT_PEOPLE:
            # Run with very low threshold just for person class
            fallback_results = self.model(frame, conf=0.15)
            for result in fallback_results:
                boxes = result.boxes
                for box in boxes:
                    # Get coordinates, confidence and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Only add person class from this fallback pass
                    if class_name == 'person' and confidence >= 0.15:
                        # Check if this is a reasonable size for a person
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = height / width if width > 0 else 0
                        
                        # Typical person has aspect ratio > 1.5 (taller than wide)
                        if aspect_ratio > 1.5 and height > 30:  # Minimum size check
                            detections.append([
                                int(x1), int(y1), int(x2), int(y2), 
                                confidence, class_id, class_name
                            ])
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        
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

class WeaponDetector:
    """Specialized detector for various types of weapons using a fine-tuned model"""
    
    def __init__(self):
        # Load specialized weapon detection model (if available)
        weapon_model_path = os.path.join(settings.MODELS_DIR, "weapons_detector.pt")
        if os.path.exists(weapon_model_path):
            logger.info(f"Loading specialized weapon detection model from {weapon_model_path}")
            self.model = YOLO(weapon_model_path)
        else:
            # Fall back to standard model if specialized model not found
            logger.info("Specialized weapon model not found, falling back to standard model")
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
        logger.info(f"Weapon detector using device: {self.device}")
        self.model.to(self.device)
        
        # Lower threshold for weapons to increase detection rate
        # But not too low to prevent false positives
        self.threshold = max(0.35, settings.DETECTION_THRESHOLD - 0.1)
        
        # Define weapon class mapping for standard YOLOv8 model
        # This is used as a fallback if specialized model isn't available
        self.standard_model_class_map = {
            # Map standard COCO classes to our weapon categories
            43: 'knife',     # knife in COCO
            # Removed person (0) from weapon mappings to prevent confusion
        }
        
        # Flag to determine if we're using the specialized model
        self.using_specialized_model = os.path.exists(weapon_model_path)
        
        # Define specialized weapon classes
        self.weapon_classes = [
            'rifle',
            'bazooka',
            'shotgun', 
            'handgun',
            'knife',
            'grenade',
            'weapon'  # Generic weapon class
        ]
        
        logger.info("Weapon detector initialized")
    
    def detect(self, frame):
        """
        Detect weapons in a frame using specialized detector
        
        Args:
            frame: OpenCV image (numpy array)
            
        Returns:
            list: List of weapon detections in format [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        # First run the standard object detector for people with high confidence
        # to avoid classifying people as weapons
        try:
            # Create a temporary detector to find people with high confidence
            standard_detector = ObjectDetector()
            # Only keep person class with high confidence for reference
            person_detections = []
            all_detections = standard_detector.detect(frame)
            for det in all_detections:
                _, _, _, _, conf, _, class_name = det
                if class_name == 'person' and conf > 0.6:  # High confidence people
                    person_detections.append(det)
        except Exception as e:
            logger.warning(f"Error detecting people for reference: {e}")
            person_detections = []
        
        # Run YOLOv8 inference on the frame for weapons
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
                
                # Get class name based on which model we're using
                if self.using_specialized_model:
                    # Our specialized model has the weapon classes directly
                    if class_id < len(self.weapon_classes):
                        class_name = self.weapon_classes[class_id]
                    else:
                        class_name = 'weapon'  # Default if out of range
                else:
                    # Using standard model, map COCO classes to weapon classes
                    class_name = self.standard_model_class_map.get(class_id, None)
                    
                    # Skip if not a weapon class
                    if class_name is None:
                        continue
                    
                    # Skip person class from standard model to prevent misclassification
                    if class_id == 0:  # person in COCO
                        continue
                        
                    # Boost confidence for certain classes we know are reliable
                    if class_name == 'knife':
                        confidence = min(confidence * 1.3, 1.0)  # Boost knife confidence
                
                # Skip very small detections which are likely to be false positives
                width = x2 - x1
                height = y2 - y1
                if width < 10 or height < 10:
                    continue
                
                # Check if this weapon detection substantially overlaps with a person
                # If so, it might be a valid weapon being held
                is_valid_weapon = True
                for person in person_detections:
                    px1, py1, px2, py2, _, _, _ = person
                    # Calculate overlap between weapon and person
                    overlap_x1 = max(x1, px1)
                    overlap_y1 = max(y1, py1)
                    overlap_x2 = min(x2, px2)
                    overlap_y2 = min(y2, py2)
                    
                    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                        # Calculate overlap area and weapon area
                        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                        weapon_area = (x2 - x1) * (y2 - y1)
                        
                        # If weapon is mostly inside person and takes up significant portion
                        # of person width, it's very likely a false positive (person's arm/leg)
                        # except for expected positions of carried weapons
                        if (overlap_area / weapon_area > 0.85 and 
                            width > 0.4 * (px2 - px1) and
                            class_name != 'knife'):  # Knives can be small
                            
                            # Extra check if it's in a position where weapons are commonly held
                            # Middle torso or hands for handguns, side for rifles
                            weapon_center_x = (x1 + x2) / 2
                            weapon_center_y = (y1 + y2) / 2
                            person_center_x = (px1 + px2) / 2
                            person_center_y = (py1 + py2) / 2
                            
                            # Weapon in middle or top of person is more likely valid
                            if (weapon_center_y < person_center_y and 
                                abs(weapon_center_x - person_center_x) < (px2 - px1) * 0.3):
                                is_valid_weapon = True
                            else:
                                # Likely false positive
                                is_valid_weapon = False
                                break
                
                if not is_valid_weapon:
                    continue
                
                # Boost overall weapon confidence for specialized model but cap it
                if self.using_specialized_model:
                    confidence = min(confidence * 1.1, 1.0)  # Boost by 10%
                
                # Add to detections
                detections.append([
                    int(x1), int(y1), int(x2), int(y2), 
                    confidence, class_id, class_name
                ])
        
        # Apply secondary weapon-specific filtering
        filtered_detections = self._filter_weapon_detections(detections, frame)
        
        return filtered_detections
    
    def _filter_weapon_detections(self, detections, frame):
        """Filter weapon detections to reduce false positives"""
        if not detections:
            return []
            
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        
        # Remove overlapping detections of the same class
        filtered = []
        used_areas = []
        
        for detection in detections:
            x1, y1, x2, y2, conf, class_id, class_name = detection
            
            # Apply higher threshold for generic weapon class
            if class_name == 'weapon' and conf < 0.55:
                continue
                
            # Minimum confidence based on weapon type
            min_confidence = {
                'handgun': 0.45,
                'rifle': 0.42,
                'knife': 0.5,
                'grenade': 0.55,
                'bazooka': 0.48,
                'shotgun': 0.45
            }.get(class_name, 0.55)
            
            if conf < min_confidence:
                continue
            
            # Check if this detection overlaps significantly with a higher confidence one
            overlaps = False
            for used_x1, used_y1, used_x2, used_y2, used_class in used_areas:
                if used_class != class_name:
                    continue  # Different class, so don't filter
                    
                # Calculate intersection
                inter_x1 = max(x1, used_x1)
                inter_y1 = max(y1, used_y1)
                inter_x2 = min(x2, used_x2)
                inter_y2 = min(y2, used_y2)
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    # Calculate overlap ratio
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    current_area = (x2 - x1) * (y2 - y1)
                    overlap_ratio = inter_area / current_area
                    
                    if overlap_ratio > 0.7:  # 70% overlap
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(detection)
                used_areas.append((x1, y1, x2, y2, class_name))
        
        return filtered

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