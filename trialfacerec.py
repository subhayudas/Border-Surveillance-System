import cv2
import os
import numpy as np
from deepface import DeepFace
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import scipy.spatial as spatial
import argparse
import sys

class FaceRecognitionSystem:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.model_name = 'ArcFace'
        self.detector_backend = 'retinaface'  # Change to RetinaFace for better accuracy
        
        # Performance options 
        self.processing = False
        self.last_detection_time = 0
        self.detection_interval = 0.05  # Process more frequently (every 50ms)
        self.last_process_result = []
        self.frame_counter = 0
        self.skip_frames = 0  # Process every frame
        
        # Tracking for consistent recognition
        self.face_trackers = {}  # Store face trackers for each detected face
        self.next_track_id = 0
        self.tracking_timeout = 0.3  # Shorter timeout for more frequent detection
        
        # Identity consistency
        self.identity_history = {}  # Track IDs -> list of recent identifications
        self.identity_history_max = 10  # Store last 10 identifications for voting
        self.identity_vote_threshold = 0.4  # Lower threshold for better recall
        
        # Adaptive threshold settings - much lower for better recognition
        self.base_threshold = 0.45  # Lower base confidence threshold for better recall
        self.min_threshold = 0.38   # Very low minimum threshold to detect more matches
        self.max_threshold = 0.65   # Lower maximum threshold
        self.current_light_quality = 0.5  # Start with middle value
        
        # Debug options
        self.debug_output = False  # Only show detailed output in debug mode
        self.similarity_difference = 0.05  # Much lower difference threshold to accept more matches
        
        # Select appropriate tracker based on OpenCV version
        self.tracker_type = self._get_best_tracker()
        self._debug_print(f"Using {self.tracker_type} tracker")
        
        # Ensure dataset directory exists
        self._ensure_dataset_path()
        
        try:
            # Load DeepFace model
            self._debug_print(f"Loading face recognition model ({self.model_name})...")
            self.model = DeepFace.build_model(self.model_name)
            self._debug_print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading face model: {e}")
            print("Will continue with limited functionality")
            self.model = None
            
        # Dynamic threshold based on adaptive logic
        self.threshold = self.base_threshold  
        self.known_embeddings = {}  # Will store person_name -> embedding
        
        # Executor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Load face dataset
        self.load_dataset()

    def _ensure_dataset_path(self):
        """Make sure the dataset directory exists, create if not"""
        if not os.path.exists(self.dataset_path):
            print(f"Creating dataset directory: {self.dataset_path}")
            os.makedirs(self.dataset_path, exist_ok=True)
            
            # Create a README file with instructions
            readme_path = os.path.join(self.dataset_path, "README.txt")
            with open(readme_path, "w") as f:
                f.write("Face Recognition Dataset\n")
                f.write("======================\n\n")
                f.write("Instructions:\n")
                f.write("1. Create a folder for each person (e.g., 'John_Smith')\n")
                f.write("2. Add face images (.jpg, .jpeg, .png) of that person to their folder\n")
                f.write("3. Restart the application\n\n")
                f.write("The system will only recognize people who have folders with images in this directory.\n")
        else:
            print(f"Using existing dataset directory: {os.path.abspath(self.dataset_path)}")

    def load_dataset(self):
        """Load face embeddings from dataset directory"""
        print("Loading face recognition dataset...")
        person_embs = {}
        
        # Check if dataset directory exists
        if not os.path.exists(self.dataset_path):
            print(f"Dataset directory not found: {self.dataset_path}")
            return
            
        # Iterate through each person's directory
        for person_dir in Path(self.dataset_path).iterdir():
            if not person_dir.is_dir():
                continue
                
            name = person_dir.name
            embs = []
            
            # Get all image files in the person's directory
            image_files = [f for f in os.listdir(os.path.join(self.dataset_path, name)) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                print(f"No valid images found for {name}")
                continue
            
            print(f"Processing {len(image_files)} images for {name}...")
                
            for img_path in image_files:
                try:
                    full_path = os.path.join(self.dataset_path, name, img_path)
                    rep = DeepFace.represent(
                        img_path=full_path,
                        model_name=self.model_name,
                        detector_backend=self.detector_backend,
                        enforce_detection=False,
                        align=True
                    )
                    
                    # Handle both list and direct dictionary result formats
                    if isinstance(rep, list) and rep:
                        if 'embedding' in rep[0]:
                            embs.append(rep[0]['embedding'])
                    elif isinstance(rep, dict) and 'embedding' in rep:
                        embs.append(rep['embedding'])
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            # Only add if we have valid embeddings
            if embs:
                # Store individual embeddings instead of averaging
                person_embs[name] = embs
                print(f"Loaded {name} with {len(embs)} face embeddings")
            else:
                print(f"No valid faces found in images for {name}")
                
        self.known_embeddings = person_embs
        print(f"Total people loaded: {len(self.known_embeddings)}")
        
        if not self.known_embeddings:
            print("WARNING: No people were loaded into the dataset!")
            print(f"Check that your dataset path is correct: {os.path.abspath(self.dataset_path)}")
            print("And that it contains folders named after people with their face images inside")

    def _debug_print(self, message):
        """Print debug messages only if debug mode is enabled"""
        if self.debug_output:
            print(message)
        
    def _assess_face_quality(self, face_img):
        """Evaluate face image quality for adaptive thresholds (0-1 scale)"""
        try:
            # Calculate brightness (simple average)
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray) / 255.0
            
            # Calculate contrast
            contrast = np.std(gray) / 128.0
            
            # Calculate blur detection (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = np.var(laplacian) / 500.0  # Normalize
            blur_score = min(1.0, blur_score)  # Cap at 1.0
            
            # Calculate face size score (proportion of frame)
            face_size_score = min(1.0, (face_img.shape[0] * face_img.shape[1]) / (640 * 480))
            
            # Weighted quality score 
            quality = (0.3 * brightness * (1.0 if 0.2 <= brightness <= 0.8 else 0.5) + 
                       0.3 * min(1.0, contrast * 2) + 
                       0.2 * blur_score +
                       0.2 * face_size_score)
                       
            # Update adaptive threshold based on face quality
            self._adjust_threshold(quality)
            
            return quality
        except Exception as e:
            self._debug_print(f"Error assessing face quality: {e}")
            return 0.5  # Default to middle value on error
    
    def _adjust_threshold(self, quality):
        """Adjust recognition threshold based on image quality"""
        # Update current light quality (moving average)
        self.current_light_quality = 0.7 * self.current_light_quality + 0.3 * quality
        
        # Calculate new threshold - lower threshold for poorer conditions
        quality_factor = self.current_light_quality
        self.threshold = self.min_threshold + quality_factor * (self.max_threshold - self.min_threshold)
        
        self._debug_print(f"Quality: {quality:.2f}, Adjusted threshold: {self.threshold:.2f}")
    
    def _align_face(self, face_img, facial_area):
        """Align face for more consistent recognition"""
        try:
            # Get face region from image
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            face = face_img[y:y+h, x:x+w]
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Try to detect eyes
            eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_detector.detectMultiScale(gray, 1.1, 3)
            
            # Need at least two eyes for alignment
            if len(eyes) >= 2:
                # Sort eyes by x-position
                eyes = sorted(eyes, key=lambda x: x[0])
                
                # Get eye centers
                eye_1 = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
                eye_2 = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
                
                # Calculate angle
                dx = eye_2[0] - eye_1[0]
                dy = eye_2[1] - eye_1[1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Rotate to align eyes horizontally
                center = (w//2, h//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                aligned_face = cv2.warpAffine(face, M, (w, h))
                
                # Return aligned face
                return aligned_face
            
            # If eye detection fails, return original face
            return face
            
        except Exception as e:
            self._debug_print(f"Face alignment error: {e}")
            # If alignment fails, return original face region
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            return face_img[y:y+h, x:x+w]

    def find_closest_match(self, embedding):
        """Find the closest matching identity for the given face embedding"""
        if not self.known_embeddings:
            return "Unknown", 0.0
            
        # Initialize scores dictionary
        match_scores = {}
        
        # Check all people using multiple distance metrics
        for name, stored_embeddings in self.known_embeddings.items():
            person_scores = []
            
            for stored_embedding in stored_embeddings:
                # 1. Cosine similarity (higher is better)
                cosine_similarity = 1 - spatial.distance.cosine(embedding, stored_embedding)
                
                # 2. Euclidean distance (normalized to 0-1, higher is better)
                euclidean_dist = np.linalg.norm(np.array(embedding) - np.array(stored_embedding))
                euclidean_sim = 1 / (1 + euclidean_dist)
                
                # 3. Manhattan distance (normalized to 0-1, higher is better)
                manhattan_dist = np.sum(np.abs(np.array(embedding) - np.array(stored_embedding)))
                manhattan_sim = 1 / (1 + manhattan_dist)
                
                # 4. Correlation coefficient (higher is better)
                correlation = np.corrcoef(embedding, stored_embedding)[0, 1]
                # Handle NaN values
                if np.isnan(correlation):
                    correlation = 0.0
                correlation = (correlation + 1) / 2  # Scale from -1:1 to 0:1
                
                # Combined multi-metric score (weighted more heavily toward cosine similarity)
                combined_score = (0.7 * cosine_similarity + 
                                  0.15 * euclidean_sim + 
                                  0.05 * manhattan_sim +
                                  0.1 * correlation)
                
                # Apply cosine-similarity boost for high values
                if cosine_similarity > 0.85:
                    combined_score += 0.1  # Significant boost for very similar faces
                
                person_scores.append(combined_score)
                
                # Print detailed metrics for debugging
                self._debug_print(
                    f"Match: {name}, Score: {combined_score:.4f} "
                    f"(Cosine: {cosine_similarity:.4f}, Euclidean: {euclidean_sim:.4f}, "
                    f"Manhattan: {manhattan_sim:.4f}, Corr: {correlation:.4f})")
            
            # Use the best score for this person
            if person_scores:
                match_scores[name] = max(person_scores)
        
        # If no matches found
        if not match_scores:
            return "Unknown", 0.0
        
        # Find the best match
        matched_name, confidence = max(match_scores.items(), key=lambda x: x[1])
        
        # Apply threshold
        if confidence < self.threshold:
            self._debug_print(f"Best match {matched_name} rejected - confidence too low: {confidence:.4f} < {self.threshold:.4f}")
            return "Unknown", confidence
        
        # Find second best match (if any)
        second_best = 0.0
        second_name = "Unknown"
        
        for name, score in match_scores.items():
            if name != matched_name and score > second_best:
                second_best = score
                second_name = name
        
        # Check for ambiguous matches only if there's a close second match
        if second_best > 0.3:  # Only check if second match is reasonable
            difference = confidence - second_best
            self._debug_print(f"Best: {matched_name} ({confidence:.4f}), Second: {second_name} ({second_best:.4f}), Diff: {difference:.4f}")
            
            if difference < self.similarity_difference:
                self._debug_print(f"Match rejected - ambiguous result between {matched_name} and {second_name}")
                
                # If a face has history, use that to break the tie
                for track_id, tracker_info in self.face_trackers.items():
                    tx1, ty1, tx2, ty2 = tracker_info['bbox']
                    
                    # Check history for similar faces to help break the tie
                    if track_id in self.identity_history:
                        votes = {}
                        for past_name, _ in self.identity_history[track_id]:
                            if past_name not in votes:
                                votes[past_name] = 0
                            votes[past_name] += 1
                        
                        if votes:
                            most_common = max(votes.items(), key=lambda x: x[1])
                            if most_common[0] in [matched_name, second_name]:
                                # If history supports one of the top matches, use it
                                self._debug_print(f"Breaking tie using history in favor of {most_common[0]}")
                                return most_common[0], confidence
                
                # If we can't break the tie, consider it unknown
                return "Unknown", confidence
        
        self._debug_print(f"Match accepted: {matched_name} with confidence {confidence:.4f}")
        return matched_name, confidence

    def _update_identity_history(self, track_id, name, confidence):
        """Update identity history for temporal consistency"""
        if track_id not in self.identity_history:
            self.identity_history[track_id] = []
            
        # Add new identification with confidence
        self.identity_history[track_id].append((name, confidence))
        
        # Keep only recent history
        if len(self.identity_history[track_id]) > self.identity_history_max:
            self.identity_history[track_id].pop(0)
    
    def _get_consistent_identity(self, track_id):
        """Get temporally consistent identity through voting"""
        if track_id not in self.identity_history or not self.identity_history[track_id]:
            return "Unknown", 0.0
            
        # Count occurrences of each identity, weighted by confidence
        identity_votes = {}
        total_weight = 0.0
        
        for name, confidence in self.identity_history[track_id]:
            # Weight more recent identifications higher
            recency_weight = 1.0  # Could be adjusted based on position in history
            weight = confidence * recency_weight
            
            if name not in identity_votes:
                identity_votes[name] = 0.0
                
            identity_votes[name] += weight
            total_weight += weight
        
        if total_weight == 0:
            return "Unknown", 0.0
            
        # Normalize votes
        for name in identity_votes:
            identity_votes[name] /= total_weight
            
        # Get identity with highest votes
        best_name, vote_share = max(identity_votes.items(), key=lambda x: x[1])
        
        # Only return identity if it has sufficient votes
        if vote_share >= self.identity_vote_threshold:
            return best_name, vote_share
            
        return "Unknown", vote_share

    def _process_face_async(self, frame):
        """Process faces in a separate thread to avoid blocking UI"""
        try:
            current_time = time.time()
            
            # Skip if we're still processing or not enough time has passed since last detection
            if self.processing or (current_time - self.last_detection_time < self.detection_interval):
                return
                
            self.processing = True
            
            # Process in a separate thread
            self.executor.submit(self._detect_and_recognize, frame)
            
        except Exception as e:
            print(f"Error in async processing: {e}")
            self.processing = False

    def _detect_and_recognize(self, frame):
        """Perform face detection and recognition (runs in separate thread)"""
        try:
            faces = self._process_frame_internal(frame)
            
            # Update tracking with new detections
            self._update_face_tracking(frame, faces)
            
            # Store result for UI thread to display
            self.last_process_result = faces
            self.last_detection_time = time.time()
            
        except Exception as e:
            print(f"Error in detection thread: {e}")
        finally:
            self.processing = False

    def _update_face_tracking(self, frame, new_faces):
        """Update face tracking with new detections"""
        current_time = time.time()
        
        # Create/update trackers for new detections
        for face in new_faces:
            x1, y1, x2, y2 = face['bbox']
            name = face['name']
            confidence = face['confidence']
            quality = face.get('quality', 0.5)
            
            # Check if this face overlaps with any existing tracker
            matched_id = None
            for track_id, tracker_info in self.face_trackers.items():
                tx1, ty1, tx2, ty2 = tracker_info['bbox']
                
                # Check for significant overlap
                if self._calculate_iou((x1, y1, x2, y2), (tx1, ty1, tx2, ty2)) > 0.5:
                    matched_id = track_id
                    break
            
            if matched_id is not None:
                # Update existing tracker
                try:
                    tracker = self._create_tracker(frame, (x1, y1, x2-x1, y2-y1))
                    
                    # Update with new info
                    self.face_trackers[matched_id] = {
                        'tracker': tracker,
                        'bbox': (x1, y1, x2, y2),
                        'name': name,
                        'confidence': confidence,
                        'quality': quality,
                        'last_seen': current_time
                    }
                    
                    # Update identity history for voting
                    self._update_identity_history(matched_id, name, confidence)
                except Exception as e:
                    print(f"Error updating tracker: {e}")
            else:
                # Create new tracker
                try:
                    tracker = self._create_tracker(frame, (x1, y1, x2-x1, y2-y1))
                    
                    # Store tracker with new ID
                    self.face_trackers[self.next_track_id] = {
                        'tracker': tracker,
                        'bbox': (x1, y1, x2, y2),
                        'name': name,
                        'confidence': confidence,
                        'quality': quality,
                        'last_seen': current_time
                    }
                    
                    # Initialize identity history
                    self._update_identity_history(self.next_track_id, name, confidence)
                    
                    # Increment ID counter
                    self.next_track_id += 1
                except Exception as e:
                    print(f"Error creating tracker: {e}")
        
        # Remove expired trackers
        ids_to_remove = []
        for track_id, tracker_info in self.face_trackers.items():
            if current_time - tracker_info['last_seen'] > self.tracking_timeout:
                ids_to_remove.append(track_id)
                
        for track_id in ids_to_remove:
            if track_id in self.face_trackers:
                del self.face_trackers[track_id]
                
            # Clean up identity history
            if track_id in self.identity_history:
                del self.identity_history[track_id]

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        # Calculate IoU
        if union_area == 0:
            return 0
            
        return inter_area / union_area

    def _process_frame_internal(self, frame):
        """Internal implementation of face processing"""
        faces = []
        
        try:
            # Use DeepFace detection with RetinaFace backend
            results = DeepFace.represent(
                img_path=frame,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
            
            # Make sure we have a list of results
            if not isinstance(results, list):
                results = [results]

            # Process each detected face
            for result in results:
                if not isinstance(result, dict) or 'embedding' not in result or 'facial_area' not in result:
                    continue
                    
                embedding = result['embedding']
                region = result['facial_area']
                
                # Skip small regions
                if region['w'] < 20 or region['h'] < 20:
                    continue
                
                # Extract face for quality assessment
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                
                # Ensure region is within frame bounds
                if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                    continue
                    
                face_img = frame[y:y+h, x:x+w]
                
                # Skip if region is invalid
                if face_img.size == 0:
                    continue
                
                # Assess image quality for better matching
                quality = self._assess_face_quality(face_img)
                
                # Find closest matching identity with confidence
                name, confidence = self.find_closest_match(embedding)
                
                # Add face detection results
                face_data = {
                    'bbox': [x, y, x+w, y+h],
                    'name': name,
                    'confidence': confidence,
                    'quality': quality
                }
                faces.append(face_data)
            
            if faces:
                print(f"DeepFace detected {len(faces)} faces")
                
        except Exception as e:
            print(f"DeepFace detection failed: {e}")
        
        # If DeepFace failed, fall back to OpenCV
        if not faces:
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Use more conservative parameters
                detected_faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(detected_faces) > 0:
                    print(f"OpenCV detected {len(detected_faces)} faces")
                
                # For each detected face region, process it with DeepFace directly
                for (x, y, w, h) in detected_faces:
                    try:
                        # Create a copy of the face region
                        face_region = frame[y:y+h, x:x+w].copy()
                        
                        # Skip if region is invalid
                        if face_region.size == 0:
                            continue
                            
                        # Save face to temporary file (DeepFace works better with files than arrays)
                        temp_file = "temp_face.jpg"
                        cv2.imwrite(temp_file, face_region)
                        
                        # Get embedding using DeepFace
                        rep = DeepFace.represent(
                            img_path=temp_file,
                            model_name=self.model_name,
                            detector_backend=self.detector_backend,
                            enforce_detection=False,
                            align=True
                        )
                        
                        # Clean up temp file
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        
                        # Process embedding
                        if isinstance(rep, list) and rep and 'embedding' in rep[0]:
                            embedding = rep[0]['embedding']
                        elif isinstance(rep, dict) and 'embedding' in rep:
                            embedding = rep['embedding']
                        else:
                            continue
                            
                        # Assess quality
                        quality = self._assess_face_quality(face_region)
                        
                        # Find identity match
                        name, confidence = self.find_closest_match(embedding)
                        
                        # Add face data
                        face_data = {
                            'bbox': [x, y, x+w, y+h],
                            'name': name,
                            'confidence': confidence,
                            'quality': quality
                        }
                        faces.append(face_data)
                        
                    except Exception as e:
                        print(f"Error processing face region: {e}")
                        # Add face with Unknown identity
                        face_data = {
                            'bbox': [x, y, x+w, y+h],
                            'name': "Unknown",
                            'confidence': 0.4,
                            'quality': 0.5
                        }
                        faces.append(face_data)
                
            except Exception as e:
                print(f"OpenCV face detection failed: {e}")
            
        return faces

    def process_frame(self, frame):
        """Process frame for face recognition with tracking for better performance"""
        # Check if frame is valid
        if frame is None or frame.size == 0:
            print("Warning: Empty frame received")
            return []
            
        # Increment frame counter
        self.frame_counter += 1
        
        # Process frames more often to improve detection
        self.skip_frames = 0  # Process every frame
        
        # Process detection in a non-blocking way if frame counter matches
        if self.frame_counter % (self.skip_frames + 1) == 0:
            # Directly process frame for immediate results
            faces = self._process_frame_internal(frame)
            
            # Update tracking with new detections
            if faces:
                self._update_face_tracking(frame, faces)
                self.last_process_result = faces
                self.last_detection_time = time.time()
            
        # Update existing trackers with the current frame
        tracked_faces = []
        current_time = time.time()
        
        # List to collect trackers that need to be removed
        ids_to_remove = []
        
        # Use tracked information
        for track_id, tracker_info in list(self.face_trackers.items()):
            # Simple tracking - just use the last known position with slight decay
            x1, y1, x2, y2 = tracker_info['bbox']
            
            # Use consistent identity from history
            name, vote_confidence = self._get_consistent_identity(track_id)
            
            # Create face info
            face_data = {
                'bbox': [x1, y1, x2, y2],
                'name': name,
                'confidence': max(0, tracker_info['confidence'] - 0.01),  # Reduce confidence slowly
                'quality': tracker_info.get('quality', 0.5)
            }
            
            # Add to results if confidence is still acceptable
            if face_data['confidence'] > 0.3:
                tracked_faces.append(face_data)
            else:
                ids_to_remove.append(track_id)
            
            # Mark as expired if too old
            if current_time - tracker_info['last_seen'] > self.tracking_timeout:
                ids_to_remove.append(track_id)
        
        # Remove failed trackers
        for track_id in ids_to_remove:
            if track_id in self.face_trackers:
                del self.face_trackers[track_id]
                
            # Clean up identity history
            if track_id in self.identity_history:
                del self.identity_history[track_id]
        
        # If no tracked faces, just use the last detection result
        if not tracked_faces and self.last_process_result:
            return self.last_process_result
        
        # Always use OpenCV face detection as a backup (more reliable)
        if not tracked_faces:
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
                
                for (x, y, w, h) in detected_faces:
                    # Create a face record
                    face_data = {
                        'bbox': [x, y, x+w, y+h],
                        'name': "Unknown",
                        'confidence': 0.4,
                        'quality': 0.5
                    }
                    tracked_faces.append(face_data)
                    
                if tracked_faces:
                    print("Used OpenCV face detection")
            except Exception as e:
                print(f"Error in fallback face detection: {e}")
            
        return tracked_faces

    def _get_best_tracker(self):
        """Get the best available tracker based on OpenCV version"""
        # Use Simple tracking by default to avoid errors
        print("Using simple tracking for better reliability")
        return "Simple"
        
    def _create_tracker(self, frame, bbox):
        """Create a tracker appropriate for the OpenCV version"""
        # Just return simple tracking to avoid errors
        return "Simple"

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--dataset', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--threshold', type=float, default=0.75, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--difference', type=float, default=0.15, help='Required difference between best and second best match')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    print(f"Starting face recognition with threshold={args.threshold}, difference={args.difference}")
    
    # Create face recognition system with custom parameters
    face_system = FaceRecognitionSystem(dataset_path=args.dataset)
    
    # Update thresholds based on command-line args
    face_system.threshold = args.threshold
    face_system.similarity_difference = args.difference
    face_system.debug_output = args.debug
    
    # Access the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)
        
    # Create a resizable window
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    
    # Initialize variables for FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    # Variables for adding faces
    add_face_mode = False
    adding_face_name = ""
    captured_frames = []
    
    # Main loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
            
        # Flip the frame horizontally for more natural interaction
        frame = cv2.flip(frame, 1)
        
        # Process the frame for faces
        faces = face_system.process_frame(frame)
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Draw face detection results
        for face in faces:
            name = face['name']
            confidence = face['confidence']
            x1, y1, x2, y2 = face['bbox']
            
            # Choose color based on confidence and identity
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
            else:
                # Scale from yellow to green based on confidence (higher = greener)
                green = int(255 * min(1.0, confidence))
                red = int(255 * (1.0 - min(1.0, confidence/0.9)))
                color = (0, green, red)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare text with confidence
            confidence_text = f"{int(confidence * 100)}%"
            if name != "Unknown":
                display_text = f"{name} ({confidence_text})"
            else:
                display_text = f"Unknown ({confidence_text})"
                
            # Draw a better-looking label with background
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_size[0] + 10, y1), color, -1)
            cv2.putText(frame, display_text, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw confidence bar
            bar_width = x2 - x1
            filled_width = int(bar_width * confidence)
            cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 15), (100, 100, 100), -1)
            cv2.rectangle(frame, (x1, y2 + 5), (x1 + filled_width, y2 + 15), color, -1)
        
        # Add face mode UI
        if add_face_mode:
            # Draw overlay indicating we're in add face mode
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 140, 255), -1)
            cv2.putText(frame, f"Adding face: {adding_face_name} - Capturing {len(captured_frames)}/5", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display FPS and help text
        cv2.putText(frame, f"FPS: {fps}", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'a' to add face, 'q' to quit", (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow("Face Recognition", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' to exit
        if key == ord('q'):
            break
            
        # 'a' to add a face
        if key == ord('a') and not add_face_mode:
            adding_face_name = input("Enter name for the new face: ")
            if adding_face_name:
                add_face_mode = True
                captured_frames = []
                print(f"Adding face for {adding_face_name}. Press 's' to capture 5 frames.")
                
        # 's' to capture a frame in add face mode
        if key == ord('s') and add_face_mode:
            if len(captured_frames) < 5:
                captured_frames.append(frame.copy())
                print(f"Captured frame {len(captured_frames)}/5")
                
                if len(captured_frames) == 5:
                    # Create directory for the new face
                    face_dir = os.path.join(face_system.dataset_path, adding_face_name)
                    os.makedirs(face_dir, exist_ok=True)
                    
                    # Save frames
                    for i, img in enumerate(captured_frames):
                        img_path = os.path.join(face_dir, f"{adding_face_name}_{i+1}.jpg")
                        cv2.imwrite(img_path, img)
                    
                    print(f"Added 5 images for {adding_face_name}. Reloading dataset...")
                    face_system.load_dataset()
                    add_face_mode = False
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


