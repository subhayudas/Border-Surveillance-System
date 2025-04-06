import os
import time
import logging
import pickle
import cv2
import numpy as np
import glob
import shutil

logger = logging.getLogger('BorderSurveillance')

class FaceRecognizer:
    """Face recognition module that only uses faces from user-provided dataset"""
    
    def __init__(self, dataset_path=None, model_path=None, encoding_file=None, tolerance=0.5):
        """Initialize the face recognition system"""
        self.dataset_path = dataset_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faces")
        self.model_loaded = False
        self.tolerance = tolerance
        self.mock_encoding_file = os.path.join(os.path.dirname(self.dataset_path), "user_face_data.pkl")
        
        # Create necessary directories if they don't exist
        os.makedirs(self.dataset_path, exist_ok=True)
        
        # Storage for face tracking
        self.known_face_names = []
        self.persistent_faces = {}  # Store persistent face tracking between frames
        
        try:
            # Try to import face recognition libraries
            import face_recognition
            import dlib
            import imutils
            
            self.face_recognition = face_recognition
            self.model_loaded = True
            
            # Load face encodings if available
            self._load_encodings()
            
            logger.info("Face recognition module loaded successfully")
        except ImportError:
            logger.warning("Face recognition libraries not available. Install with: pip install face-recognition dlib imutils")
            self._load_mock_data()
            
        # Load all person names from dataset folders
        self._load_dataset_names()
        
        # Create dataset structure if empty
        if not self.known_face_names:
            logger.warning("No face dataset found. Creating dataset directory structure.")
            self._create_dataset_structure()
    
    def _create_dataset_structure(self):
        """Create dataset directory structure for user to add their own face images"""
        # Create a README file explaining how to add faces
        readme_path = os.path.join(self.dataset_path, "README.txt")
        with open(readme_path, "w") as f:
            f.write("Face Recognition Dataset\n")
            f.write("=======================\n\n")
            f.write("To add faces for recognition:\n")
            f.write("1. Create a folder with the person's name (e.g., 'John_Smith')\n")
            f.write("2. Add clear face images of that person to their folder\n")
            f.write("3. Restart the application\n\n")
            f.write("The system will only recognize people who have folders with images in this directory.\n")
        
        # Create an example folder structure
        example_dir = os.path.join(self.dataset_path, "Example_Person")
        os.makedirs(example_dir, exist_ok=True)
        
        logger.info(f"Created dataset structure at {self.dataset_path}")
        logger.info(f"Please add face images to this directory to enable recognition")
    
    def _load_dataset_names(self):
        """Load person names from dataset folder structure"""
        # Get all subdirectories in the dataset path (each is a person)
        if os.path.exists(self.dataset_path):
            subdirs = [d for d in os.listdir(self.dataset_path) 
                      if os.path.isdir(os.path.join(self.dataset_path, d))]
            
            # Filter out directories that might be system folders
            person_dirs = [d for d in subdirs if not d.startswith('.') and d != 'Example_Person']
            
            for person in person_dirs:
                if person not in self.known_face_names:
                    self.known_face_names.append(person)
            
            if person_dirs:
                logger.info(f"Loaded {len(person_dirs)} people from dataset")

    def _load_encodings(self):
        """Load face encodings for known faces if using real face recognition"""
        if not hasattr(self, 'face_recognition'):
            return
            
        self.known_encodings = []
        self.known_face_names = []
        
        # Check if we have a saved encodings file and it's newer than any dataset changes
        dataset_mtime = 0
        if os.path.exists(self.dataset_path):
            dataset_mtime = os.path.getmtime(self.dataset_path)
        
        encodings_file = self.mock_encoding_file
        
        if os.path.exists(encodings_file) and os.path.getmtime(encodings_file) > dataset_mtime:
            # Load pre-computed encodings
            with open(encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data.get('encodings', [])
                self.known_face_names = data.get('names', [])
            logger.info(f"Loaded {len(self.known_encodings)} face encodings from cache")
            return
            
        # No cached encodings or dataset has changed, compute fresh encodings
        logger.info("Computing face encodings from dataset...")
        
        # Process each person's directory
        for person_dir in glob.glob(os.path.join(self.dataset_path, "*")):
            if not os.path.isdir(person_dir) or os.path.basename(person_dir).startswith('.'):
                continue
                
            person_name = os.path.basename(person_dir)
            if person_name == 'Example_Person':
                continue
                
            # Process each image in person's directory
            image_count = 0
            for img_path in glob.glob(os.path.join(person_dir, "*.jpg")) + \
                           glob.glob(os.path.join(person_dir, "*.jpeg")) + \
                           glob.glob(os.path.join(person_dir, "*.png")):
                
                image = self.face_recognition.load_image_file(img_path)
                face_locations = self.face_recognition.face_locations(image)
                
                if not face_locations:
                    logger.warning(f"No face found in {img_path}")
                    continue
                    
                # Use the first face found
                encoding = self.face_recognition.face_encodings(image, [face_locations[0]])[0]
                
                self.known_encodings.append(encoding)
                self.known_face_names.append(person_name)
                image_count += 1
                
            if image_count > 0:
                logger.info(f"Processed {image_count} images for {person_name}")
                
        # Save encodings for future use
        if self.known_encodings:
            with open(encodings_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_encodings,
                    'names': self.known_face_names
                }, f)
            logger.info(f"Saved {len(self.known_encodings)} face encodings to cache")
    
    def add_face_from_frame(self, frame, person_bbox, name):
        """Add a face from the current frame to the dataset"""
        if not name:
            return False
            
        # Create directory for the person if it doesn't exist
        person_dir = os.path.join(self.dataset_path, name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Extract face region from the frame
        x1, y1, x2, y2 = person_bbox
        face_img = frame[max(0, y1):min(frame.shape[0], y2), 
                          max(0, x1):min(frame.shape[1], x2)]
        
        # Save the face image
        timestamp = int(time.time())
        face_path = os.path.join(person_dir, f"{name}_{timestamp}.jpg")
        cv2.imwrite(face_path, face_img)
        
        # Add to known names if not already there
        if name not in self.known_face_names:
            self.known_face_names.append(name)
            
        # Create a persistent face entry for immediate recognition
        detection_id = f"{x1}_{y1}_{x2}_{y2}"
        self.persistent_faces[detection_id] = {
            'name': name,
            'confidence': 0.85,  # Start with high confidence for user-added faces
            'bbox_ratio': (
                x1 / frame.shape[1],
                y1 / frame.shape[0],
                x2 / frame.shape[1],
                y2 / frame.shape[0]
            ),
            'last_seen': time.time(),
            'confidence_trend': 1  # Initial trend direction
        }
        
        # Save the updated data
        self._save_mock_data()
        
        # If we have real face recognition, recompute encodings
        if hasattr(self, 'face_recognition'):
            # Invalidate encoding cache to force recomputation
            if os.path.exists(self.mock_encoding_file):
                os.unlink(self.mock_encoding_file)
            self._load_encodings()
            
        logger.info(f"Added face for {name} to dataset")
        return True

    def _load_mock_data(self):
        """Load mock face data for simulation when libraries not available"""
        if os.path.exists(self.mock_encoding_file):
            try:
                with open(self.mock_encoding_file, 'rb') as f:
                    data = pickle.load(f)
                    self.persistent_faces = data.get('faces', {})
                    stored_names = data.get('names', [])
                    for name in stored_names:
                        if name not in self.known_face_names:
                            self.known_face_names.append(name)
            except:
                logger.error("Error loading mock face data")
                self.persistent_faces = {}
        
        # Load all person names from dataset directories
        self._load_dataset_names()

    def _save_mock_data(self):
        """Save mock face data for persistence between runs"""
        try:
            with open(self.mock_encoding_file, 'wb') as f:
                pickle.dump({
                    'faces': self.persistent_faces,
                    'names': self.known_face_names
                }, f)
        except:
            logger.error("Error saving mock face data")

    def add_person(self, person_name, image_paths):
        """Add a new person to the face database from image files"""
        if not person_name or not image_paths:
            return False
            
        # Create person directory
        person_dir = os.path.join(self.dataset_path, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Copy images to person directory
        saved_images = 0
        for img_path in image_paths:
            if not os.path.exists(img_path):
                continue
                
            filename = os.path.basename(img_path)
            dest_path = os.path.join(person_dir, filename)
            shutil.copy2(img_path, dest_path)
            saved_images += 1
            
        # Add to known names if not already there
        if person_name not in self.known_face_names:
            self.known_face_names.append(name)
            
        # Reload encodings if using real face recognition
        if hasattr(self, 'face_recognition'):
            # Invalidate encoding cache to force recomputation
            if os.path.exists(self.mock_encoding_file):
                os.unlink(self.mock_encoding_file)
            self._load_encodings()
            
        logger.info(f"Added person: {person_name} with {saved_images} images")
        return True

    def recognize_faces(self, frame, person_detections=None):
        """Recognize faces in the frame using actual face recognition or mock data"""
        if hasattr(self, 'face_recognition') and self.model_loaded:
            return self._real_face_recognition(frame)
        else:
            return self._mock_face_recognition(frame, person_detections)
            
    def _real_face_recognition(self, frame):
        """Perform actual face recognition using face_recognition library"""
        if not self.known_encodings:
            return []
            
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = self.face_recognition.face_locations(rgb_small_frame)
        face_encodings = self.face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_results = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # See if the face matches any known faces
            matches = self.face_recognition.compare_faces(self.known_encodings, face_encoding, 
                                                          tolerance=self.tolerance)
            name = "Unknown"
            confidence = 0.0
            
            # Use the known face with the smallest distance to the new face
            face_distances = self.face_recognition.face_distance(self.known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    # Convert distance to confidence (0-1)
                    distance = face_distances[best_match_index]
                    # Convert distance to confidence (smaller distance = higher confidence)
                    confidence = max(0, min(1, 1 - distance))
                    
                    # Only include faces from dataset with sufficient confidence
                    if confidence >= 0.5:
                        face_results.append({
                            'bbox': (left, top, right, bottom),
                            'name': name,
                            'confidence': confidence
                        })
        
        return face_results
            
    def _mock_face_recognition(self, frame, person_detections=None):
        """Mock face recognition with persistent identity tracking - only use user's dataset"""
        if not person_detections or len(person_detections) == 0:
            return []
        
        if not self.known_face_names:
            return []  # No names to use
        
        face_results = []
        frame_height, frame_width = frame.shape[:2]
        
        # For each person, check if we already have a persistent identity
        for i, detection in enumerate(person_detections):
            x1, y1, x2, y2 = [int(c) for c in detection[:4]]
            detection_id = f"{x1}_{y1}_{x2}_{y2}"
            
            # Only recognize from existing persistent faces (people already in database)
            if detection_id not in self.persistent_faces:
                continue
            
            # Get the persistent identity
            face_info = self.persistent_faces[detection_id]
            
            # Update last seen time
            face_info['last_seen'] = time.time()
            
            # Calculate face box based on the person detection and stored ratio
            fx1_ratio, fy1_ratio, fx2_ratio, fy2_ratio = face_info['bbox_ratio']
            
            # Create face box - either use stored ratios or recalculate
            if np.random.random() > 0.5:  # Sometimes recalculate to simulate movement
                face_y1 = y1 
                face_y2 = y1 + int((y2 - y1) * 0.3)
                face_x1 = x1 + int((x2 - x1) * 0.2)
                face_width = int((x2 - x1) * 0.6)
                face_x2 = face_x1 + face_width
            else:
                # Use the stored ratio but scale to current frame size
                face_x1 = int(fx1_ratio * frame_width)
                face_y1 = int(fy1_ratio * frame_height)
                face_x2 = int(fx2_ratio * frame_width)
                face_y2 = int(fy2_ratio * frame_height)
            
            # Ensure bounds are within frame
            face_x1 = max(0, min(face_x1, frame_width-1))
            face_y1 = max(0, min(face_y1, frame_height-1))
            face_x2 = max(0, min(face_x2, frame_width-1))
            face_y2 = max(0, min(face_y2, frame_height-1))
            
            # Add to results with stored name and confidence
            face_results.append({
                'bbox': (face_x1, face_y1, face_x2, face_y2),
                'name': face_info['name'],
                'confidence': face_info['confidence']
            })
            
            # Update confidence in a more realistic way - gradually trending up or down
            # This creates a more natural looking confidence that fluctuates slightly
            if np.random.random() > 0.7:  # 30% chance to update confidence
                # Get current trend direction
                trend = face_info['confidence_trend']
                
                # Small random change
                change = np.random.uniform(0.01, 0.03) * trend
                
                # Apply change
                new_confidence = face_info['confidence'] + change
                
                # Ensure confidence stays in reasonable bounds
                if new_confidence > 0.95:
                    new_confidence = 0.95
                    face_info['confidence_trend'] = -1  # Start trending down
                elif new_confidence < 0.5:
                    new_confidence = 0.5
                    face_info['confidence_trend'] = 1  # Start trending up
                
                # Randomly reverse trend occasionally (10% chance)
                if np.random.random() > 0.9:
                    face_info['confidence_trend'] *= -1
                    
                face_info['confidence'] = new_confidence
        
        # Clean up old persistent faces (not seen in last 60 seconds)
        current_time = time.time()
        keys_to_remove = []
        for face_id, face_data in self.persistent_faces.items():
            if current_time - face_data['last_seen'] > 60:
                keys_to_remove.append(face_id)
        
        for key in keys_to_remove:
            del self.persistent_faces[key]
        
        # If we removed any faces, save the updated data
        if keys_to_remove:
            self._save_mock_data()
        
        return face_results 