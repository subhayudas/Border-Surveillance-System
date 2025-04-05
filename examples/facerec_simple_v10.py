import dlib
import cv2
import numpy as np
import os
from PIL import Image
from datetime import datetime
from collections import deque  # Add this import

# Directory for known faces
KNOWN_FACES_DIR = "known_faces"
# Directory for unknown faces
UNKNOWN_FACES_DIR = "unknown_faces"

# Create unknown faces directory if it doesn't exist
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

# Recognition history for temporal smoothing
HISTORY_FRAMES = 8  # Increased from 5 to 8 frames
REQUIRED_HISTORY = 4  # Minimum frames needed for stable decision
KNOWN_FACE_THRESHOLD = 0.5  # 50% of frames must show known face
UNKNOWN_FACE_THRESHOLD = 0.7  # 70% of frames must show unknown face for saving
MIN_CONFIDENCE = 25  # Base confidence threshold
MIN_MARGIN = 1.5  # Reduced margin requirement
recognition_history = {}  # Dictionary to store recognition history for each face

# Initialize the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def get_face_encoding(image_array, print_debug=False):
    """Get face encodings using dlib directly"""
    try:
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Detect faces with higher upsampling for better detection
        faces = detector(gray, 1)  # Reduced from 2 to 1 for more consistent detection
        if len(faces) > 0:
            if print_debug:
                print(f"Found {len(faces)} faces in image")
            
            # Process all faces
            face_encodings = []
            for face in faces:
                # Get facial landmarks
                shape = predictor(image_array, face)
                
                # Get face encoding with fewer jitters for more consistent results
                face_descriptor = face_rec.compute_face_descriptor(image_array, shape, 1)  # Reduced jitters
                
                if face_descriptor is not None:
                    face_encodings.append(np.array(face_descriptor))
                    if print_debug:
                        print("Successfully got face encoding")
                elif print_debug:
                    print("No encoding found")
            
            return face_encodings
    except Exception as e:
        if print_debug:
            print(f"Error getting face encoding: {e}")
    return []

# Load known faces
print("\nLoading known faces...")
known_face_encodings = []
known_face_names = []

# First, try to load the faces
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        print(f"\nProcessing {name}...")
        
        try:
            # Load image using PIL first
            pil_image = Image.open(image_path)
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # Convert to numpy array
            image_array = np.array(pil_image)
            print(f"Image loaded, shape: {image_array.shape}")
            
            # Get encodings with debug info
            encodings = get_face_encoding(image_array, print_debug=True)
            
            if encodings:
                # Use the first face found in the reference image
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"✓ Successfully loaded face of: {name}")
            else:
                print(f"✗ No face found in {name}")
        except Exception as e:
            print(f"✗ Error processing {name}: {e}")

if len(known_face_encodings) == 0:
    print("\nNo faces were loaded! Please check your images.")
    exit(1)

print(f"\nSuccessfully loaded {len(known_face_names)} faces: {', '.join(known_face_names)}")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Set a lower resolution for faster processing
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nStarting face recognition...")
print("Press 'q' to quit")
print("Press 'd' to toggle debug mode")
print("Press 's' to save current frame for debugging")

debug_mode = True  # Start with debug mode on
frame_count = 0
save_count = 0

# For FPS calculation
fps = 0
start_time = cv2.getTickCount()

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    
    # Only process every other frame to improve performance
    if frame_count % 2 != 0:
        continue
    
    # Calculate FPS every 30 frames
    if frame_count >= 30:
        end_time = cv2.getTickCount()
        fps = frame_count / ((end_time - start_time) / cv2.getTickFrequency())
        frame_count = 0
        start_time = end_time
        if debug_mode:
            print(f"\nFPS: {fps:.1f}")

    try:
        # Convert BGR to RGB for dlib
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        
        if debug_mode:
            print(f"\nFound {len(faces)} faces in frame")
        
        # Process each face
        for face in faces:
            try:
                # Get facial landmarks
                shape = predictor(rgb_frame, face)
                
                # Get face encoding with more jitters
                face_encoding = np.array(face_rec.compute_face_descriptor(rgb_frame, shape, 2))
                
                # Initialize variables
                name = "Unknown"
                confidence = 0
                
                if len(known_face_encodings) > 0:
                    # Calculate face distances for all known faces
                    face_distances = [np.linalg.norm(face_encoding - enc) for enc in known_face_encodings]
                    similarities = [(1 - dist) * 100 for dist in face_distances]
                    
                    # Find best match and its similarity
                    best_match_index = np.argmin(face_distances)
                    best_similarity = similarities[best_match_index]
                    best_name = known_face_names[best_match_index]
                    
                    # Calculate the difference between best and second best match
                    sorted_similarities = sorted(similarities, reverse=True)
                    margin = sorted_similarities[0] - sorted_similarities[1] if len(sorted_similarities) > 1 else 0
                    
                    # Create a unique key for this face based on its position
                    face_key = f"{face.left()}_{face.top()}_{face.right()}_{face.bottom()}"
                    
                    # Initialize history for new faces
                    if face_key not in recognition_history:
                        recognition_history[face_key] = {
                            'history': deque(maxlen=HISTORY_FRAMES),
                            'last_seen': datetime.now()
                        }
                    
                    # Update last seen time
                    recognition_history[face_key]['last_seen'] = datetime.now()
                    
                    # Add current recognition to history
                    current_recognition = {
                        'name': best_name if best_similarity > 25 and margin > 2 else "Unknown",
                        'confidence': best_similarity,
                        'margin': margin
                    }
                    recognition_history[face_key]['history'].append(current_recognition)
                    
                    # Calculate the most common name in recent history
                    recent_history = recognition_history[face_key]['history']
                    if len(recent_history) >= REQUIRED_HISTORY:  # Wait for enough history
                        # Count known face recognitions
                        known_faces = [r for r in recent_history if r['name'] != "Unknown"]
                        if len(known_faces) >= len(recent_history) * KNOWN_FACE_THRESHOLD:  # If enough frames show a known face
                            # Get the most common name among known faces
                            name_counts = {}
                            for r in known_faces:
                                name_counts[r['name']] = name_counts.get(r['name'], 0) + 1
                            
                            most_common_name = max(name_counts.items(), key=lambda x: x[1])[0]
                            # Use the most confident recognition for this name
                            matching_recognitions = [r for r in known_faces if r['name'] == most_common_name]
                            best_recognition = max(matching_recognitions, key=lambda x: x['confidence'])
                            name = best_recognition['name']
                            confidence = best_recognition['confidence']
                        else:
                            name = "Unknown"
                            confidence = best_similarity
                    else:
                        # Not enough history yet, use current frame with lower requirements
                        if best_similarity > MIN_CONFIDENCE and margin > MIN_MARGIN:
                            name = best_name
                            confidence = best_similarity
                    
                    if debug_mode:
                        if name != "Unknown":
                            print(f"\nMatch found at ({face.left()}, {face.top()}): {name} with {confidence:.1f}% confidence")
                            print(f"Margin to next best match: {margin:.1f}%")
                            if len(recent_history) >= REQUIRED_HISTORY:
                                known_ratio = len(known_faces) / len(recent_history)
                                print(f"Known face ratio: {known_ratio:.1%}")
                        else:
                            print(f"\nUnknown face at ({face.left()}, {face.top()})")
                            print(f"Best match was {best_name} with {best_similarity:.1f}% confidence")
                            if margin < MIN_MARGIN:
                                print(f"Margin too small: {margin:.1f}% (needs > {MIN_MARGIN}%)")
                    
                    # Only save unknown faces if consistently unknown for multiple frames
                    if name == "Unknown" and len(recent_history) >= REQUIRED_HISTORY:
                        unknown_count = sum(1 for r in recent_history if r['name'] == "Unknown")
                        if unknown_count >= len(recent_history) * UNKNOWN_FACE_THRESHOLD:  # If enough frames show unknown
                            # Capture unknown face
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Added milliseconds
                            face_img = frame[face.top():face.bottom(), face.left():face.right()]
                            if face_img.size > 0:  # Make sure the face region is valid
                                # Add timestamp overlay to the image
                                time_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.35  # Reduced from 0.5
                                thickness = 1
                                # Get text size
                                (text_width, text_height), _ = cv2.getTextSize(time_text, font, font_scale, thickness)
                                
                                # Create space for the timestamp at the bottom (smaller padding)
                                img_with_text = np.zeros((face_img.shape[0] + text_height + 6, face_img.shape[1], 3), dtype=np.uint8)
                                img_with_text[:face_img.shape[0], :] = face_img
                                
                                # Add white background for text (reduced height)
                                cv2.rectangle(img_with_text, 
                                            (0, face_img.shape[0]), 
                                            (face_img.shape[1], face_img.shape[0] + text_height + 6), 
                                            (255, 255, 255), 
                                            -1)
                                
                                # Add timestamp text (adjusted position)
                                cv2.putText(img_with_text, 
                                          time_text, 
                                          (2, face_img.shape[0] + text_height + 1),  # Adjusted position
                                          font, 
                                          font_scale, 
                                          (0, 0, 0), 
                                          thickness)
                                
                                unknown_face_path = os.path.join(UNKNOWN_FACES_DIR, f"unknown_{timestamp}.jpg")
                                cv2.imwrite(unknown_face_path, img_with_text)
                                if debug_mode:
                                    print(f"Saved unknown face to {unknown_face_path}")
                
                # Get face rectangle coordinates
                left = face.left()
                top = face.top()
                right = face.right()
                bottom = face.bottom()
                
                # Draw box with thicker lines
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
                thickness = 2 if name != "Unknown" else 3  # Thicker lines for unknown faces
                cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
                
                # Draw label with larger font and background
                label = f"{name} ({confidence:.1f}%)" if confidence > 0 else "Unknown"
                font_scale = 0.8 if name != "Unknown" else 1.0  # Larger font for unknown faces
                thickness = 2
                font = cv2.FONT_HERSHEY_DUPLEX
                
                # Get size of the text for the background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw background rectangle and ensure it doesn't go outside frame bounds
                rect_left = max(left, 0)
                rect_bottom = min(bottom, frame.shape[0])
                rect_top = max(rect_bottom - text_height - 20, 0)
                rect_right = min(rect_left + text_width + 10, frame.shape[1])
                
                cv2.rectangle(frame, (rect_left, rect_top), (rect_right, rect_bottom), color, cv2.FILLED)
                
                # Draw text
                text_bottom = min(bottom - 10, frame.shape[0] - 10)
                cv2.putText(frame, label, (rect_left + 5, text_bottom), font, font_scale, (255, 255, 255), thickness)
                
            except Exception as e:
                if debug_mode:
                    print(f"Error processing face: {e}")
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        if debug_mode:
            import traceback
            traceback.print_exc()
    
    # Draw FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
    
    # Display the resulting image
    cv2.imshow('Face Recognition', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"\nDebug mode {'enabled' if debug_mode else 'disabled'}")
    elif key == ord('s'):
        # Save the current frame for debugging
        save_count += 1
        save_path = f"debug_frame_{save_count}.jpg"
        cv2.imwrite(save_path, frame)
        print(f"\nSaved debug frame to {save_path}")

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows() 