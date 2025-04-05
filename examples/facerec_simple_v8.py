import face_recognition
import cv2
import numpy as np
import os
from PIL import Image

# Directory for known faces
KNOWN_FACES_DIR = "known_faces"

def get_face_encoding(image_array, print_debug=False):
    """Get face encoding while handling the compatibility issues"""
    try:
        # Get face locations first
        face_locations = face_recognition.face_locations(image_array, model="hog")
        if face_locations:
            if print_debug:
                print(f"Found {len(face_locations)} faces in image")
            # Get encodings using the face locations
            encodings = face_recognition.face_encodings(image_array, face_locations)
            if len(encodings) > 0:
                if print_debug:
                    print("Successfully got face encoding")
                return encodings[0]
            elif print_debug:
                print("No encodings found")
    except Exception as e:
        if print_debug:
            print(f"Error getting face encoding: {e}")
    return None

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
            
            # Get encoding with debug info
            encoding = get_face_encoding(image_array, print_debug=True)
            
            if encoding is not None:
                known_face_encodings.append(encoding)
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
video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use V4L2 backend explicitly

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
    
    # Calculate FPS every 30 frames
    if frame_count >= 30:
        end_time = cv2.getTickCount()
        fps = frame_count / ((end_time - start_time) / cv2.getTickFrequency())
        frame_count = 0
        start_time = end_time
        if debug_mode:
            print(f"\nFPS: {fps:.1f}")

    # Convert BGR to RGB
    rgb_frame = frame[:, :, ::-1]
    
    try:
        # Find all face locations in the frame
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        if debug_mode:
            print(f"\nFound {len(face_locations)} faces in frame")
        
        # Get face encodings for any faces in the frame
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Loop through each face in this frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            confidence = 0
            
            if len(known_face_encodings) > 0:
                # Calculate face distances
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                min_distance = face_distances[best_match_index]
                
                # Convert distance to similarity score (0-100%)
                similarity = (1 - min_distance) * 100
                
                if debug_mode:
                    print(f"Best match: {known_face_names[best_match_index]} with {similarity:.1f}% similarity")
                    print(f"All similarities: {[(name, (1-dist)*100) for name, dist in zip(known_face_names, face_distances)]}")
                
                # Use a lenient threshold
                if similarity > 45:  # Accept matches with >45% similarity
                    name = known_face_names[best_match_index]
                    confidence = similarity
            
            # Draw box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            label = f"{name} ({confidence:.1f}%)" if confidence > 0 else name
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    except Exception as e:
        print(f"Error processing frame: {e}")
    
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