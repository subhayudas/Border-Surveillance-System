import face_recognition
import cv2
import numpy as np
import os

# Directory for known faces
KNOWN_FACES_DIR = "known_faces"

def get_face_encoding(image, print_debug=False):
    """Get face encoding with error handling"""
    try:
        # First try to get face locations
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            if print_debug:
                print("No face locations found")
            return None
            
        # Try to get face encodings directly from the image and locations
        encodings = face_recognition.face_encodings(image, face_locations)
        
        if len(encodings) > 0:
            if print_debug:
                print("Successfully got face encoding")
            return encodings[0]
        else:
            if print_debug:
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
            # Load the image
            image = face_recognition.load_image_file(image_path)
            print(f"Image loaded, shape: {image.shape}")
            
            # Get encoding with debug info
            encoding = get_face_encoding(image, print_debug=True)
            
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
video_capture = cv2.VideoCapture(0)

# Set a lower resolution for faster processing
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nStarting face recognition...")
print("Press 'q' to quit")
print("Press 'd' to toggle debug mode")

debug_mode = False

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = frame[:, :, ::-1]
    
    try:
        # Find faces
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if debug_mode and len(face_locations) > 0:
            print(f"\nFound {len(face_locations)} faces in frame")
        
        # Process each face
        for face_location in face_locations:
            top, right, bottom, left = face_location
            
            # Get face encoding from the full frame
            face_encodings = face_recognition.face_encodings(rgb_frame, [face_location])
            
            name = "Unknown"
            confidence = 0
            
            if len(face_encodings) > 0 and len(known_face_encodings) > 0:
                face_encoding = face_encodings[0]
                
                # Calculate face distances
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                min_distance = face_distances[best_match_index]
                
                # Convert distance to similarity score (0-100%)
                similarity = (1 - min_distance) * 100
                
                if debug_mode:
                    print(f"Best match: {known_face_names[best_match_index]} with {similarity:.1f}% similarity")
                
                # Use a very lenient threshold
                if similarity > 35:  # Accept matches with >35% similarity
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
    
    # Display the resulting image
    cv2.imshow('Face Recognition', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"\nDebug mode {'enabled' if debug_mode else 'disabled'}")

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows() 