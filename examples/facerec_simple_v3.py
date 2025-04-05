import face_recognition
import cv2
import numpy as np
import os

# Directory for known faces
KNOWN_FACES_DIR = "known_faces"

def get_face_encoding(image):
    """Get face encoding with error handling"""
    try:
        # First try to get face locations
        face_locations = face_recognition.face_locations(image, model="hog")
        if not face_locations:
            return None
            
        # Try to get face encodings
        face_encodings = face_recognition.face_encodings(
            image,
            known_face_locations=face_locations,
            num_jitters=1,
            model="small"
        )
        
        if face_encodings:
            return face_encodings[0]
    except Exception as e:
        print(f"Error getting face encoding: {e}")
    return None

# Load known faces
print("Loading known faces...")
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        print(f"Loading {name}...")
        
        try:
            # Load the image
            image = face_recognition.load_image_file(image_path)
            # Get encoding
            encoding = get_face_encoding(image)
            
            if encoding is not None:
                known_face_encodings.append(encoding)
                known_face_names.append(name)
                print(f"✓ Loaded {name} successfully!")
            else:
                print(f"✗ No face found in {name}")
        except Exception as e:
            print(f"✗ Error processing {name}: {e}")

print(f"\nSuccessfully loaded {len(known_face_names)} faces: {', '.join(known_face_names)}")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Set a lower resolution for faster processing
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = frame[:, :, ::-1]
    
    try:
        # Find faces
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        # Process each face
        for face_location in face_locations:
            top, right, bottom, left = face_location
            
            # Extract the face ROI (Region of Interest)
            face_image = rgb_frame[top:bottom, left:right]
            
            # Get face encoding
            face_encoding = get_face_encoding(face_image)
            
            name = "Unknown"
            confidence = 0
            
            if face_encoding is not None and len(known_face_encodings) > 0:
                # Calculate face distances
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                min_distance = face_distances[best_match_index]
                
                # Convert distance to similarity score (0-100%)
                similarity = (1 - min_distance) * 100
                
                # Use a more lenient threshold
                if similarity > 40:  # Accept matches with >40% similarity
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
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows() 