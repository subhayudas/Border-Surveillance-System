import face_recognition
import cv2
import numpy as np
import os
from PIL import Image

# This is where we'll store our known face images
KNOWN_FACES_DIR = "known_faces"

def get_face_encoding(image_array):
    """Get face encoding while handling the compatibility issues"""
    try:
        # Get face locations first
        face_locations = face_recognition.face_locations(image_array, model="hog")
        if face_locations:
            # Get encodings using the face locations
            encodings = face_recognition.face_encodings(image_array, face_locations)
            if len(encodings) > 0:
                return encodings[0]
    except Exception as e:
        print(f"Error getting face encoding: {e}")
    return None

# Load known faces
print("Loading known faces...")
known_face_encodings = []
known_face_names = []

# Load each face image from the known_faces directory
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        print(f"\nProcessing {image_path}...")
        
        try:
            # Load image using PIL first
            pil_image = Image.open(image_path)
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            # Try to get face encoding
            face_encoding = get_face_encoding(image_array)
            if face_encoding is not None:
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
                print(f"✓ Successfully loaded face of: {name}")
            else:
                print(f"✗ No face found in image: {filename}")
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")

print(f"\nSuccessfully loaded {len(known_face_names)} faces: {', '.join(known_face_names)}")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Set a lower resolution for faster processing
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# For FPS calculation
frame_count = 0
fps = 0
start_time = cv2.getTickCount()

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Calculate FPS
    frame_count += 1
    if frame_count >= 30:
        end_time = cv2.getTickCount()
        fps = frame_count / ((end_time - start_time) / cv2.getTickFrequency())
        frame_count = 0
        start_time = end_time
        
    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_frame = frame[:, :, ::-1]
    
    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    
    # Process each face in the frame
    for (top, right, bottom, left) in face_locations:
        # Extract face image
        face_image = rgb_frame[top:bottom, left:right]
        
        # Try to get face encoding
        face_encoding = get_face_encoding(face_image)
        
        name = "Unknown"
        confidence = 0
        
        if face_encoding is not None and len(known_face_encodings) > 0:
            # Get face distances
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            min_distance = face_distances[best_match_index]
            
            # Convert distance to similarity score (0-100%)
            similarity = (1 - min_distance) * 100
            
            # Use a more lenient threshold
            if similarity > 50:  # Accept matches with >50% similarity
                name = known_face_names[best_match_index]
                confidence = similarity
        
        # Draw a box around the face
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw a label with a name below the face
        label = f"{name} ({confidence:.1f}%)" if confidence > 0 else name
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
    
    # Draw FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
    
    # Display the resulting image
    cv2.imshow('Face Recognition', frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows() 