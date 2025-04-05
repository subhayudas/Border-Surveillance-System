import face_recognition
import cv2
import numpy as np
import os
import dlib

# Directory for known faces
KNOWN_FACES_DIR = "known_faces"

# Initialize face detector
face_detector = dlib.get_frontal_face_detector()

def get_face_encoding(image, print_debug=False):
    """Get face encoding with error handling using dlib directly"""
    try:
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = image
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Detect faces using dlib
        faces = face_detector(rgb_image)
        if len(faces) == 0:
            if print_debug:
                print("No faces detected by dlib")
            return None
            
        if print_debug:
            print(f"Found {len(faces)} faces in image")
            
        # Get the largest face
        largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Convert dlib rectangle to face_recognition format
        top = largest_face.top()
        right = largest_face.right()
        bottom = largest_face.bottom()
        left = largest_face.left()
        
        face_location = (top, right, bottom, left)
        
        # Get face encoding
        face_encodings = face_recognition.face_encodings(rgb_image, [face_location])
        
        if len(face_encodings) > 0:
            if print_debug:
                print("Successfully got face encoding")
            return face_encodings[0]
        else:
            if print_debug:
                print("Failed to get face encoding")
            return None
            
    except Exception as e:
        if print_debug:
            print(f"Error in get_face_encoding: {e}")
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
print("Press 's' to save current frame for debugging")

debug_mode = True  # Start with debug mode on
frame_count = 0
save_count = 0

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:  # Process every 3rd frame
        continue

    # Convert BGR to RGB
    rgb_frame = frame[:, :, ::-1]
    
    try:
        # Detect faces using dlib
        faces = face_detector(rgb_frame)
        
        if debug_mode:
            print(f"\nFound {len(faces)} faces in frame")
        
        # Process each face
        for face in faces:
            # Convert dlib rectangle to face_recognition format
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            left = face.left()
            
            # Get face encoding
            face_location = (top, right, bottom, left)
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
                    print(f"All similarities: {[(name, (1-dist)*100) for name, dist in zip(known_face_names, face_distances)]}")
                
                # Use a very lenient threshold
                if similarity > 30:  # Even more lenient threshold
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
    elif key == ord('s'):
        # Save the current frame for debugging
        save_count += 1
        save_path = f"debug_frame_{save_count}.jpg"
        cv2.imwrite(save_path, frame)
        print(f"\nSaved debug frame to {save_path}")

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows() 