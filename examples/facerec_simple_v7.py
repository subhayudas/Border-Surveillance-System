import dlib
import cv2
import numpy as np
import os

# Directory for known faces
KNOWN_FACES_DIR = "known_faces"

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def get_face_encoding(image, print_debug=False):
    """Get face encoding with error handling using dlib directly"""
    try:
        # Convert to RGB if needed (dlib expects RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = image
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Detect faces
        faces = detector(rgb_image)
        if len(faces) == 0:
            if print_debug:
                print("No faces detected")
            return None
            
        if print_debug:
            print(f"Found {len(faces)} faces in image")
            
        # Get the largest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get facial landmarks
        shape = predictor(rgb_image, face)
        
        # Get face encoding
        face_encoding = np.array(face_rec_model.compute_face_descriptor(rgb_image, shape))
        
        if print_debug:
            print("Successfully got face encoding")
        return face_encoding
            
    except Exception as e:
        if print_debug:
            print(f"Error in get_face_encoding: {e}")
        return None

def compare_faces(known_encoding, face_encoding, tolerance=0.5):
    """Compare faces and return similarity score"""
    if known_encoding is None or face_encoding is None:
        return 0
    
    # Calculate Euclidean distance
    dist = np.linalg.norm(known_encoding - face_encoding)
    # Convert to similarity score (0-100%)
    similarity = max(0, min(100, (1 - dist) * 100))
    return similarity

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
            image = cv2.imread(image_path)
            if image is None:
                print(f"✗ Failed to load image: {image_path}")
                continue
                
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
    
    try:
        # Convert BGR to RGB (dlib expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using dlib
        faces = detector(rgb_frame)
        
        if debug_mode:
            print(f"\nFound {len(faces)} faces in frame")
        
        # Process each face
        for face in faces:
            try:
                # Get landmarks and encoding
                shape = predictor(rgb_frame, face)
                face_encoding = np.array(face_rec_model.compute_face_descriptor(rgb_frame, shape))
                
                if debug_mode:
                    print("Successfully got face encoding for frame")
                
                name = "Unknown"
                best_similarity = 0
                
                # Compare with known faces
                for known_encoding, known_name in zip(known_face_encodings, known_face_names):
                    similarity = compare_faces(known_encoding, face_encoding)
                    
                    if debug_mode:
                        print(f"Similarity with {known_name}: {similarity:.1f}%")
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        name = known_name if similarity > 45 else "Unknown"
                
                # Draw box
                left = face.left()
                top = face.top()
                right = face.right()
                bottom = face.bottom()
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Draw label
                label = f"{name} ({best_similarity:.1f}%)" if best_similarity > 0 else name
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
            except Exception as e:
                if debug_mode:
                    print(f"Error processing face: {e}")
    
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