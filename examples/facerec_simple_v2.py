import face_recognition
import cv2
import numpy as np
import os
import dlib

# Directory for known faces
KNOWN_FACES_DIR = "known_faces"

# Initialize face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def get_face_encoding(image):
    # Convert to dlib format
    dlib_faces = face_detector(image, 1)
    if len(dlib_faces) == 0:
        return None
    
    # Get face landmarks
    shape = predictor(image, dlib_faces[0])
    # Get face encoding
    return np.array(face_rec_model.compute_face_descriptor(image, shape))

# Load known faces
print("Loading known faces...")
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        print(f"Loading {name}...")
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load {name}")
            continue
            
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get encoding
        encoding = get_face_encoding(rgb_image)
        if encoding is not None:
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            print(f"Loaded {name} successfully!")
        else:
            print(f"No face found in {name}")

print(f"Loaded {len(known_face_names)} faces")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find faces using dlib
    faces = face_detector(rgb_frame, 1)
    
    # Process each face
    for face in faces:
        # Get face landmarks
        shape = predictor(rgb_frame, face)
        # Get face encoding
        face_encoding = np.array(face_rec_model.compute_face_descriptor(rgb_frame, shape))
        
        # Compare with known faces
        if len(known_face_encodings) > 0:
            # Calculate distances
            face_distances = [np.linalg.norm(face_encoding - enc) for enc in known_face_encodings]
            best_match_index = np.argmin(face_distances)
            min_distance = face_distances[best_match_index]
            
            # If distance is small enough, we have a match
            if min_distance < 0.6:
                name = known_face_names[best_match_index]
                confidence = (1 - min_distance) * 100
            else:
                name = "Unknown"
                confidence = 0
        else:
            name = "Unknown"
            confidence = 0
        
        # Draw rectangle around face
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        
        # Draw box
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw label
        label = f"{name} ({confidence:.1f}%)" if confidence > 0 else name
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows() 