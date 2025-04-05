import face_recognition
import cv2
import numpy as np
import os

# This is where we'll store our known face images
KNOWN_FACES_DIR = "known_faces"

def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    
    # Load each face image from the known_faces directory
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            
            # Get face locations first
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                # Get the encoding of the first face
                top, right, bottom, left = face_locations[0]
                face_image = image[top:bottom, left:right]
                face_encoding = face_recognition.face_encodings(face_image)[0]
                
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
                print(f"Loaded face of: {name}")
            else:
                print(f"No face found in image: {filename}")
    
    return known_face_encodings, known_face_names

# Load known faces
print("Loading known faces...")
known_face_encodings, known_face_names = load_known_faces()

# Get a reference to webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(frame)
    
    # Process each face in the frame
    for face_location in face_locations:
        top, right, bottom, left = face_location
        
        # Extract the face image
        face_image = frame[top:bottom, left:right]
        
        # Get face encoding
        face_encodings = face_recognition.face_encodings(face_image)
        
        if face_encodings:
            face_encoding = face_encodings[0]
            
            # Find matches
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw box and name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows() 