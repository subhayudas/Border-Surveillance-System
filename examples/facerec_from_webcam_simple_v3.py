import face_recognition
import cv2
import numpy as np
import os

# This is where we'll store our known face images
KNOWN_FACES_DIR = "known_faces"

def load_known_face(image_path):
    # Load the image
    image = face_recognition.load_image_file(image_path)
    # Get the face encoding directly
    encodings = face_recognition.face_encodings(image)
    if len(encodings) > 0:
        return encodings[0]
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
        
        encoding = load_known_face(image_path)
        if encoding is not None:
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            print(f"Loaded face of: {name}")
        else:
            print(f"No face found in image: {filename}")

print(f"Loaded {len(known_face_names)} faces")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_frame = frame[:, :, ::-1]
    
    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    
    if face_locations:
        # Get face encodings for the frame
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Loop through each face in this frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known faces
            if len(known_face_encodings) > 0:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
            else:
                name = "Unknown"

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
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