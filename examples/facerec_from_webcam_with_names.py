import face_recognition
import cv2
import numpy as np
import os

# This is where we'll store our known face images
KNOWN_FACES_DIR = "known_faces"

# Create the directory if it doesn't exist
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    
    # Load each face image from the known_faces directory
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            # Get the name from the filename (without extension)
            name = os.path.splitext(filename)[0]
            
            # Load the image file
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            
            # Try to get the face encoding
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                # Take the first face found in the image
                face_encoding = face_encodings[0]
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

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Only process every other frame to save time
    if process_this_frame:
        # Resize frame for faster face recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert the image from BGR color to RGB color
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            if len(known_face_encodings) > 0:
                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

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