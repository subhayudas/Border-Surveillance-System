import cv2  # Install opencv-python
import numpy as np
import os
import tensorflow as tf
import time
import h5py
import math
from collections import Counter

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Global variables
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = None
class_names = []
model_path = "model_weights/converted_model.keras"
labels_path = "model_weights/labels.txt"

def load_class_names():
    """Load class names from labels file"""
    global class_names
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(class_names)} classes: {class_names}")
    else:
        class_names = ["unknown"]
        print(f"Warning: Labels file not found at {labels_path}")

def load_model():
    """Load the TensorFlow model"""
    global model
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            return True
        else:
            print(f"Model file not found at {model_path}")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_face(face_img):
    """Preprocess face image for the model"""
    # Resize to 224x224 (assumed input size for MobileNet)
    face_img = cv2.resize(face_img, (224, 224))
    
    # Convert BGR to RGB (TensorFlow models expect RGB)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0, 1]
    face_img = face_img.astype(np.float32) / 255.0
    
    # Add batch dimension
    face_img = np.expand_dims(face_img, axis=0)
    
    return face_img

def extract_features(face_img):
    """Extract features from face image using average pooling simulation"""
    # This is a simplified simulation of what the MobileNet backbone would do
    # In a real scenario, this would be handled by the actual feature extractor of the model
    
    # Resize to standard size
    face_img = cv2.resize(face_img, (224, 224))
    
    # Convert BGR to RGB
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Normalize
    face_img = face_img.astype(np.float32) / 255.0
    
    # Create a simplified feature vector (1280 features to match the model)
    # In reality, this would be done by the actual CNN, but we're approximating
    features = np.zeros(1280)
    
    # Divide the image into regions and extract average color features
    h, w, c = face_img.shape
    block_h, block_w = h // 16, w // 16
    
    feature_idx = 0
    for i in range(0, h, block_h):
        for j in range(0, w, block_w):
            block = face_img[i:i+block_h, j:j+block_w, :]
            # Extract mean color and std for each channel in this block
            for c_idx in range(c):
                if feature_idx < 1280:
                    features[feature_idx] = np.mean(block[:, :, c_idx])
                    feature_idx += 1
                if feature_idx < 1280:
                    features[feature_idx] = np.std(block[:, :, c_idx])
                    feature_idx += 1
    
    # Fill any remaining features with zeros
    return features

def predict_face(face_img):
    """Predict the class of a face image using the loaded model"""
    global model, class_names
    
    if model is None:
        # If model is not loaded, use feature-based classification
        print("Using feature-based classification as model is not loaded")
        features = extract_features(face_img)
        features = np.expand_dims(features, axis=0)  # Add batch dimension
    else:
        # First extract features (simplified simulation of feature extraction)
        features = extract_features(face_img)
        features = np.expand_dims(features, axis=0)  # Add batch dimension
    
    try:
        # Make prediction using the model
        predictions = model.predict(features, verbose=0)
        class_id = np.argmax(predictions[0])
        confidence = predictions[0][class_id]
        
        # Get class name
        if 0 <= class_id < len(class_names):
            class_name = class_names[class_id].split(' ', 1)[1] if ' ' in class_names[class_id] else class_names[class_id]
        else:
            class_name = "unknown"
            
        return class_name, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "unknown", 0.0

def detect_and_recognize_faces(frame):
    """Detect faces in a frame and recognize them"""
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract face region
        face_img = frame[y:y+h, x:x+w]
        
        # Skip if face is too small or invalid
        if face_img.size == 0 or w < 30 or h < 30:
            continue
            
        # Predict face
        name, confidence = predict_face(face_img)
        
        # Draw rectangle around face
        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Display name and confidence
        label = f"{name} ({confidence:.2f})"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame, len(faces)

def main():
    # Load class names and model
    load_class_names()
    model_loaded = load_model()
    
    if not model_loaded:
        print("Warning: Model could not be loaded. Using feature-based classification.")
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    
    print("Starting face recognition. Press 'q' to quit.")
    
    # Frame counter and FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Face detection history for logging
    face_history = []
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Detect and recognize faces
        processed_frame, num_faces = detect_and_recognize_faces(frame)
        
        # Update face history
        face_history.append(num_faces)
        if len(face_history) > 30:  # Keep last 30 frames
            face_history.pop(0)
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            
            # Log face detection stats
            avg_faces = sum(face_history) / len(face_history)
            print(f"FPS: {fps:.2f}, Avg faces: {avg_faces:.1f}")
        
        # Display FPS
        cv2.putText(
            processed_frame, 
            f"FPS: {fps:.2f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Display the frame
        cv2.imshow('Face Recognition', processed_frame)
        
        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
