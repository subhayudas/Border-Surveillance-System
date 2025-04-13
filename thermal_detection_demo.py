import cv2
import os
import sys
import time
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue

# Global variables for GUI
frame_queue = queue.Queue(maxsize=10)
stop_event = threading.Event()

def display_thread_function():
    """Thread function to display frames using matplotlib"""
    plt.figure(figsize=(16, 9))
    plt.ion()  # Enable interactive mode
    plt.show()
    
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)
            if frame is not None:
                plt.clf()
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title('Thermal Detection Demo')
                plt.draw()
                plt.pause(0.001)  # Small pause to allow GUI to update
            frame_queue.task_done()
        except queue.Empty:
            plt.pause(0.1)
        except Exception as e:
            print(f"Display error: {e}")
            break
    
    plt.close()

def main():
    print("Thermal Detection Demo using HOT models with Getty Images Video")
    print("-------------------------------------------------------------")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load both models
    models = []
    
    # Path to detection model
    detection_model_path = os.path.join("HOT-main", "HOT-main", "yolov8nano640", "weights", "best.pt")
    if os.path.exists(detection_model_path):
        print(f"Loading detection model from {detection_model_path}...")
        try:
            detection_model = YOLO(detection_model_path)
            detection_model.to(device)
            models.append(("Detection", detection_model))
            print("Detection model loaded successfully!")
        except Exception as e:
            print(f"Error loading detection model: {e}")
    else:
        print(f"Detection model not found at {detection_model_path}")
    
    # Path to segmentation model
    segmentation_model_path = os.path.join("HOT-main", "HOT-main", "yolov8n-seg-640x640-aug1", "weights", "best.pt")
    if os.path.exists(segmentation_model_path):
        print(f"Loading segmentation model from {segmentation_model_path}...")
        try:
            segmentation_model = YOLO(segmentation_model_path)
            segmentation_model.to(device)
            models.append(("Segmentation", segmentation_model))
            print("Segmentation model loaded successfully!")
        except Exception as e:
            print(f"Error loading segmentation model: {e}")
    else:
        print(f"Segmentation model not found at {segmentation_model_path}")
    
    if not models:
        print("No models could be loaded. Please ensure the HOT-main folder contains the trained models.")
        return
    
    # Use Getty Images video directly
    getty_video = "gettyimages-1382583689-640_adpp.mp4"
    if not os.path.exists(getty_video):
        print(f"Error: Getty Images video file '{getty_video}' not found!")
        return
    
    source = getty_video
    print(f"Using Getty Images video: {source}")
    
    # Open video source
    print(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video source opened: {width}x{height} at {fps} FPS, {total_frames} total frames")
    
    # Set up output directories
    output_dir = "output_thermal"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup output video
    output_path = os.path.join(output_dir, "thermal_getty_demo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Also create separate output for segmentation if available
    if len(models) > 1:
        seg_output_path = os.path.join(output_dir, "thermal_getty_demo_segmentation.mp4")
        seg_output_video = cv2.VideoWriter(seg_output_path, fourcc, fps, (width, height))
    
    print(f"Saving output video to: {output_path}")
    
    # Start display thread for GUI
    print("Starting display GUI (matplotlib window)")
    display_thread = threading.Thread(target=display_thread_function)
    display_thread.daemon = True
    display_thread.start()
    
    # Process frames
    frame_count = 0
    model_processing_times = {model_name: [] for model_name, _ in models}
    
    try:
        print("Starting video processing...")
        print("Showing real-time detection in separate window")
        print("Press Ctrl+C to stop processing")
        
        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            frame_count += 1
            
            # Update progress every 30 frames
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {progress:.1f}% (Frame {frame_count}/{total_frames})")
            
            # Process frame with each model
            result_frames = []
            for i, (model_name, model) in enumerate(models):
                start_time = time.time()
                
                # Run inference with confidence threshold
                results = model(frame, conf=0.25)
                
                # Get processed frame with detections
                result_frame = results[0].plot()
                
                # Calculate processing time
                processing_time = time.time() - start_time
                model_processing_times[model_name].append(processing_time)
                
                # Calculate average FPS over last 10 frames
                avg_times = model_processing_times[model_name][-10:]
                avg_time = sum(avg_times) / len(avg_times)
                fps_text = f"FPS: {1/avg_time:.1f}"
                
                # Add model name and stats
                model_text = f"{model_name} Model - Getty Images Thermal"
                cv2.putText(result_frame, model_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, fps_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add frame counter
                progress_text = f"Frame: {frame_count}/{total_frames}"
                cv2.putText(result_frame, progress_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                result_frames.append(result_frame)
                
                # Save to output video
                if i == 0:
                    output_video.write(result_frame)
                    
                    # Save snapshot images every 100 frames
                    if frame_count % 100 == 0 or frame_count == 1:
                        snapshot_path = os.path.join(output_dir, f"detection_frame_{frame_count:04d}.jpg")
                        cv2.imwrite(snapshot_path, result_frame)
                        print(f"Saved detection snapshot: {snapshot_path}")
                
                elif i == 1 and len(models) > 1:
                    seg_output_video.write(result_frame)
                    
                    # Save snapshot images every 100 frames
                    if frame_count % 100 == 0 or frame_count == 1:
                        snapshot_path = os.path.join(output_dir, f"segmentation_frame_{frame_count:04d}.jpg")
                        cv2.imwrite(snapshot_path, result_frame)
                        print(f"Saved segmentation snapshot: {snapshot_path}")
            
            # Create combined display for visualization
            if len(result_frames) == 2:
                # Stack them horizontally
                h1, w1 = result_frames[0].shape[:2]
                h2, w2 = result_frames[1].shape[:2]
                
                # Ensure same height for side-by-side display
                if h1 != h2:
                    target_height = min(h1, h2)
                    aspect_ratio1 = w1 / h1
                    aspect_ratio2 = w2 / h2
                    
                    new_w1 = int(target_height * aspect_ratio1)
                    new_w2 = int(target_height * aspect_ratio2)
                    
                    result_frames[0] = cv2.resize(result_frames[0], (new_w1, target_height))
                    result_frames[1] = cv2.resize(result_frames[1], (new_w2, target_height))
                
                combined_frame = np.hstack(result_frames)
                
                # Save combined view at regular intervals
                if frame_count % 50 == 0 or frame_count == 1:
                    combined_path = os.path.join(output_dir, "current_combined.jpg")
                    cv2.imwrite(combined_path, combined_frame)
                
                # Send to display thread
                if not frame_queue.full():
                    frame_queue.put(combined_frame)
            else:
                # Just display the detection frame
                if not frame_queue.full() and result_frames:
                    frame_queue.put(result_frames[0])
                    
            # Slow down processing slightly to allow for better visualization
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    
    finally:
        # Signal display thread to stop
        stop_event.set()
        
        # Release resources
        cap.release()
        output_video.release()
        if len(models) > 1:
            seg_output_video.release()
        
        # Wait for display thread to finish
        display_thread.join(timeout=1.0)
        
        # Calculate and show statistics
        print("\nProcessing Statistics:")
        print(f"Processed {frame_count} frames")
        
        for model_name in model_processing_times:
            times = model_processing_times[model_name]
            if times:
                avg_time = sum(times) / len(times)
                avg_fps = 1 / avg_time
                print(f"{model_name} Model:")
                print(f"  Average processing time: {avg_time:.4f} seconds per frame")
                print(f"  Average FPS: {avg_fps:.2f}")
        
        print(f"Output video saved to: {output_path}")
        if len(models) > 1:
            print(f"Segmentation output video saved to: {seg_output_path}")
        print(f"Snapshot images saved to: {output_dir}")
        print("Demo completed")

if __name__ == "__main__":
    main() 