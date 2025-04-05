import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import time
import threading
from datetime import datetime
import numpy as np

# Import detection and visualization modules
from src.detection import ObjectDetector, BehaviorAnalyzer, FenceTamperingDetector, WeaponDetector, BorderCrossingDetector
from src.visualization import Visualizer
from config import settings

class BorderSurveillanceSystem:
    def __init__(self):
        self.cameras = {}  # Dictionary to store multiple camera objects
        self.active_feeds = []  # List to track active camera feeds
        self.is_running = False
        self.alert_history = []  # Store alert history
        self.video_file = None  # For uploaded video analysis
        self.video_player = None  # For video playback
        self.setup_ui()
        
    def setup_ui(self):
        # Create main window with dark theme
        self.root = tk.Tk()
        self.root.title("Border Surveillance System")
        self.root.geometry("1280x900")
        self.root.configure(bg="#1e1e1e")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Set style for ttk widgets
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#1e1e1e')
        self.style.configure('TButton', background='#3a7ebf', foreground='white', borderwidth=1, focusthickness=3, focuscolor='none')
        self.style.map('TButton', background=[('active', '#2a6eaf')])
        self.style.configure('TLabel', background='#1e1e1e', foreground='white')
        self.style.configure('Header.TLabel', background='#3a7ebf', foreground='white', font=('Arial', 18, 'bold'))
        self.style.configure('Status.TLabel', background='#3a7ebf', foreground='yellow', font=('Arial', 12))
        
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create header
        header_frame = ttk.Frame(self.main_container, style='TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        header_inner = ttk.Frame(header_frame, style='TFrame')
        header_inner.pack(fill=tk.X)
        header_inner.configure(style='TFrame')
        header_inner['padding'] = (10, 10, 10, 10)
        header_inner['relief'] = 'raised'
        
        ttk.Label(
            header_inner, 
            text="BORDER SURVEILLANCE SYSTEM", 
            style='Header.TLabel'
        ).pack(side=tk.LEFT)
        
        # Status indicator
        self.status_label = ttk.Label(
            header_inner,
            text="STANDBY",
            style='Status.TLabel'
        )
        self.status_label.pack(side=tk.RIGHT)
        
        # Create tab control for different views
        self.tab_control = ttk.Notebook(self.main_container)
        self.tab_control.pack(fill=tk.BOTH, expand=True)
        
        # Live monitoring tab
        self.monitoring_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.monitoring_tab, text="Live Monitoring")
        
        # Video analysis tab
        self.video_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.video_tab, text="Video Analysis")
        
        # Dashboard tab
        self.dashboard_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.dashboard_tab, text="Dashboard")
        
        # Settings tab
        self.settings_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.settings_tab, text="Settings")
        
        # Setup each tab
        self.setup_monitoring_tab()
        self.setup_video_tab()
        self.setup_dashboard_tab()
        self.setup_settings_tab()
        
    def setup_monitoring_tab(self):
        # Control panel
        control_frame = ttk.Frame(self.monitoring_tab)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Surveillance controls
        surveillance_frame = ttk.LabelFrame(control_frame, text="Surveillance Controls")
        surveillance_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.start_button = ttk.Button(surveillance_frame, text="Start Surveillance", command=self.start_surveillance)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_button = ttk.Button(surveillance_frame, text="Stop Surveillance", command=self.stop_surveillance)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Camera controls
        camera_frame = ttk.LabelFrame(control_frame, text="Camera Management")
        camera_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(camera_frame, text="Add Camera", command=self.add_camera).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(camera_frame, text="Remove Camera", command=self.remove_camera).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Create a frame for displaying multiple camera feeds
        self.feeds_container = ttk.Frame(self.monitoring_tab)
        self.feeds_container.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create a frame for camera feeds
        self.feeds_frame = ttk.Frame(self.feeds_container)
        self.feeds_frame.pack(fill=tk.BOTH, expand=True)
        
        # Alert panel
        self.alert_frame = ttk.LabelFrame(self.monitoring_tab, text="Live Alerts")
        self.alert_frame.pack(fill=tk.X, pady=10)
        
        self.alert_text = scrolledtext.ScrolledText(self.alert_frame, height=5)
        self.alert_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.alert_text.config(state=tk.DISABLED)
        
    def setup_video_tab(self):
        # Video upload and analysis controls
        control_frame = ttk.Frame(self.video_tab)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(control_frame, text="Upload Video", command=self.upload_video).pack(side=tk.LEFT, padx=5)
        self.analyze_button = ttk.Button(control_frame, text="Analyze Video", command=self.analyze_video, state=tk.DISABLED)
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        self.stop_analysis_button = ttk.Button(control_frame, text="Stop Analysis", command=self.stop_video_analysis, state=tk.DISABLED)
        self.stop_analysis_button.pack(side=tk.LEFT, padx=5)
        
        # Video file info
        self.video_info_var = tk.StringVar(value="No video selected")
        ttk.Label(control_frame, textvariable=self.video_info_var).pack(side=tk.LEFT, padx=20)
        
        # Video display frame
        self.video_display_frame = ttk.Frame(self.video_tab)
        self.video_display_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.video_label = ttk.Label(self.video_display_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Video analysis results
        self.video_results_frame = ttk.LabelFrame(self.video_tab, text="Analysis Results")
        self.video_results_frame.pack(fill=tk.X, pady=10)
        
        self.video_results_text = scrolledtext.ScrolledText(self.video_results_frame, height=8)
        self.video_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_results_text.config(state=tk.DISABLED)
        
    def setup_dashboard_tab(self):
        # Statistics panel
        stats_frame = ttk.LabelFrame(self.dashboard_tab, text="Detection Statistics")
        stats_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Create statistics counters
        stats_inner = ttk.Frame(stats_frame)
        stats_inner.pack(fill=tk.X, padx=10, pady=10)
        
        # Person detections
        ttk.Label(stats_inner, text="People:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.person_count = tk.StringVar(value="0")
        ttk.Label(stats_inner, textvariable=self.person_count).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Vehicle detections
        ttk.Label(stats_inner, text="Vehicles:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.vehicle_count = tk.StringVar(value="0")
        ttk.Label(stats_inner, textvariable=self.vehicle_count).grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Weapon detections
        ttk.Label(stats_inner, text="Weapons:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.weapon_count = tk.StringVar(value="0")
        ttk.Label(stats_inner, textvariable=self.weapon_count).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Alerts count
        ttk.Label(stats_inner, text="Total Alerts:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.alert_count = tk.StringVar(value="0")
        ttk.Label(stats_inner, textvariable=self.alert_count).grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Alert history
        alert_history_frame = ttk.LabelFrame(self.dashboard_tab, text="Alert History")
        alert_history_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)
        
        self.alert_history_text = scrolledtext.ScrolledText(alert_history_frame)
        self.alert_history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.alert_history_text.config(state=tk.DISABLED)
        
        # Export button
        ttk.Button(self.dashboard_tab, text="Export Report", command=self.export_report).pack(pady=10)
        
    def setup_settings_tab(self):
        settings_frame = ttk.Frame(self.settings_tab)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Detection settings
        detection_frame = ttk.LabelFrame(settings_frame, text="Detection Settings")
        detection_frame.pack(fill=tk.X, pady=10)
        
        # Confidence threshold
        ttk.Label(detection_frame, text="Detection Confidence Threshold:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.confidence_var = tk.DoubleVar(value=settings.DETECTION_THRESHOLD)
        confidence_scale = ttk.Scale(detection_frame, from_=0.1, to=0.9, variable=self.confidence_var, orient=tk.HORIZONTAL, length=200)
        confidence_scale.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(detection_frame, textvariable=self.confidence_var).grid(row=0, column=2, padx=5, pady=5)
        
        # Border crossing settings
        border_frame = ttk.LabelFrame(settings_frame, text="Border Crossing Detection")
        border_frame.pack(fill=tk.X, pady=10)
        
        # Border line position
        ttk.Label(border_frame, text="Border Line Position:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.border_position_var = tk.StringVar(value="middle")
        ttk.Combobox(border_frame, textvariable=self.border_position_var, 
                    values=["top", "middle", "bottom", "custom"]).grid(row=0, column=1, padx=5, pady=5)
        
        # Border direction
        ttk.Label(border_frame, text="Alert Direction:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.border_direction_var = tk.StringVar(value="both")
        ttk.Combobox(border_frame, textvariable=self.border_direction_var, 
                    values=["north_to_south", "south_to_north", "both"]).grid(row=1, column=1, padx=5, pady=5)
        
        # Classes of interest
        classes_frame = ttk.LabelFrame(settings_frame, text="Classes of Interest")
        classes_frame.pack(fill=tk.X, pady=10)
        
        # Create checkboxes for common classes
        self.class_vars = {}
        common_classes = ["person", "car", "truck", "motorcycle", "bicycle", "backpack", "suitcase", "knife", "gun"]
        
        for i, cls in enumerate(common_classes):
            var = tk.BooleanVar(value=cls in settings.CLASSES_OF_INTEREST)
            self.class_vars[cls] = var
            ttk.Checkbutton(classes_frame, text=cls, variable=var).grid(row=i//3, column=i%3, sticky=tk.W, padx=10, pady=5)
        
        # Save settings button
        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings).pack(pady=10)
        
    def add_camera(self):
        # Open dialog to get camera source (could be a device index or IP camera URL)
        camera_source = simpledialog.askstring("Camera Source", "Enter camera index (0, 1, 2...) or URL:")
        if camera_source is None:
            return
            
        try:
            # Try to convert to integer for webcam index
            camera_id = camera_source
            try:
                camera_id = int(camera_source)
            except ValueError:
                # If not an integer, treat as URL
                pass
                
            # Create a new camera object
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                messagebox.showerror("Error", f"Could not open camera source: {camera_source}")
                return
                
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, settings.FPS)
                
            # Add to cameras dictionary
            self.cameras[camera_source] = {
                'cap': cap,
                'label': None,
                'active': True,
                'name': f"Camera {camera_source}"
            }
            
            # Create a label for this camera feed
            self.rebuild_camera_grid()
            
            # Start the camera feed if system is running
            if self.is_running:
                self.active_feeds.append(camera_source)
                
            messagebox.showinfo("Success", f"Camera {camera_source} added successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add camera: {str(e)}")
    
    def rebuild_camera_grid(self):
        # Clear existing camera displays
        for widget in self.feeds_frame.winfo_children():
            widget.destroy()
            
        # Calculate grid dimensions based on number of cameras
        num_cameras = len(self.cameras)
        if num_cameras == 0:
            return
            
        cols = min(2, num_cameras)  # Max 2 columns
        rows = (num_cameras + cols - 1) // cols  # Ceiling division
        
        # Configure grid weights
        for i in range(cols):
            self.feeds_frame.columnconfigure(i, weight=1)
        for i in range(rows):
            self.feeds_frame.rowconfigure(i, weight=1)
        
        # Create frames for each camera
        for i, (source, camera) in enumerate(self.cameras.items()):
            row = i // cols
            col = i % cols
            
            camera_frame = ttk.Frame(self.feeds_frame, borderwidth=2, relief=tk.GROOVE)
            camera_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            # Camera header with name and controls
            header = ttk.Frame(camera_frame)
            header.pack(fill=tk.X)
            
            ttk.Label(header, text=camera['name']).pack(side=tk.LEFT, anchor=tk.W)
            
            # Camera settings button
            ttk.Button(header, text="⚙️", width=3, 
                      command=lambda s=source: self.camera_settings(s)).pack(side=tk.RIGHT)
            
            # Create label for video feed
            video_frame = ttk.Frame(camera_frame)
            video_frame.pack(fill=tk.BOTH, expand=True)
            
            video_label = ttk.Label(video_frame)
            video_label.pack(fill=tk.BOTH, expand=True)
            self.cameras[source]['label'] = video_label
    
    def camera_settings(self, camera_id):
        # Dialog to change camera settings
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Settings for {self.cameras[camera_id]['name']}")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Camera Name:").pack(anchor=tk.W, padx=10, pady=5)
        name_var = tk.StringVar(value=self.cameras[camera_id]['name'])
        name_entry = ttk.Entry(dialog, textvariable=name_var, width=30)
        name_entry.pack(fill=tk.X, padx=10, pady=5)
        
        def save_settings():
            self.cameras[camera_id]['name'] = name_var.get()
            self.rebuild_camera_grid()
            dialog.destroy()
            
        ttk.Button(dialog, text="Save", command=save_settings).pack(pady=10)
    
    def remove_camera(self):
        if not self.cameras:
            messagebox.showinfo("Info", "No cameras to remove")
            return
            
        # Ask which camera to remove
        camera_list = [f"{camera['name']} ({source})" for source, camera in self.cameras.items()]
        camera_to_remove = simpledialog.askstring(
            "Remove Camera", 
            f"Enter camera to remove:\n{', '.join(camera_list)}"
        )
        
        # Find the camera by name or source
        found = False
        for source, camera in self.cameras.items():
            if str(source) == camera_to_remove or camera['name'] == camera_to_remove:
                camera_to_remove = source
                found = True
                break
                
        if found:
            # Release the camera
            self.cameras[camera_to_remove]['cap'].release()
            
            # Remove from active feeds if present
            if camera_to_remove in self.active_feeds:
                self.active_feeds.remove(camera_to_remove)
                
            # Remove from cameras dictionary
            del self.cameras[camera_to_remove]
            
            # Rebuild the UI for camera feeds
            self.rebuild_camera_grid()
            
            messagebox.showinfo("Success", f"Camera removed successfully")
        else:
            messagebox.showerror("Error", "Camera not found")
    
    def start_surveillance(self):
        if not self.cameras:
            messagebox.showinfo("Info", "Please add at least one camera first")
            return
            
        self.is_running = True
        self.active_feeds = list(self.cameras.keys())
        self.status_label.config(text="MONITORING", foreground="lime green")
        self.update_frames()
        
        # Log to alert panel
        self.add_alert("System", "Surveillance started")
        
    def stop_surveillance(self):
        self.is_running = False
        self.active_feeds = []
        self.status_label.config(text="STANDBY", foreground="yellow")
        
        # Log to alert panel
        self.add_alert("System", "Surveillance stopped")
        
    def update_frames(self):
        if not self.is_running:
            return
            
        for camera_id in self.active_feeds:
            camera = self.cameras[camera_id]
            ret, frame = camera['cap'].read()
            
            if ret:
                # Initialize object detector if not already done
                if 'detector' not in camera:
                    camera['detector'] = ObjectDetector()
                    camera['behavior_analyzer'] = BehaviorAnalyzer()
                    camera['weapon_detector'] = WeaponDetector()
                    camera['border_detector'] = BorderCrossingDetector()  # Add border detector
                    camera['visualizer'] = Visualizer()
                
                # Process frame with object detection
                detections = camera['detector'].detect(frame)
                
                # Check for weapons
                weapon_detections = camera['weapon_detector'].detect(frame)
                if weapon_detections:
                    detections.extend(weapon_detections)
                    # Update weapon count
                    current_count = int(self.weapon_count.get())
                    self.weapon_count.set(str(current_count + len(weapon_detections)))
                    
                    # Add weapon alert
                    for weapon in weapon_detections:
                        weapon_type = weapon[6]  # class_name
                        self.add_alert("WEAPON ALERT", f"{weapon_type} detected in {camera['name']}", is_critical=True)
                
                # Analyze behaviors
                behavior_alerts = camera['behavior_analyzer'].update(detections, frame)
                
                # Detect border crossings
                border_alerts = camera['border_detector'].detect(detections, frame)
                if border_alerts:
                    for alert in border_alerts:
                        self.add_alert("BORDER CROSSING", f"{alert['message']} in {camera['name']}", is_critical=True)
                        
                        # Update alert count
                        current_count = int(self.alert_count.get())
                        self.alert_count.set(str(current_count + 1))
                
                # Process behavior alerts
                if behavior_alerts:
                    for alert in behavior_alerts:
                        self.add_alert(f"Behavior Alert ({camera['name']})", alert['message'])
                        
                        # Update alert count
                        current_count = int(self.alert_count.get())
                        self.alert_count.set(str(current_count + 1))
                
                # Update statistics
                self.update_statistics(detections)
                
                # Combine all alerts
                all_alerts = behavior_alerts + border_alerts
                
                # Visualize results
                processed_frame = camera['visualizer'].draw_detections(frame.copy(), detections)
                processed_frame = camera['visualizer'].draw_alerts(processed_frame, all_alerts)
                processed_frame = camera['visualizer'].draw_border_lines(processed_frame, settings.BORDER_LINES)
                processed_frame = camera['visualizer'].add_info_overlay(processed_frame, len(detections))
                
                # Resize to fixed display resolution
                processed_frame = cv2.resize(processed_frame, (settings.DISPLAY_WIDTH, settings.DISPLAY_HEIGHT))
                
                # Convert to format suitable for tkinter
                cv_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cv_image)
                
                # Create PhotoImage
                tk_image = ImageTk.PhotoImage(image=pil_image)
                camera['label'].configure(image=tk_image)
                camera['label'].image = tk_image  # Keep a reference
        
        # Schedule the next update
        self.root.after(30, self.update_frames)
    
    def update_statistics(self, detections):
        """Update detection statistics for dashboard"""
        # resetting the counts back to 0
        self.person_count.set(0)
        self.vehicle_count.set(0)
        self.weapon_count.set(0)

        person_count = 0
        vehicle_count = 0
        
        for detection in detections:
            class_name = detection[6]
            if class_name == 'person':
                person_count += 1
            elif class_name in ['car', 'truck', 'motorcycle', 'bicycle']:
                vehicle_count += 1
        
        # Update only if there are new detections
        if person_count > 0:
            current = int(self.person_count.get())
            self.person_count.set(str(current + person_count))
            
        if vehicle_count > 0:
            current = int(self.vehicle_count.get())
            self.vehicle_count.set(str(current + vehicle_count))
    
    def add_alert(self, alert_type, message, is_critical=False):
        """Add an alert to the alert panel and history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_text = f"[{timestamp}] {alert_type}: {message}\n"
        
        # Add to alert panel
        self.alert_text.config(state=tk.NORMAL)
        if is_critical:
            self.alert_text.tag_config("critical", foreground="red", font=("Arial", 10, "bold"))
            self.alert_text.insert(tk.END, alert_text, "critical")
        else:
            self.alert_text.insert(tk.END, alert_text)
        self.alert_text.see(tk.END)
        self.alert_text.config(state=tk.DISABLED)
        
        # Add to alert history
        self.alert_history_text.config(state=tk.NORMAL)
        if is_critical:
            self.alert_history_text.tag_config("critical", foreground="red", font=("Arial", 10, "bold"))
            self.alert_history_text.insert(tk.END, alert_text, "critical")
        else:
            self.alert_history_text.insert(tk.END, alert_text)
        self.alert_history_text.see(tk.END)
        self.alert_history_text.config(state=tk.DISABLED)
        
        # Store in alert history list
        self.alert_history.append({
            'timestamp': timestamp,
            'type': alert_type,
            'message': message,
            'critical': is_critical
        })
    
    def upload_video(self):
        """Open file dialog to select a video file for analysis"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        
        if not video_path:
            return
            
        # Try to open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open the video file")
            return
            
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Update video info
        self.video_file = {
            'path': video_path,
            'cap': cap,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'width': width,
            'height': height,
            'current_frame': 0
        }
        
        # Update UI
        filename = os.path.basename(video_path)
        self.video_info_var.set(f"Video: {filename} | Duration: {duration:.1f}s | Resolution: {width}x{height}")
        self.analyze_button.config(state=tk.NORMAL)
        
        # Display first frame
        ret, frame = cap.read()
        if ret:
            self.display_video_frame(frame)
            
        # Reset results
        self.video_results_text.config(state=tk.NORMAL)
        self.video_results_text.delete(1.0, tk.END)
        self.video_results_text.config(state=tk.DISABLED)
    
    def display_video_frame(self, frame):
        """Display a frame in the video analysis tab"""
        cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image)
        
        # Get label dimensions
        label_width = self.video_label.winfo_width() or 800
        label_height = self.video_label.winfo_height() or 600
        
        # Maintain aspect ratio
        img_ratio = pil_image.width / pil_image.height
        target_ratio = label_width / label_height
        
        if img_ratio > target_ratio:
            # Image is wider than the target area
            new_width = label_width
            new_height = int(label_width / img_ratio)
        else:
            # Image is taller than the target area
            new_height = label_height
            new_width = int(label_height * img_ratio)
        
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        tk_image = ImageTk.PhotoImage(image=pil_image)
        self.video_label.configure(image=tk_image)
        self.video_label.image = tk_image  # Keep a reference
    
    def analyze_video(self):
        """Analyze the uploaded video file"""
        if not self.video_file:
            messagebox.showinfo("Info", "Please upload a video file first")
            return
            
        # Disable analyze button during analysis
        self.analyze_button.config(state=tk.DISABLED)
        self.stop_analysis_button.config(state=tk.NORMAL)
        
        # Clear previous results
        self.video_results_text.config(state=tk.NORMAL)
        self.video_results_text.delete(1.0, tk.END)
        self.video_results_text.insert(tk.END, "Analysis in progress...\n")
        self.video_results_text.config(state=tk.DISABLED)
        
        # Start analysis in a separate thread
        self.video_analysis_running = True
        threading.Thread(target=self.run_video_analysis, daemon=True).start()
    
    def run_video_analysis(self):
        """Run video analysis in a background thread"""
        try:
            # Initialize detectors
            object_detector = ObjectDetector()
            behavior_analyzer = BehaviorAnalyzer()
            weapon_detector = WeaponDetector()
            border_detector = BorderCrossingDetector()  # Add border crossing detector
            visualizer = Visualizer()
            
            # Statistics
            total_frames = self.video_file['frame_count']
            processed_frames = 0
            detections_count = 0
            alerts_count = 0
            weapon_detections = 0
            border_crossings = 0  # Track border crossings
            
            # Reset video to beginning
            self.video_file['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Process each frame
            while self.video_analysis_running:
                ret, frame = self.video_file['cap'].read()
                if not ret:
                    break
                    
                processed_frames += 1
                
                # Update progress every 10 frames
                if processed_frames % 10 == 0:
                    progress = processed_frames / total_frames * 100
                    self.update_analysis_progress(progress, processed_frames)
                
                # Object detection
                detections = object_detector.detect(frame)
                
                # Check for weapons
                weapons = weapon_detector.detect(frame)
                if weapons:
                    detections.extend(weapons)
                    weapon_detections += len(weapons)
                    
                    # Log weapon detections
                    for weapon in weapons:
                        weapon_type = weapon[6]  # class_name
                        self.add_analysis_result(f"WEAPON DETECTED: {weapon_type} at frame {processed_frames}")
                
                detections_count += len(detections)
                
                # Behavior analysis
                behavior_alerts = behavior_analyzer.update(detections, frame)
                if behavior_alerts:
                    alerts_count += len(behavior_alerts)
                    
                    # Log alerts
                    for alert in behavior_alerts:
                        self.add_analysis_result(f"ALERT: {alert['message']} at frame {processed_frames}")
                
                # Border crossing detection
                border_alerts = border_detector.detect(detections, frame)
                if border_alerts:
                    border_crossings += len(border_alerts)
                    
                    # Log border crossing alerts
                    for alert in border_alerts:
                        self.add_analysis_result(f"BORDER CROSSING: {alert['message']} at frame {processed_frames}")
                        
                        # Add to alert history with critical flag
                        self.alert_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'type': "Border Crossing",
                            'message': f"{alert['message']} (Video Analysis)",
                            'critical': True
                        })
                
                # Combine all alerts
                all_alerts = behavior_alerts + border_alerts
                if all_alerts:
                    alerts_count += len(all_alerts)
                    
                    # Log behavior alerts
                    for alert in behavior_alerts:
                        self.add_analysis_result(f"ALERT: {alert['message']} at frame {processed_frames}")
                
                # Visualize results
                processed_frame = visualizer.draw_detections(frame.copy(), detections)
                processed_frame = visualizer.draw_alerts(processed_frame, all_alerts)
                processed_frame = visualizer.draw_border_lines(processed_frame, settings.BORDER_LINES)
                processed_frame = visualizer.add_info_overlay(processed_frame, len(detections))
                
                # Display the processed frame
                self.root.after_idle(lambda f=processed_frame: self.display_video_frame(f))
                
                # Slow down processing to make it viewable
                time.sleep(0.03)
            
            # Analysis complete
            if processed_frames >= total_frames:
                self.add_analysis_result("\nAnalysis complete!")
                self.add_analysis_result(f"Total frames processed: {processed_frames}")
                self.add_analysis_result(f"Total detections: {detections_count}")
                self.add_analysis_result(f"Weapon detections: {weapon_detections}")
                self.add_analysis_result(f"Border crossings detected: {border_crossings}")
                self.add_analysis_result(f"Suspicious behavior alerts: {alerts_count - border_crossings}")
                self.add_analysis_result(f"Total alerts: {alerts_count}")
            else:
                self.add_analysis_result("\nAnalysis stopped by user.")
                
            # Reset UI
            self.root.after_idle(self.reset_video_analysis_ui)
            
        except Exception as e:
            self.add_analysis_result(f"\nError during analysis: {str(e)}")
            self.root.after_idle(self.reset_video_analysis_ui)
    
    def update_analysis_progress(self, progress, frame_number):
        """Update the progress display in the UI thread"""
        def update():
            self.video_results_text.config(state=tk.NORMAL)
            self.video_results_text.delete(1.0, 2.0)  # Replace first line
            self.video_results_text.insert(1.0, f"Analysis in progress: {progress:.1f}% (Frame {frame_number})\n")
            self.video_results_text.config(state=tk.DISABLED)
        self.root.after_idle(update)
    
    def add_analysis_result(self, message):
        """Add a message to the analysis results"""
        def update():
            self.video_results_text.config(state=tk.NORMAL)
            self.video_results_text.insert(tk.END, message + "\n")
            self.video_results_text.see(tk.END)
            self.video_results_text.config(state=tk.DISABLED)
        self.root.after_idle(update)
    
    def stop_video_analysis(self):
        """Stop the video analysis"""
        self.video_analysis_running = False
        self.stop_analysis_button.config(state=tk.DISABLED)
    
    def reset_video_analysis_ui(self):
        """Reset the video analysis UI after analysis is complete"""
        self.analyze_button.config(state=tk.NORMAL)
        self.stop_analysis_button.config(state=tk.DISABLED)
    
    def export_report(self):
        """Export a report of all alerts and statistics"""
        if not self.alert_history:
            messagebox.showinfo("Info", "No alerts to export")
            return
            
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Surveillance Report"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w') as f:
                # Write header
                f.write("BORDER SURVEILLANCE SYSTEM - ACTIVITY REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write statistics
                f.write("DETECTION STATISTICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"People detected: {self.person_count.get()}\n")
                f.write(f"Vehicles detected: {self.vehicle_count.get()}\n")
                f.write(f"Weapons detected: {self.weapon_count.get()}\n")
                f.write(f"Total alerts: {self.alert_count.get()}\n\n")
                
                # Write alert history
                f.write("ALERT HISTORY\n")
                f.write("-" * 20 + "\n")
                for alert in self.alert_history:
                    priority = "HIGH" if alert['critical'] else "Normal"
                    f.write(f"[{alert['timestamp']}] {alert['type']} ({priority}): {alert['message']}\n")
            
            messagebox.showinfo("Success", f"Report saved to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {str(e)}")
    
    def save_settings(self):
        """Save the current settings"""
        # Update detection threshold
        settings.DETECTION_THRESHOLD = self.confidence_var.get()
        
        # Update classes of interest
        new_classes = []
        for cls, var in self.class_vars.items():
            if var.get():
                new_classes.append(cls)
        
        settings.CLASSES_OF_INTEREST = new_classes
        
        # Update border line settings
        border_position = self.border_position_var.get()
        border_direction = self.border_direction_var.get()
        
        # Configure border lines based on position
        if border_position == "top":
            y_pos = int(settings.FRAME_HEIGHT * 0.25)
        elif border_position == "middle":
            y_pos = int(settings.FRAME_HEIGHT * 0.5)
        elif border_position == "bottom":
            y_pos = int(settings.FRAME_HEIGHT * 0.75)
        else:  # custom - keep existing
            y_pos = settings.BORDER_LINES[0]['points'][0][1] if settings.BORDER_LINES else int(settings.FRAME_HEIGHT * 0.5)
        
        # Update border lines
        settings.BORDER_LINES = [
            {
                'id': 'main_border',
                'points': [(0, y_pos), (settings.FRAME_WIDTH, y_pos)],
                'direction': border_direction
            }
        ]
        
        messagebox.showinfo("Success", "Settings saved successfully")
    
    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            # Stop surveillance if running
            self.is_running = False
            
            # Release all camera resources
            for camera_id, camera in self.cameras.items():
                if 'cap' in camera and camera['cap'] is not None:
                    camera['cap'].release()
            
            # Release video file if open
            if self.video_file and 'cap' in self.video_file:
                self.video_file['cap'].release()
            
            self.root.destroy()

if __name__ == "__main__":
    app = BorderSurveillanceSystem()
    app.root.mainloop()
