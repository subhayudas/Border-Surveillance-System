import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import time
import threading
from datetime import datetime
import numpy as np
import webbrowser
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BorderSurveillance')

# Import detection and visualization modules
from src.detection import ObjectDetector, BehaviorAnalyzer, FenceTamperingDetector, ItemDetector, BorderCrossingDetector
from src.visualization import Visualizer, GeoMapVisualizer
from config import settings

# Optional MQTT support - don't fail if not available
try:
    import paho.mqtt.client as mqtt
    mqtt_available = True
except ImportError:
    mqtt_available = False
    logger.warning("MQTT not available. Install paho-mqtt for alert distribution.")

class BorderSurveillanceSystem:
    def __init__(self):
        self.cameras = {}  # Dictionary to store multiple camera objects
        self.active_feeds = []  # List to track active camera feeds
        self.is_running = False
        self.alert_history = []  # Store alert history
        self.video_files = []  # List to store multiple uploaded videos
        self.current_video_index = 0  # Index of the currently displayed video
        self.video_player = None  # For video playback
        self.geo_map_visualizer = GeoMapVisualizer()  # Initialize geographical map visualizer
        
        # Generate mock data for the map to demonstrate functionality
        self.geo_map_visualizer.generate_mock_data(num_points=30, spread_km=10)
        
        # Try to setup MQTT client for alerts if enabled
        self.mqtt_client = None
        if settings.DASHBOARD_ALERTS_ENABLED and mqtt_available:
            try:
                self.setup_mqtt()
            except Exception as e:
                logger.error(f"Failed to connect to MQTT broker: {str(e)}")
        
        self.setup_ui()
        
    def setup_mqtt(self):
        """Setup MQTT client for alert distribution"""
        if not mqtt_available:
            return
            
        try:
            # Use the new MQTT client API version
            self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.mqtt_client.connect(settings.MQTT_BROKER, settings.MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            logger.info(f"Connected to MQTT broker at {settings.MQTT_BROKER}:{settings.MQTT_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {str(e)}")
            self.mqtt_client = None
        
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
        
        # Geographical Map tab
        self.map_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.map_tab, text="Geo Map")
        
        # Settings tab
        self.settings_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.settings_tab, text="Settings")
        
        # Setup each tab
        self.setup_monitoring_tab()
        self.setup_video_tab()
        self.setup_dashboard_tab()
        self.setup_map_tab()
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
        
        ttk.Button(control_frame, text="Upload Videos", command=self.upload_video).pack(side=tk.LEFT, padx=5)
        self.analyze_button = ttk.Button(control_frame, text="Analyze All Videos", command=self.analyze_video, state=tk.DISABLED)
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        self.stop_analysis_button = ttk.Button(control_frame, text="Stop Analysis", command=self.stop_video_analysis, state=tk.DISABLED)
        self.stop_analysis_button.pack(side=tk.LEFT, padx=5)
        
        # Video navigation buttons for multiple videos
        self.video_nav_frame = ttk.Frame(control_frame)
        self.video_nav_frame.pack(side=tk.LEFT, padx=20)
        self.prev_video_btn = ttk.Button(self.video_nav_frame, text="◀ Previous", command=self.show_previous_video, state=tk.DISABLED)
        self.prev_video_btn.pack(side=tk.LEFT, padx=5)
        self.next_video_btn = ttk.Button(self.video_nav_frame, text="Next ▶", command=self.show_next_video, state=tk.DISABLED)
        self.next_video_btn.pack(side=tk.LEFT, padx=5)
        
        # Video file info
        self.video_info_var = tk.StringVar(value="No videos selected")
        self.video_count_var = tk.StringVar(value="")
        ttk.Label(control_frame, textvariable=self.video_info_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, textvariable=self.video_count_var).pack(side=tk.LEFT, padx=5)
        
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
        ttk.Label(stats_inner, text="Items:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.item_count = tk.StringVar(value="0")
        ttk.Label(stats_inner, textvariable=self.item_count).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
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
        
    def setup_map_tab(self):
        """Setup the geographical map tab"""
        # Control panel
        control_frame = ttk.Frame(self.map_tab)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Map controls
        map_control_frame = ttk.LabelFrame(control_frame, text="Map Controls")
        map_control_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Refresh map button
        self.refresh_map_button = ttk.Button(map_control_frame, text="Refresh Map", command=self.refresh_map)
        self.refresh_map_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Open in browser button (kept for compatibility)
        self.open_browser_button = ttk.Button(map_control_frame, text="Open in Browser", command=self.open_map_in_browser)
        self.open_browser_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Clear detections button
        self.clear_detections_button = ttk.Button(map_control_frame, text="Clear Detections", command=self.clear_map_detections)
        self.clear_detections_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Add mock data button
        self.add_mock_data_button = ttk.Button(map_control_frame, text="Add Mock Data", command=self.add_mock_map_data)
        self.add_mock_data_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Toggle heatmap button
        self.heatmap_var = tk.BooleanVar(value=True)
        self.heatmap_checkbutton = ttk.Checkbutton(
            map_control_frame, 
            text="Show Heatmap", 
            variable=self.heatmap_var, 
            command=self.toggle_heatmap
        )
        self.heatmap_checkbutton.pack(side=tk.LEFT, padx=5, pady=5)
        
        # GPS status
        self.gps_status_frame = ttk.LabelFrame(control_frame, text="GPS Status")
        self.gps_status_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.gps_status_label = ttk.Label(self.gps_status_frame, text="GPS: Connecting...")
        self.gps_status_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.location_label = ttk.Label(self.gps_status_frame, text="Location: N/A")
        self.location_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Create a frame to display the map directly
        self.map_display_frame = ttk.LabelFrame(self.map_tab, text="Real-time Detection Map")
        self.map_display_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)
        
        # Label to hold the map image
        self.map_display_label = ttk.Label(self.map_display_frame)
        self.map_display_label.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame to display the map info (kept smaller now that we show the map)
        self.map_info_frame = ttk.LabelFrame(self.map_tab, text="Map Information")
        self.map_info_frame.pack(fill=tk.X, pady=10)
        
        self.map_info_text = scrolledtext.ScrolledText(self.map_info_frame, height=4)
        self.map_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.map_info_text.config(state=tk.NORMAL)
        self.map_info_text.insert(tk.END, "Geographic Map Initialized. Detections will be displayed on the map with your current location.\n")
        self.map_info_text.insert(tk.END, "The map updates automatically as new detections are found.\n")
        self.map_info_text.config(state=tk.DISABLED)
        
        # Start GPS status update thread
        threading.Thread(target=self.update_gps_status, daemon=True).start()
        
        # Start map update thread
        threading.Thread(target=self.update_map_display, daemon=True).start()
        
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
        
        # Map settings
        map_settings_frame = ttk.LabelFrame(settings_frame, text="Map Settings")
        map_settings_frame.pack(fill=tk.X, pady=10)
        
        # Default location setting
        ttk.Label(map_settings_frame, text="Default Latitude:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.default_lat_var = tk.StringVar(value=str(settings.DEFAULT_LAT))
        ttk.Entry(map_settings_frame, textvariable=self.default_lat_var, width=15).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(map_settings_frame, text="Default Longitude:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.default_lon_var = tk.StringVar(value=str(settings.DEFAULT_LON))
        ttk.Entry(map_settings_frame, textvariable=self.default_lon_var, width=15).grid(row=0, column=3, padx=5, pady=5)
        
        # Map zoom level
        ttk.Label(map_settings_frame, text="Map Zoom Level:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.map_zoom_var = tk.IntVar(value=settings.MAP_ZOOM_LEVEL)
        zoom_scale = ttk.Scale(map_settings_frame, from_=5, to=18, variable=self.map_zoom_var, orient=tk.HORIZONTAL, length=200)
        zoom_scale.grid(row=1, column=1, columnspan=2, padx=5, pady=5)
        ttk.Label(map_settings_frame, textvariable=self.map_zoom_var).grid(row=1, column=3, padx=5, pady=5)
        
        # Map tile provider
        ttk.Label(map_settings_frame, text="Map Tile Provider:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.map_tile_var = tk.StringVar(value=settings.MAP_TILE_PROVIDER)
        ttk.Combobox(map_settings_frame, textvariable=self.map_tile_var, 
                    values=["OpenStreetMap", "Stamen Terrain", "Stamen Toner", "CartoDB positron"]).grid(row=2, column=1, columnspan=3, padx=5, pady=5, sticky=tk.W)
        
        # Update interval
        ttk.Label(map_settings_frame, text="Map Update Interval (seconds):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.map_update_var = tk.IntVar(value=settings.MAP_UPDATE_INTERVAL)
        update_scale = ttk.Scale(map_settings_frame, from_=1, to=30, variable=self.map_update_var, orient=tk.HORIZONTAL, length=200)
        update_scale.grid(row=3, column=1, columnspan=2, padx=5, pady=5)
        ttk.Label(map_settings_frame, textvariable=self.map_update_var).grid(row=3, column=3, padx=5, pady=5)
        
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
        """Update frames from all active cameras"""
        if not self.is_running:
            return
        
        for camera_id in self.active_feeds:
            if camera_id in self.cameras:
                camera = self.cameras[camera_id]
                
                # Check if camera is a video file that has ended
                if camera.get('is_video', False) and camera.get('cap') is not None:
                    if not camera['cap'].isOpened():
                        continue
                
                # Read frame
                ret, frame = camera['cap'].read()
                
                if not ret:
                    # Handle video end or camera error
                    if camera.get('is_video', False):
                        # Restart video from beginning
                        camera['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = camera['cap'].read()
                        if not ret:
                            continue
                    else:
                        continue
                
                # Resize frame for consistent display
                frame = cv2.resize(frame, (settings.DISPLAY_WIDTH, settings.DISPLAY_HEIGHT))
                
                # Make a deep copy for detection to avoid modification issues
                detection_frame = frame.copy()
                
                # Perform object detection using separate thread for better UI responsiveness
                def process_frame(camera_id, frame):
                    # Perform detection
                    detector = ObjectDetector()
                    detections = detector.detect(frame)
                    
                    # Analyze behaviors
                    behavior_analyzer = BehaviorAnalyzer()
                    behavior_alerts = behavior_analyzer.analyze(detections, frame)
                    
                    # Check for border crossings
                    border_detector = BorderCrossingDetector()
                    border_alerts = border_detector.detect(detections, frame)
                    
                    # Check for fence tampering
                    fence_detector = FenceTamperingDetector()
                    fence_alerts = fence_detector.detect(frame)
                    
                    # Check for prohibited items
                    item_detector = ItemDetector()
                    item_alerts = item_detector.detect(detections, frame)
                    
                    # Combine all alerts
                    alerts = behavior_alerts + border_alerts + fence_alerts + item_alerts
                    
                    # Add detections to the geographical map
                    if hasattr(self, 'geo_map_visualizer'):
                        for detection in detections:
                            _, _, _, _, confidence, _, class_name = detection
                            # Add each detection to the geo map
                            self.geo_map_visualizer.add_detection(class_name, confidence=confidence)
                            
                            # Update map info when a new detection is added
                            self.update_map_info(f"New detection added to map: {class_name} (conf: {confidence:.2f})")
                    
                    # Update statistics with new detections
                    self.update_statistics(detections)
                    
                    # Process alerts
                    for alert in alerts:
                        is_critical = alert.get('critical', False)
                        self.add_alert(alert['type'], alert['message'], is_critical)
                    
                    # Visualize results
                    visualizer = Visualizer()
                    
                    # Draw detection boxes
                    frame_with_detections = visualizer.draw_detections(frame.copy(), detections)
                    
                    # Draw alerts if any
                    if alerts:
                        frame_with_detections = visualizer.draw_alerts(frame_with_detections, alerts)
                    
                    # Draw borders
                    frame_with_detections = visualizer.draw_border_lines(frame_with_detections, settings.BORDER_LINES)
                    
                    # Draw fence regions if defined
                    frame_with_detections = visualizer.draw_fence_regions(frame_with_detections, settings.FENCE_REGIONS)
                    
                    # Add info overlay
                    frame_with_detections = visualizer.add_info_overlay(frame_with_detections, len(detections))
                    
                    # Update the display
                    self.update_camera_display(camera_id, frame_with_detections)
                
                # Start processing thread
                threading.Thread(target=process_frame, args=(camera_id, detection_frame), daemon=True).start()
        
        # Schedule the next update
        self.root.after(10, self.update_frames)
    
    def update_statistics(self, detections):
        """Update detection statistics for dashboard"""
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
        """Open file dialog to select multiple video files for analysis"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
        video_paths = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=filetypes
        )
        
        if not video_paths:
            return
        
        # Clear existing videos
        for video in self.video_files:
            if 'cap' in video and video['cap'] is not None:
                video['cap'].release()
        
        self.video_files = []
        self.current_video_index = 0
        
        valid_videos = 0
        
        for video_path in video_paths:
            # Try to open the video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                messagebox.showwarning("Warning", f"Could not open the video file: {os.path.basename(video_path)}")
                continue
                
            # Get video info
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Add to video list
            self.video_files.append({
                'path': video_path,
                'cap': cap,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'width': width,
                'height': height,
                'current_frame': 0,
                'analyzed': False,
                'results': [],
                'filename': os.path.basename(video_path)
            })
            
            valid_videos += 1
        
        if valid_videos == 0:
            self.video_info_var.set("No valid videos selected")
            return
            
        # Update UI for video navigation
        if valid_videos > 1:
            self.next_video_btn.config(state=tk.NORMAL)
            self.prev_video_btn.config(state=tk.DISABLED)  # Start with first video
        
        # Update UI
        self.update_video_info()
        self.analyze_button.config(state=tk.NORMAL)
        
        # Display first frame of first video
        self.display_current_video()
            
        # Reset results
        self.video_results_text.config(state=tk.NORMAL)
        self.video_results_text.delete(1.0, tk.END)
        self.video_results_text.config(state=tk.DISABLED)
    
    def update_video_info(self):
        """Update the video info display for the current video"""
        if not self.video_files:
            self.video_info_var.set("No videos selected")
            self.video_count_var.set("")
            return
            
        current_video = self.video_files[self.current_video_index]
        filename = current_video['filename']
        duration = current_video['duration']
        width = current_video['width']
        height = current_video['height']
        
        self.video_info_var.set(f"Video: {filename} | Duration: {duration:.1f}s | Resolution: {width}x{height}")
        self.video_count_var.set(f"Video {self.current_video_index + 1} of {len(self.video_files)}")
    
    def show_next_video(self):
        """Display the next video in the list"""
        if self.current_video_index < len(self.video_files) - 1:
            self.current_video_index += 1
            self.update_video_info()
            self.display_current_video()
            
            # Update navigation buttons
            self.prev_video_btn.config(state=tk.NORMAL)
            if self.current_video_index >= len(self.video_files) - 1:
                self.next_video_btn.config(state=tk.DISABLED)
    
    def show_previous_video(self):
        """Display the previous video in the list"""
        if self.current_video_index > 0:
            self.current_video_index -= 1
            self.update_video_info()
            self.display_current_video()
            
            # Update navigation buttons
            self.next_video_btn.config(state=tk.NORMAL)
            if self.current_video_index <= 0:
                self.prev_video_btn.config(state=tk.DISABLED)
    
    def display_current_video(self):
        """Display the first frame of the current video"""
        if not self.video_files:
            return
            
        current_video = self.video_files[self.current_video_index]
        cap = current_video['cap']
        
        # Reset to first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Read the frame
        ret, frame = cap.read()
        if ret:
            self.display_video_frame(frame)
            current_video['current_frame'] = 0
        
        # If this video has been analyzed, show its results
        if current_video['analyzed'] and current_video['results']:
            self.video_results_text.config(state=tk.NORMAL)
            self.video_results_text.delete(1.0, tk.END)
            for result in current_video['results']:
                self.video_results_text.insert(tk.END, result + "\n")
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
        """Analyze all the uploaded video files"""
        if not self.video_files:
            messagebox.showinfo("Info", "Please upload videos first")
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
        """Run video analysis in a background thread for all videos"""
        try:
            # Initialize detectors
            object_detector = ObjectDetector()
            behavior_analyzer = BehaviorAnalyzer()
            item_detector = ItemDetector()
            border_detector = BorderCrossingDetector()
            visualizer = Visualizer()
            
            # Combined statistics for all videos
            total_detections_count = 0
            total_alerts_count = 0
            total_item_detections = 0
            total_border_crossings = 0
            total_frames_processed = 0
            
            # Process each video
            for video_index, video in enumerate(self.video_files):
                if not self.video_analysis_running:
                    break
                
                # Skip already analyzed videos
                if video['analyzed']:
                    continue
                
                # Update UI to show which video is being processed
                self.add_analysis_result(f"\nProcessing video {video_index + 1} of {len(self.video_files)}: {video['filename']}")
                
                # Statistics for this video
                video_frames = video['frame_count']
                processed_frames = 0
                detections_count = 0
                people_count = 0
                vehicle_count = 0
                item_count = 0
                border_crossings = 0
                suspicious_behaviors = 0
                video_results = []
                
                # Reset video to beginning
                video['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # Process each frame
                while self.video_analysis_running:
                    ret, frame = video['cap'].read()
                    if not ret:
                        break
                        
                    processed_frames += 1
                    total_frames_processed += 1
                    
                    # Update progress every 10 frames
                    if processed_frames % 10 == 0:
                        progress = processed_frames / video_frames * 100
                        self.update_analysis_progress(progress, processed_frames, video_index)
                    
                    # Get detections from different detectors
                    general_detections = object_detector.detect(frame)
                    detected_items = item_detector.detect(frame)
                    
                    # Check for items
                    if detected_items:
                        general_detections.extend(detected_items)
                        item_count += len(detected_items)
                        total_item_detections += len(detected_items)
                        
                        # Log item detections
                        for item in detected_items:
                            item_type = item[6]  # class_name
                            result_message = f"ITEM DETECTED: {item_type} at frame {processed_frames}"
                            video_results.append(result_message)
                            self.add_analysis_result(result_message)
                    
                    detections_count += len(general_detections)
                    total_detections_count += len(general_detections)
                    
                    # Behavior analysis
                    behavior_alerts = behavior_analyzer.update(general_detections, frame)
                    
                    # Detect border crossings
                    border_alerts = border_detector.detect(general_detections, frame)
                    if border_alerts:
                        border_crossings += len(border_alerts)
                        total_border_crossings += len(border_alerts)
                        
                        # Log border crossing alerts
                        for alert in border_alerts:
                            result_message = f"BORDER CROSSING: {alert['message']} at frame {processed_frames}"
                            video_results.append(result_message)
                            self.add_analysis_result(result_message)
                            
                            # Add to alert history with critical flag
                            self.alert_history.append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'type': "Border Crossing",
                                'message': f"{alert['message']} (Video {video_index + 1})",
                                'critical': True
                            })
                    
                    # Combine all alerts
                    all_alerts = behavior_alerts + border_alerts
                    if all_alerts:
                        alerts_count += len(all_alerts)
                        total_alerts_count += len(all_alerts)
                        
                        # Log behavior alerts
                        for alert in behavior_alerts:
                            result_message = f"ALERT: {alert['message']} at frame {processed_frames}"
                            video_results.append(result_message)
                            self.add_analysis_result(result_message)
                    
                    # Visualize results
                    processed_frame = visualizer.draw_detections(frame.copy(), general_detections)
                    processed_frame = visualizer.draw_alerts(processed_frame, all_alerts)
                    processed_frame = visualizer.draw_border_lines(processed_frame, settings.BORDER_LINES)
                    processed_frame = visualizer.add_info_overlay(processed_frame, len(general_detections))
                    
                    # Add video name overlay
                    cv2.putText(
                        processed_frame, 
                        f"Video {video_index + 1}: {video['filename']}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 255, 255), 
                        2
                    )
                    
                    # Display the processed frame
                    self.root.after_idle(lambda f=processed_frame: self.display_video_frame(f))
                    
                    # Slow down processing to make it viewable
                    time.sleep(0.03)
                
                # Video analysis complete
                if processed_frames >= video_frames:
                    video_summary = f"\nAnalysis complete for video {video_index + 1}: {video['filename']}"
                    video_results.append(video_summary)
                    video_results.append(f"Total frames processed: {processed_frames}")
                    video_results.append(f"Total detections: {detections_count}")
                    video_results.append(f"Item detections: {item_count}")
                    video_results.append(f"Border crossings detected: {border_crossings}")
                    video_results.append(f"Suspicious behavior alerts: {alerts_count - border_crossings}")
                    video_results.append(f"Total alerts: {alerts_count}")
                    
                    for result in video_results[-7:]:  # Add the summary to analysis results
                        self.add_analysis_result(result)
                
                # Mark this video as analyzed and store its results
                video['analyzed'] = True
                video['results'] = video_results
                
                # If analysis was stopped, break out of the loop
                if not self.video_analysis_running:
                    self.add_analysis_result("\nAnalysis stopped by user.")
                    break
            
            # All videos processed, show final summary
            if self.video_analysis_running:
                self.add_analysis_result("\n=== FINAL ANALYSIS SUMMARY ===")
                self.add_analysis_result(f"Total videos processed: {sum(1 for v in self.video_files if v['analyzed'])}")
                self.add_analysis_result(f"Total frames analyzed: {total_frames_processed}")
                self.add_analysis_result(f"Total detections: {total_detections_count}")
                self.add_analysis_result(f"Total item detections: {total_item_detections}")
                self.add_analysis_result(f"Total border crossings: {total_border_crossings}")
                self.add_analysis_result(f"Total alerts: {total_alerts_count}")
            
            # Reset UI
            self.root.after_idle(self.reset_video_analysis_ui)
            
        except Exception as e:
            self.add_analysis_result(f"\nError during analysis: {str(e)}")
            self.root.after_idle(self.reset_video_analysis_ui)
    
    def update_analysis_progress(self, progress, frame_number, video_index):
        """Update the progress display in the UI thread"""
        def update():
            self.video_results_text.config(state=tk.NORMAL)
            self.video_results_text.delete(1.0, 2.0)  # Replace first line
            self.video_results_text.insert(1.0, f"Analyzing video {video_index + 1}/{len(self.video_files)}: {progress:.1f}% (Frame {frame_number})\n")
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
                f.write(f"Items detected: {self.item_count.get()}\n")
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
        
        # Update map settings
        try:
            settings.DEFAULT_LAT = float(self.default_lat_var.get())
            settings.DEFAULT_LON = float(self.default_lon_var.get())
            settings.MAP_ZOOM_LEVEL = int(self.map_zoom_var.get())
            settings.MAP_TILE_PROVIDER = self.map_tile_var.get()
            settings.MAP_UPDATE_INTERVAL = int(self.map_update_var.get())
            
            # Update the map visualizer with new settings if it exists
            if hasattr(self, 'geo_map_visualizer'):
                self.geo_map_visualizer.location = [settings.DEFAULT_LAT, settings.DEFAULT_LON]
                # Force map update
                self.refresh_map()
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter valid numeric values for map settings: {str(e)}")
            return
        
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
            
            # Release video files if open
            for video in self.video_files:
                if 'cap' in video and video['cap'] is not None:
                    video['cap'].release()
            
            self.root.destroy()
    
    def update_gps_status(self):
        """Update GPS status periodically"""
        while True:
            if hasattr(self, 'geo_map_visualizer'):
                if self.geo_map_visualizer.connected_to_gps:
                    status_text = "GPS: Connected"
                    location_text = f"Location: {self.geo_map_visualizer.location[0]:.6f}, {self.geo_map_visualizer.location[1]:.6f}"
                else:
                    status_text = "GPS: Not Connected"
                    location_text = f"Location: Using Default ({settings.DEFAULT_LAT:.6f}, {settings.DEFAULT_LON:.6f})"
                
                # Update labels using thread-safe method
                self.root.after(0, lambda: self.update_status_labels(status_text, location_text))
            
            time.sleep(2)  # Update every 2 seconds
    
    def update_status_labels(self, status_text, location_text):
        """Update GPS status labels (thread-safe)"""
        if hasattr(self, 'gps_status_label'):
            self.gps_status_label.config(text=status_text)
        if hasattr(self, 'location_label'):
            self.location_label.config(text=location_text)
    
    def refresh_map(self):
        """Refresh the geographical map"""
        if hasattr(self, 'geo_map_visualizer'):
            self.geo_map_visualizer.update_map()
            messagebox.showinfo("Map Refreshed", "The geographical map has been updated.")
            
            # Update map info
            self.update_map_info("Map manually refreshed.")
    
    def open_map_in_browser(self):
        """Open the geographical map in a web browser"""
        if hasattr(self, 'geo_map_visualizer'):
            self.geo_map_visualizer.open_map_in_browser()
            
            # Update map info
            self.update_map_info("Map opened in browser.")
    
    def clear_map_detections(self):
        """Clear all detection points from the map"""
        if hasattr(self, 'geo_map_visualizer'):
            # Clear detection points
            self.geo_map_visualizer.detection_points = []
            
            # Update the map
            self.refresh_map()
            self.update_map_info("All detection points have been cleared from the map.")
    
    def update_map_info(self, message):
        """Update the map information text box"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if hasattr(self, 'map_info_text'):
            self.map_info_text.config(state=tk.NORMAL)
            self.map_info_text.insert(tk.END, f"[{current_time}] {message}\n")
            self.map_info_text.see(tk.END)  # Scroll to end
            self.map_info_text.config(state=tk.DISABLED)

    def update_camera_display(self, camera_id, frame):
        """Update the camera display with processed frame (thread-safe)"""
        try:
            if camera_id in self.cameras and 'label' in self.cameras[camera_id]:
                # Convert to format suitable for tkinter
                cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cv_image)
                
                # Create PhotoImage
                tk_image = ImageTk.PhotoImage(image=pil_image)
                
                # Update the label in a thread-safe way
                def update_label():
                    self.cameras[camera_id]['label'].configure(image=tk_image)
                    self.cameras[camera_id]['label'].image = tk_image  # Keep a reference
                
                self.root.after(0, update_label)
        except Exception as e:
            print(f"Error updating camera display: {str(e)}")

    def update_map_display(self):
        """Thread to update the map display in the dashboard"""
        while True:
            if hasattr(self, 'geo_map_visualizer') and hasattr(self, 'map_display_label'):
                try:
                    # Generate a map image
                    map_width = self.map_display_label.winfo_width() or 800
                    map_height = self.map_display_label.winfo_height() or 600
                    
                    # Only update if we have a reasonable size
                    if map_width > 50 and map_height > 50:
                        map_image = self.geo_map_visualizer.get_map_image(map_width, map_height)
                        
                        # Update the label with the new map image
                        self.root.after(0, lambda: self.update_map_label(map_image))
                        
                except Exception as e:
                    print(f"Error updating map display: {str(e)}")
            
            # Update every few seconds
            time.sleep(5)
    
    def update_map_label(self, map_image):
        """Update the map label with new image (thread-safe)"""
        if hasattr(self, 'map_display_label'):
            self.map_display_label.configure(image=map_image)
            self.map_display_label.image = map_image  # Keep a reference

    def toggle_heatmap(self):
        """Toggle the heatmap display on/off"""
        if hasattr(self, 'geo_map_visualizer'):
            # Force map update
            self.refresh_map()
            
            # Update map info
            if self.heatmap_var.get():
                self.update_map_info("Heatmap visualization enabled.")
            else:
                self.update_map_info("Heatmap visualization disabled.")

    def add_mock_map_data(self):
        """Add more mock data to the map"""
        if hasattr(self, 'geo_map_visualizer'):
            self.geo_map_visualizer.generate_mock_data(num_points=30, spread_km=10)
            self.refresh_map()
            self.update_map_info("Added more mock data to the map.")

if __name__ == "__main__":
    app = BorderSurveillanceSystem()
    app.root.mainloop()
