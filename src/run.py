#!/usr/bin/env python3
"""
Border Surveillance System - Runner

Starts both the surveillance system and dashboard API in parallel
"""

import os
import sys
import time
import argparse
import threading
import multiprocessing
import signal
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.main import SurveillanceSystem
from src.dashboard_api import run_server as run_dashboard_api
from utils.logger import logger

def run_surveillance_system(video_source, output_path, show_display):
    """Run the surveillance system in a separate process"""
    try:
        system = SurveillanceSystem(
            video_source=video_source,
            output_path=output_path,
            show_display=show_display
        )
        system.run()
    except KeyboardInterrupt:
        logger.info("Surveillance system interrupted")
    except Exception as e:
        logger.error(f"Error in surveillance system: {str(e)}")

def run_dashboard(host, port):
    """Run the dashboard API in a separate process"""
    try:
        run_dashboard_api(host=host, port=port)
    except KeyboardInterrupt:
        logger.info("Dashboard API interrupted")
    except Exception as e:
        logger.error(f"Error in dashboard API: {str(e)}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Border Surveillance System")
    
    parser.add_argument(
        "-s", "--source",
        help="Video source (camera index or file path), can also be toggled in the UI",
        default=settings.VIDEO_SOURCE
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output video file or directory",
        default=None
    )
    
    parser.add_argument(
        "--no-display",
        help="Disable the video display window",
        action="store_true"
    )
    
    parser.add_argument(
        "--dashboard-host",
        help="Dashboard API host",
        default="0.0.0.0"
    )
    
    parser.add_argument(
        "--dashboard-port",
        help="Dashboard API port",
        default=8000,
        type=int
    )
    
    parser.add_argument(
        "--dashboard-only",
        help="Run only the dashboard API",
        action="store_true"
    )
    
    parser.add_argument(
        "--surveillance-only",
        help="Run only the surveillance system",
        action="store_true"
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    processes = []
    
    # Start surveillance system if requested
    if not args.dashboard_only:
        # Add reminder about video toggling feature
        if not args.no_display:
            logger.info("UI Controls: Click the button or press 's' key to toggle between webcam and video file input")
        
        surveillance_process = multiprocessing.Process(
            target=run_surveillance_system,
            args=(args.source, args.output, not args.no_display)
        )
        surveillance_process.start()
        processes.append(surveillance_process)
        logger.info(f"Surveillance system started (PID: {surveillance_process.pid})")
    
    # Start dashboard API if requested
    if not args.surveillance_only:
        dashboard_process = multiprocessing.Process(
            target=run_dashboard,
            args=(args.dashboard_host, args.dashboard_port)
        )
        dashboard_process.start()
        processes.append(dashboard_process)
        logger.info(f"Dashboard API started at http://{args.dashboard_host}:{args.dashboard_port} (PID: {dashboard_process.pid})")
    
    # Set up signal handling for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received, terminating processes...")
        for process in processes:
            if process.is_alive():
                process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for processes to complete
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down...")
    finally:
        # Ensure all processes are terminated
        for process in processes:
            if process.is_alive():
                process.terminate()

if __name__ == "__main__":
    main() 