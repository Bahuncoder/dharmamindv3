#!/usr/bin/env python3
"""
Simple DharmaMind Backend Startup
"""
import subprocess
import sys
import time

def start_backend():
    """Start the backend server"""
    print("Starting DharmaMind Backend...")
    try:
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "backend.app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ]
        
        process = subprocess.Popen(cmd)
        print(f"Backend started with PID: {process.pid}")
        print("Backend running at: http://localhost:8000")
        print("API Docs at: http://localhost:8000/docs")
        print("Press Ctrl+C to stop")
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("Stopping backend...")
            process.terminate()
            
    except Exception as e:
        print(f"Failed to start backend: {e}")

if __name__ == "__main__":
    start_backend()
