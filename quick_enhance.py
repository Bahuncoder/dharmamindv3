#!/usr/bin/env python3
"""
DharmaMind Quick Enhancement Script
==================================
Immediate improvements for better reliability
"""

import subprocess
import sys
import os
from pathlib import Path

class QuickEnhancer:
    def __init__(self):
        self.project_root = Path.cwd()
    
    def print_step(self, step: str, status: str = "INFO"):
        """Print step with status"""
        icons = {"INFO": "INFO", "SUCCESS": "SUCCESS", "ERROR": "ERROR", "WARNING": "WARNING"}
        print(f"[{icons.get(status, 'INFO')}] {step}")
    
    def run_command(self, command: str, cwd: str = None) -> bool:
        """Run shell command and return success status"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                cwd=cwd or self.project_root,
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                self.print_step(f"Command succeeded: {command}", "SUCCESS")
                return True
            else:
                self.print_step(f"Command failed: {command}", "ERROR")
                print(f"   Error: {result.stderr}")
                return False
        except Exception as e:
            self.print_step(f"Command error: {command} - {str(e)}", "ERROR")
            return False
    
    def create_simple_startup_script(self):
        """Create simple startup script"""
        self.print_step("Creating startup scripts...")
        
        # Simple Python startup script
        startup_content = '''#!/usr/bin/env python3
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
'''
        
        # Write startup script
        with open("start_backend_simple.py", 'w', encoding='utf-8') as f:
            f.write(startup_content)
        
        # Create batch file for Windows
        batch_content = '''@echo off
echo Starting DharmaMind Backend...
python start_backend_simple.py
pause
'''
        
        with open("start_backend.bat", 'w', encoding='utf-8') as f:
            f.write(batch_content)
        
        self.print_step("Startup scripts created", "SUCCESS")
    
    def create_requirements_fixer(self):
        """Create requirements installation script"""
        self.print_step("Creating requirements fixer...")
        
        fixer_content = '''#!/usr/bin/env python3
"""
Install essential packages for DharmaMind
"""
import subprocess
import sys

packages = [
    "fastapi",
    "uvicorn[standard]", 
    "redis",
    "fakeredis",
    "asyncpg",
    "sqlalchemy",
    "psutil"
]

def install_packages():
    """Install all required packages"""
    print("Installing DharmaMind requirements...")
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"SUCCESS: {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install {package}")
    
    print("Installation complete!")

if __name__ == "__main__":
    install_packages()
'''
        
        with open("install_requirements.py", 'w', encoding='utf-8') as f:
            f.write(fixer_content)
        
        self.print_step("Requirements fixer created", "SUCCESS")
    
    def create_simple_monitor(self):
        """Create simple monitoring page"""
        self.print_step("Creating monitoring dashboard...")
        
        monitor_content = '''<!DOCTYPE html>
<html>
<head>
    <title>DharmaMind Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .online { background-color: #d4edda; color: #155724; }
        .offline { background-color: #f8d7da; color: #721c24; }
        .service { margin: 10px 0; padding: 15px; border: 1px solid #ddd; }
        button { padding: 10px 20px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>DharmaMind System Monitor</h1>
    
    <button onclick="checkAll()">Refresh All</button>
    
    <div class="service">
        <h3>Backend API (Port 8000)</h3>
        <div id="backend-status" class="status offline">Checking...</div>
        <a href="http://localhost:8000/docs" target="_blank">API Docs</a>
    </div>
    
    <div class="service">
        <h3>Chat App (Port 3001)</h3>
        <div id="chat-status" class="status offline">Checking...</div>
        <a href="http://localhost:3001" target="_blank">Open Chat</a>
    </div>
    
    <div class="service">
        <h3>Brand Website (Port 3000)</h3>
        <div id="brand-status" class="status offline">Checking...</div>
        <a href="http://localhost:3000" target="_blank">Open Website</a>
    </div>
    
    <script>
        function checkService(url, elementId) {
            const element = document.getElementById(elementId);
            element.textContent = 'Checking...';
            element.className = 'status offline';
            
            fetch(url, {mode: 'no-cors'})
                .then(() => {
                    element.textContent = 'Online';
                    element.className = 'status online';
                })
                .catch(() => {
                    element.textContent = 'Offline';
                    element.className = 'status offline';
                });
        }
        
        function checkAll() {
            checkService('http://localhost:8000/health', 'backend-status');
            checkService('http://localhost:3001', 'chat-status');
            checkService('http://localhost:3000', 'brand-status');
        }
        
        // Check on load
        checkAll();
        
        // Auto refresh every 30 seconds
        setInterval(checkAll, 30000);
    </script>
</body>
</html>'''
        
        with open("monitor.html", 'w', encoding='utf-8') as f:
            f.write(monitor_content)
        
        self.print_step("Monitor dashboard created", "SUCCESS")
    
    def run_quick_enhancements(self):
        """Run quick enhancements"""
        print("=" * 50)
        print("DHARMAMIND QUICK ENHANCEMENTS")
        print("=" * 50)
        
        self.create_simple_startup_script()
        self.create_requirements_fixer()
        self.create_simple_monitor()
        
        print("\n" + "=" * 50)
        print("ENHANCEMENT COMPLETE!")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Run: python install_requirements.py")
        print("2. Run: python start_backend_simple.py")
        print("3. Open: monitor.html in browser")
        print("4. Test: python integration_test.py")

if __name__ == "__main__":
    enhancer = QuickEnhancer()
    enhancer.run_quick_enhancements()
