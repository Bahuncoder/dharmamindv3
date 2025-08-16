#!/usr/bin/env python3
"""
🚀 DharmaMind Quick Deployment Launcher
=======================================

Quickly starts all DharmaMind components with minimal configuration:
- Backend API (port 8000)
- Chat Frontend (port 3001) 
- Brand Website (port 3000)
- Community App (port 3003)

🕉️ For rapid testing and development
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path

class DharmaMindLauncher:
    def __init__(self):
        self.processes = []
        self.base_dir = Path(__file__).parent
        
    def start_backend(self):
        """Start the FastAPI backend"""
        print("🔱 Starting DharmaMind Backend...")
        
        backend_cmd = [
            sys.executable, "-m", "uvicorn", 
            "backend.app.main:app", 
            "--reload", "--host", "0.0.0.0", "--port", "8000"
        ]
        
        try:
            process = subprocess.Popen(
                backend_cmd,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            self.processes.append(("Backend", process))
            print("✅ Backend started on http://localhost:8000")
            return True
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self, name, port, directory):
        """Start a Next.js frontend application"""
        frontend_dir = self.base_dir / directory
        
        if not frontend_dir.exists():
            print(f"❌ {name} directory not found: {frontend_dir}")
            return False
            
        print(f"🌐 Starting {name} on port {port}...")
        
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            print(f"📦 Installing dependencies for {name}...")
            install_process = subprocess.run(
                ["npm", "install"],
                cwd=frontend_dir,
                capture_output=True,
                text=True
            )
            if install_process.returncode != 0:
                print(f"❌ Failed to install {name} dependencies")
                return False
        
        try:
            # Set environment variable for port
            env = os.environ.copy()
            env["PORT"] = str(port)
            
            process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=frontend_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            self.processes.append((name, process))
            print(f"✅ {name} started on http://localhost:{port}")
            return True
        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return False
    
    def wait_for_startup(self, seconds=10):
        """Wait for services to start up"""
        print(f"⏳ Waiting {seconds} seconds for services to initialize...")
        for i in range(seconds):
            print(f"   {seconds-i} seconds remaining...", end="\r")
            time.sleep(1)
        print("\n✨ Startup complete!")
    
    def monitor_processes(self):
        """Monitor running processes"""
        print("\n📊 Monitoring all processes...")
        print("Press Ctrl+C to stop all services\n")
        
        try:
            while True:
                for name, process in self.processes:
                    if process.poll() is not None:
                        print(f"❌ {name} has stopped (exit code: {process.returncode})")
                        return False
                
                print("🟢 All services running...", end="\r")
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested...")
            return True
    
    def stop_all(self):
        """Stop all running processes"""
        print("🔴 Stopping all services...")
        
        for name, process in self.processes:
            if process.poll() is None:
                print(f"   Stopping {name}...")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"   Force killing {name}...")
                    process.kill()
                except Exception as e:
                    print(f"   Error stopping {name}: {e}")
        
        print("✅ All services stopped")
    
    def launch_all(self):
        """Launch all DharmaMind services"""
        print("🕉️" + "="*50 + "🕉️")
        print("🔱    DHARMAMIND QUICK DEPLOYMENT LAUNCHER    🔱")
        print("🕉️" + "="*50 + "🕉️")
        
        success_count = 0
        
        # Start backend
        if self.start_backend():
            success_count += 1
        
        # Start frontends
        frontends = [
            ("Brand Website", 3000, "Brand_Webpage"),
            ("Chat App", 3001, "dharmamind-chat"),
            ("Community App", 3003, "DhramaMind_Community")
        ]
        
        for name, port, directory in frontends:
            if self.start_frontend(name, port, directory):
                success_count += 1
        
        if success_count == 0:
            print("❌ No services started successfully")
            return False
        
        # Wait for startup
        self.wait_for_startup(10)
        
        # Show status
        print("\n🌟 DharmaMind Services Status:")
        print("="*40)
        print("🔱 Backend API:     http://localhost:8000")
        print("🌐 Brand Website:   http://localhost:3000") 
        print("💬 Chat App:        http://localhost:3001")
        print("👥 Community App:   http://localhost:3003")
        print("📚 API Docs:        http://localhost:8000/docs")
        print("="*40)
        
        # Monitor processes
        if self.monitor_processes():
            self.stop_all()
        
        return True

def main():
    """Main function"""
    launcher = DharmaMindLauncher()
    
    try:
        success = launcher.launch_all()
        if not success:
            print("\n❌ Deployment failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
        launcher.stop_all()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        launcher.stop_all()
        sys.exit(1)
    
    print("\n🕉️ May this system serve all beings with wisdom and compassion 🕉️")

if __name__ == "__main__":
    main()
