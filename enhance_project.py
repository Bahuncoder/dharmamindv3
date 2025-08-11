#!/usr/bin/env python3
"""
🚀 DharmaMind Enhancement Implementation Script
===============================================

This script implements immediate improvements and enhancements
to the DharmaMind project for better reliability and performance.
"""

import asyncio
import subprocess
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

class DharmaMindEnhancer:
    def __init__(self):
        self.project_root = Path.cwd()
        self.enhancements = []
        self.status = {
            "backend_stability": False,
            "frontend_dependencies": False,
            "database_setup": False,
            "development_scripts": False,
            "error_handling": False,
            "performance_monitoring": False
        }
    
    def print_header(self, title: str):
        """Print formatted header"""
        print(f"\n{'='*60}")
        print(f"🚀 {title:^54} 🚀")
        print(f"{'='*60}")
    
    def print_step(self, step: str, status: str = "INFO"):
        """Print step with status"""
        icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
        print(f"{icons.get(status, 'ℹ️')} {step}")
    
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
    
    def enhance_backend_stability(self):
        """Enhance backend stability and error handling"""
        self.print_header("BACKEND STABILITY ENHANCEMENT")
        
        # Create enhanced startup script
        startup_script = '''#!/usr/bin/env python3
"""
Enhanced DharmaMind Backend Startup with Auto-Recovery
"""
import asyncio
import subprocess
import time
import sys
import signal
import logging
from pathlib import Path

class BackendManager:
    def __init__(self):
        self.process = None
        self.restart_count = 0
        self.max_restarts = 5
        self.running = True
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('backend_manager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Received shutdown signal")
        self.running = False
        if self.process:
            self.process.terminate()
        sys.exit(0)
    
    def start_backend(self):
        """Start the backend server"""
        try:
            cmd = [
                sys.executable, "-m", "uvicorn", 
                "backend.app.main:app", 
                "--host", "0.0.0.0", 
                "--port", "8000",
                "--reload"
            ]
            
            self.logger.info(f"Starting backend: {' '.join(cmd)}")
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to start backend: {e}")
            return False
    
    def monitor_backend(self):
        """Monitor backend health and restart if needed"""
        while self.running and self.restart_count < self.max_restarts:
            if not self.process or self.process.poll() is not None:
                self.logger.warning("Backend process died, restarting...")
                self.restart_count += 1
                
                if self.restart_count >= self.max_restarts:
                    self.logger.error("Max restarts reached, giving up")
                    break
                
                time.sleep(5)  # Wait before restart
                if self.start_backend():
                    self.logger.info(f"Backend restarted (attempt {self.restart_count})")
                else:
                    self.logger.error("Failed to restart backend")
                    break
            
            # Monitor output
            if self.process and self.process.stdout:
                try:
                    line = self.process.stdout.readline()
                    if line:
                        print(line.strip())
                        
                        # Check for successful startup
                        if "Application startup complete" in line:
                            self.logger.info("Backend startup confirmed")
                            self.restart_count = 0  # Reset on successful start
                        
                        # Check for errors
                        if "ERROR" in line and "startup failed" in line.lower():
                            self.logger.error("Backend startup failed")
                            self.process.terminate()
                except Exception as e:
                    self.logger.error(f"Error reading output: {e}")
            
            time.sleep(1)
    
    def run(self):
        """Main run method"""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("🕉️ Starting DharmaMind Backend Manager")
        
        if self.start_backend():
            self.monitor_backend()
        else:
            self.logger.error("Failed to start backend")
            sys.exit(1)

if __name__ == "__main__":
    manager = BackendManager()
    manager.run()
'''
        
        # Write the enhanced startup script
        script_path = self.project_root / "start_backend_enhanced.py"
        with open(script_path, 'w') as f:
            f.write(startup_script)
        
        self.print_step("Enhanced backend startup script created", "SUCCESS")
        
        # Create backend health check endpoint enhancement
        health_check_enhancement = '''
# Add to backend/app/main.py

@app.get("/health/detailed", tags=["monitoring"])
async def detailed_health_check():
    """Detailed health check with component status"""
    try:
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "metrics": {},
            "version": "2.0.0"
        }
        
        # Check database
        if db_manager:
            db_healthy = await db_manager.health_check()
            health_data["components"]["database"] = {
                "status": "healthy" if db_healthy else "unhealthy",
                "type": "PostgreSQL/SQLite"
            }
        
        # Check Redis
        if redis_client:
            try:
                await redis_client.ping()
                health_data["components"]["redis"] = {"status": "healthy"}
            except Exception:
                health_data["components"]["redis"] = {"status": "unhealthy"}
        
        # Check Chakra modules
        active_modules = len([m for m in chakra_modules.values() if m])
        health_data["components"]["chakra_modules"] = {
            "status": "healthy" if active_modules > 30 else "degraded",
            "active_count": active_modules,
            "total_count": len(chakra_modules)
        }
        
        # System metrics
        import psutil
        health_data["metrics"] = {
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "disk_percent": psutil.disk_usage('/').percent
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/health/ready", tags=["monitoring"])
async def readiness_check():
    """Kubernetes readiness probe endpoint"""
    try:
        # Check critical components
        critical_checks = []
        
        # Database check
        if db_manager:
            critical_checks.append(await db_manager.health_check())
        
        # Chakra modules check
        critical_checks.append(len(chakra_modules) > 30)
        
        if all(critical_checks):
            return {"status": "ready", "timestamp": datetime.now().isoformat()}
        else:
            raise HTTPException(status_code=503, detail="Not ready")
            
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Readiness check failed: {str(e)}")

@app.get("/health/live", tags=["monitoring"])
async def liveness_check():
    """Kubernetes liveness probe endpoint"""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}
'''
        
        # Write health check enhancements
        health_file = self.project_root / "backend_health_enhancements.py"
        with open(health_file, 'w') as f:
            f.write(health_check_enhancement)
        
        self.print_step("Health check enhancements created", "SUCCESS")
        self.status["backend_stability"] = True
    
    def setup_frontend_dependencies(self):
        """Setup and install frontend dependencies"""
        self.print_header("FRONTEND DEPENDENCIES SETUP")
        
        frontend_dirs = ["dharmamind-chat", "Brand_Webpage", "DhramaMind_Community"]
        
        for frontend in frontend_dirs:
            frontend_path = self.project_root / frontend
            if frontend_path.exists():
                self.print_step(f"Installing dependencies for {frontend}")
                
                # Check if package.json exists
                package_json = frontend_path / "package.json"
                if package_json.exists():
                    # Install dependencies
                    success = self.run_command("npm install", cwd=frontend_path)
                    if success:
                        self.print_step(f"{frontend} dependencies installed", "SUCCESS")
                    else:
                        self.print_step(f"{frontend} dependency installation failed", "ERROR")
                else:
                    self.print_step(f"{frontend} missing package.json", "WARNING")
            else:
                self.print_step(f"{frontend} directory not found", "WARNING")
        
        self.status["frontend_dependencies"] = True
    
    def create_development_scripts(self):
        """Create development and deployment scripts"""
        self.print_header("DEVELOPMENT SCRIPTS CREATION")
        
        # Create comprehensive startup script
        startup_script = '''#!/bin/bash
# DharmaMind Complete System Startup

echo "🕉️ Starting DharmaMind Complete System..."

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
PURPLE='\\033[0;35m'
NC='\\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if virtual environment exists
if [ ! -d ".venv_clean" ]; then
    print_error "Virtual environment not found. Creating one..."
    python -m venv .venv_clean
    source .venv_clean/Scripts/activate
    pip install fastapi uvicorn redis fakeredis asyncpg sqlalchemy
else
    print_success "Virtual environment found"
    source .venv_clean/Scripts/activate
fi

# Start backend in background
print_status "Starting backend server..."
python start_backend_enhanced.py &
BACKEND_PID=$!
echo $BACKEND_PID > backend.pid

# Wait for backend to start
sleep 10

# Check if backend is running
if kill -0 $BACKEND_PID 2>/dev/null; then
    print_success "Backend started successfully (PID: $BACKEND_PID)"
else
    print_error "Backend failed to start"
    exit 1
fi

# Start frontends
print_status "Starting frontend applications..."

# Start Chat App
if [ -d "dharmamind-chat" ]; then
    cd dharmamind-chat
    npm run dev &
    CHAT_PID=$!
    echo $CHAT_PID > ../chat.pid
    print_success "Chat app started (PID: $CHAT_PID)"
    cd ..
fi

# Start Brand Website
if [ -d "Brand_Webpage" ]; then
    cd Brand_Webpage
    npm run dev &
    BRAND_PID=$!
    echo $BRAND_PID > ../brand.pid
    print_success "Brand website started (PID: $BRAND_PID)"
    cd ..
fi

# Start Community App
if [ -d "DhramaMind_Community" ]; then
    cd DhramaMind_Community
    npm run dev &
    COMMUNITY_PID=$!
    echo $COMMUNITY_PID > ../community.pid
    print_success "Community app started (PID: $COMMUNITY_PID)"
    cd ..
fi

print_success "🕉️ DharmaMind Complete System Started!"
print_status "Backend API: http://localhost:8000"
print_status "API Docs: http://localhost:8000/docs"
print_status "Chat App: http://localhost:3001"
print_status "Brand Website: http://localhost:3000"
print_status "Community: http://localhost:3003"

print_status "Press Ctrl+C to stop all services"

# Wait for user interrupt
trap 'echo "Shutting down..."; kill $BACKEND_PID $CHAT_PID $BRAND_PID $COMMUNITY_PID 2>/dev/null; exit' INT
wait
'''
        
        # Write startup script
        startup_file = self.project_root / "start_all_services.sh"
        with open(startup_file, 'w') as f:
            f.write(startup_script)
        
        # Make executable
        os.chmod(startup_file, 0o755)
        
        # Create Windows version
        windows_script = '''@echo off
echo 🕉️ Starting DharmaMind Complete System...

REM Activate virtual environment
if not exist ".venv_clean" (
    echo Creating virtual environment...
    python -m venv .venv_clean
    call .venv_clean\\Scripts\\activate.bat
    pip install fastapi uvicorn redis fakeredis asyncpg sqlalchemy
) else (
    call .venv_clean\\Scripts\\activate.bat
)

REM Start backend
echo Starting backend server...
start "DharmaMind Backend" python start_backend_enhanced.py

REM Wait for backend
timeout /t 10 /nobreak

REM Start frontends
if exist "dharmamind-chat" (
    cd dharmamind-chat
    start "Chat App" npm run dev
    cd ..
)

if exist "Brand_Webpage" (
    cd Brand_Webpage
    start "Brand Website" npm run dev
    cd ..
)

if exist "DhramaMind_Community" (
    cd DhramaMind_Community
    start "Community App" npm run dev
    cd ..
)

echo.
echo 🕉️ DharmaMind Complete System Started!
echo Backend API: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Chat App: http://localhost:3001
echo Brand Website: http://localhost:3000
echo Community: http://localhost:3003
echo.
echo Press any key to open API docs...
pause
start http://localhost:8000/docs
'''
        
        windows_file = self.project_root / "start_all_services.bat"
        with open(windows_file, 'w') as f:
            f.write(windows_script)
        
        self.print_step("Development scripts created", "SUCCESS")
        self.status["development_scripts"] = True
    
    def create_monitoring_dashboard(self):
        """Create simple monitoring dashboard"""
        self.print_header("MONITORING DASHBOARD CREATION")
        
        dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🕉️ DharmaMind System Monitor</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .status-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 10px;
        }
        .status-online { background-color: #4ade80; }
        .status-offline { background-color: #f87171; }
        .status-warning { background-color: #fbbf24; }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .metric {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #4ade80;
        }
        .refresh-btn {
            background: #4ade80;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .refresh-btn:hover {
            background: #22c55e;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🕉️ DharmaMind System Monitor</h1>
            <button class="refresh-btn" onclick="refreshAll()">🔄 Refresh All</button>
        </div>
        
        <div class="status-grid">
            <div class="status-card">
                <h3>🔱 Backend API</h3>
                <div id="backend-status">
                    <span class="status-indicator status-offline"></span>
                    <span>Checking...</span>
                </div>
                <p>Port: 8000</p>
                <a href="http://localhost:8000/docs" target="_blank" style="color: #4ade80;">API Docs</a>
            </div>
            
            <div class="status-card">
                <h3>💬 Chat Application</h3>
                <div id="chat-status">
                    <span class="status-indicator status-offline"></span>
                    <span>Checking...</span>
                </div>
                <p>Port: 3001</p>
                <a href="http://localhost:3001" target="_blank" style="color: #4ade80;">Open App</a>
            </div>
            
            <div class="status-card">
                <h3>🌐 Brand Website</h3>
                <div id="brand-status">
                    <span class="status-indicator status-offline"></span>
                    <span>Checking...</span>
                </div>
                <p>Port: 3000</p>
                <a href="http://localhost:3000" target="_blank" style="color: #4ade80;">Open Site</a>
            </div>
            
            <div class="status-card">
                <h3>👥 Community App</h3>
                <div id="community-status">
                    <span class="status-indicator status-offline"></span>
                    <span>Checking...</span>
                </div>
                <p>Port: 3003</p>
                <a href="http://localhost:3003" target="_blank" style="color: #4ade80;">Open Community</a>
            </div>
        </div>
        
        <div class="status-card">
            <h3>📊 System Metrics</h3>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value" id="chakra-modules">--</div>
                    <div>Chakra Modules</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="uptime">--</div>
                    <div>Uptime</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="response-time">--</div>
                    <div>Response Time</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="requests">--</div>
                    <div>Total Requests</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function checkService(url, statusElementId) {
            const statusElement = document.getElementById(statusElementId);
            const indicator = statusElement.querySelector('.status-indicator');
            const text = statusElement.querySelector('span:last-child');
            
            try {
                const startTime = Date.now();
                const response = await fetch(url, { mode: 'no-cors' });
                const endTime = Date.now();
                
                indicator.className = 'status-indicator status-online';
                text.textContent = 'Online';
                
                if (statusElementId === 'backend-status') {
                    document.getElementById('response-time').textContent = `${endTime - startTime}ms`;
                }
            } catch (error) {
                indicator.className = 'status-indicator status-offline';
                text.textContent = 'Offline';
            }
        }
        
        async function fetchBackendMetrics() {
            try {
                const response = await fetch('http://localhost:8000/health/detailed');
                const data = await response.json();
                
                if (data.components && data.components.chakra_modules) {
                    document.getElementById('chakra-modules').textContent = 
                        data.components.chakra_modules.active_count || '--';
                }
                
                if (data.metrics) {
                    // Could display memory, CPU usage, etc.
                }
            } catch (error) {
                console.log('Could not fetch backend metrics');
            }
        }
        
        function refreshAll() {
            checkService('http://localhost:8000/health', 'backend-status');
            checkService('http://localhost:3001', 'chat-status');
            checkService('http://localhost:3000', 'brand-status');
            checkService('http://localhost:3003', 'community-status');
            fetchBackendMetrics();
        }
        
        // Initial load
        refreshAll();
        
        // Auto-refresh every 30 seconds
        setInterval(refreshAll, 30000);
    </script>
</body>
</html>'''
        
        dashboard_file = self.project_root / "monitor.html"
        with open(dashboard_file, 'w') as f:
            f.write(dashboard_html)
        
        self.print_step("Monitoring dashboard created at monitor.html", "SUCCESS")
        self.status["performance_monitoring"] = True
    
    def create_quick_fixes(self):
        """Create quick fixes for common issues"""
        self.print_header("QUICK FIXES IMPLEMENTATION")
        
        # Create requirements fixer
        requirements_fixer = '''#!/usr/bin/env python3
"""
Quick Requirements Installer for DharmaMind
"""
import subprocess
import sys

def install_package(package):
    """Install a package with pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    print("🔧 Installing essential DharmaMind packages...")
    
    essential_packages = [
        "fastapi",
        "uvicorn[standard]",
        "redis",
        "fakeredis",
        "asyncpg",
        "sqlalchemy",
        "pydantic",
        "python-dotenv",
        "aiohttp",
        "psutil"
    ]
    
    success_count = 0
    for package in essential_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\\n📊 Installed {success_count}/{len(essential_packages)} packages")
    
    if success_count == len(essential_packages):
        print("✅ All packages installed successfully!")
        return True
    else:
        print("⚠️ Some packages failed to install")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
        
        fixer_file = self.project_root / "fix_requirements.py"
        with open(fixer_file, 'w') as f:
            f.write(requirements_fixer)
        
        self.print_step("Requirements fixer created", "SUCCESS")
    
    def run_enhancements(self):
        """Run all enhancements"""
        self.print_header("DHARMAMIND PROJECT ENHANCEMENT")
        
        try:
            # Run all enhancement steps
            self.enhance_backend_stability()
            self.setup_frontend_dependencies()
            self.create_development_scripts()
            self.create_monitoring_dashboard()
            self.create_quick_fixes()
            
            # Summary
            self.print_header("ENHANCEMENT SUMMARY")
            
            for component, status in self.status.items():
                status_text = "✅ COMPLETED" if status else "❌ FAILED"
                self.print_step(f"{component.replace('_', ' ').title()}: {status_text}")
            
            print(f"\n🎯 Enhancement Score: {sum(self.status.values())}/{len(self.status)} components")
            
            if all(self.status.values()):
                self.print_step("🎉 ALL ENHANCEMENTS COMPLETED SUCCESSFULLY!", "SUCCESS")
                print(f"\n📋 Next Steps:")
                print(f"   1. Run: python fix_requirements.py")
                print(f"   2. Run: ./start_all_services.bat (Windows) or ./start_all_services.sh (Linux)")
                print(f"   3. Open: monitor.html in your browser")
                print(f"   4. Test: python integration_test.py")
            else:
                self.print_step("⚠️ Some enhancements failed. Check logs above.", "WARNING")
            
        except Exception as e:
            self.print_step(f"Enhancement failed: {str(e)}", "ERROR")
            raise

if __name__ == "__main__":
    enhancer = DharmaMindEnhancer()
    enhancer.run_enhancements()
