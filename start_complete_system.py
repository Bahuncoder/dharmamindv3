#!/usr/bin/env python3
"""
DharmaMind Complete System Startup Script

This script initializes and starts the entire DharmaMind ecosystem:
- Backend API with all Chakra modules
- Vector database setup
- DharmaLLM initialization
- Frontend development server
- System health checks and validation

🕉️ Universal Wisdom Platform - Complete Integration
"""

import os
import sys
import time
import subprocess
import asyncio
import logging
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DharmaMindSystemManager:
    """Complete system manager for DharmaMind platform"""
    
    def __init__(self, mode: str = "development"):
        self.mode = mode
        self.base_path = Path(__file__).parent
        self.processes: Dict[str, subprocess.Popen] = {}
        self.services = {
            "backend": {"port": 8000, "health_endpoint": "/health"},
            "frontend": {"port": 3000, "health_endpoint": "/"},
            "vector_db": {"port": 6333, "health_endpoint": "/"},
        }
        
    def print_banner(self):
        """Print startup banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║          🕉️  DharmaMind - Universal Wisdom Platform           ║
║                                                               ║
║     AI with Soul powered by Dharma - Complete System         ║
║                                                               ║
║  🧠 Advanced Consciousness  🔮 Spiritual Wisdom              ║
║  🤖 AI Intelligence         ❤️ Emotional Guidance            ║
║  🔒 Protection & Safety     🌟 System Harmony               ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        logger.info(f"🚀 Starting DharmaMind Complete System in {self.mode} mode...")
        
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are installed"""
        logger.info("🔍 Checking system prerequisites...")
        
        requirements = {
            "python": {"cmd": ["python3", "--version"], "min_version": "3.9"},
            "node": {"cmd": ["node", "--version"], "min_version": "16"},
            "npm": {"cmd": ["npm", "--version"], "min_version": "8"},
            "pip": {"cmd": ["pip3", "--version"], "min_version": "21"}
        }
        
        for name, req in requirements.items():
            try:
                result = subprocess.run(
                    req["cmd"], 
                    capture_output=True, 
                    text=True, 
                    check=True, 
                    shell=True
                )
                version = result.stdout.strip()
                logger.info(f"✅ {name}: {version}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.error(f"❌ {name} not found or version check failed")
                return False
                
        return True
        
    def setup_environment(self):
        """Setup environment variables and configuration"""
        logger.info("⚙️ Setting up environment...")
        
        # Set environment variables
        env_vars = {
            "DHARMAMIND_MODE": self.mode,
            "PYTHONPATH": str(self.base_path),
            "NEXT_PUBLIC_API_URL": "http://localhost:8000",
            "BACKEND_URL": "http://localhost:8000",
            "CORS_ORIGINS": '["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:3003"]',
            "ALLOWED_HOSTS": '["localhost", "127.0.0.1", "0.0.0.0"]'
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.info(f"🔧 Set {key}={value}")
            
    def install_dependencies(self):
        """Install Python and Node.js dependencies"""
        logger.info("📦 Installing dependencies...")
        
        # Install Python dependencies
        logger.info("📦 Installing Python dependencies...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", 
                str(self.base_path / "requirements.txt")
            ], check=True, cwd=self.base_path)
            logger.info("✅ Python dependencies installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install Python dependencies: {e}")
            
        # Install backend dependencies
        backend_requirements = self.base_path / "backend" / "requirements.txt"
        if backend_requirements.exists():
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(backend_requirements)
                ], check=True, cwd=self.base_path)
                logger.info("✅ Backend dependencies installed")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install backend dependencies: {e}")
        
        # Install frontend dependencies
        frontend_path = self.base_path / "frontend"
        if frontend_path.exists():
            logger.info("📦 Installing frontend dependencies...")
            try:
                subprocess.run(["npm", "install"], check=True, cwd=frontend_path)
                logger.info("✅ Frontend dependencies installed")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install frontend dependencies: {e}")
                
    def setup_vector_database(self):
        """Setup vector database"""
        logger.info("🗄️ Setting up vector database...")
        
        vector_setup_script = self.base_path / "vector_db" / "setup_vector_db.py"
        if vector_setup_script.exists():
            try:
                subprocess.run([
                    sys.executable, str(vector_setup_script)
                ], check=True, cwd=self.base_path)
                logger.info("✅ Vector database initialized")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ Vector database setup warning: {e}")
        else:
            logger.warning("⚠️ Vector database setup script not found, skipping...")
            
    def initialize_dharmallm(self):
        """Initialize DharmaLLM system"""
        logger.info("🧠 Initializing DharmaLLM...")
        
        dharmallm_path = self.base_path / "dharmallm"
        if dharmallm_path.exists():
            try:
                # Check if models exist or need training
                models_path = dharmallm_path / "models"
                if not models_path.exists() or not list(models_path.glob("*.pt")):
                    logger.info("📚 No trained models found, preparing data...")
                    
                    # Run data preprocessing
                    preprocess_script = dharmallm_path / "data" / "preprocess_data.py"
                    if preprocess_script.exists():
                        subprocess.run([
                            sys.executable, str(preprocess_script)
                        ], check=True, cwd=dharmallm_path)
                        logger.info("✅ Data preprocessing completed")
                else:
                    logger.info("✅ DharmaLLM models found")
                    
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ DharmaLLM initialization warning: {e}")
        else:
            logger.warning("⚠️ DharmaLLM directory not found, using external LLMs only")
            
    def start_backend(self):
        """Start the backend API server"""
        logger.info("⚡ Starting backend API server...")
        
        backend_path = self.base_path / "backend"
        if not backend_path.exists():
            logger.error("❌ Backend directory not found")
            return False
            
        try:
            cmd = [
                sys.executable, "-m", "uvicorn", 
                "app.main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload" if self.mode == "development" else "--no-reload",
                "--log-level", "info"
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd=backend_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            self.processes["backend"] = process
            logger.info("🚀 Backend server starting...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start backend: {e}")
            return False
            
    def start_frontend(self):
        """Start the frontend development servers"""
        logger.info("🎨 Starting frontend development servers...")
        
        # Start brand website on port 3002
        brand_path = self.base_path / "dharmamind-brand"
        if brand_path.exists():
            try:
                # Use npm.cmd for Windows compatibility
                cmd = ["npm.cmd", "run", "dev"]
                process = subprocess.Popen(
                    cmd,
                    cwd=brand_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    shell=True
                )
                self.processes["brand"] = process
                logger.info("🚀 Brand website starting on port 3002...")
            except Exception as e:
                logger.error(f"❌ Failed to start brand website: {e}")
        
        # Start chat application on port 3003  
        chat_path = self.base_path / "dharmamind-chat"
        if chat_path.exists():
            try:
                # Use npm.cmd for Windows compatibility
                cmd = ["npm.cmd", "run", "dev"]
                process = subprocess.Popen(
                    cmd,
                    cwd=chat_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    shell=True
                )
                self.processes["chat"] = process
                logger.info("🚀 Chat application starting on port 3003...")
            except Exception as e:
                logger.error(f"❌ Failed to start chat application: {e}")
        
        if not brand_path.exists() and not chat_path.exists():
            logger.error("❌ No frontend directories found")
            return False
            
        return True
            
    def wait_for_service(self, service_name: str, max_attempts: int = 30) -> bool:
        """Wait for a service to become available"""
        service = self.services.get(service_name)
        if not service:
            return False
            
        url = f"http://localhost:{service['port']}{service['health_endpoint']}"
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ {service_name} service is ready at port {service['port']}")
                    return True
            except requests.exceptions.RequestException:
                pass
                
            logger.info(f"⏳ Waiting for {service_name} service... ({attempt + 1}/{max_attempts})")
            time.sleep(2)
            
        logger.error(f"❌ {service_name} service failed to start")
        return False
        
    def perform_health_checks(self) -> bool:
        """Perform comprehensive system health checks"""
        logger.info("🔍 Performing system health checks...")
        
        # Check backend health
        try:
            response = requests.get("http://localhost:8000/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ Backend health: {health_data.get('status', 'unknown')}")
                
                # Check Chakra modules
                chakra_response = requests.get("http://localhost:8000/chakra/status", timeout=10)
                if chakra_response.status_code == 200:
                    chakra_data = chakra_response.json()
                    active_modules = len([m for m in chakra_data.get('modules', {}).values() 
                                        if m.get('status') == 'active'])
                    total_modules = len(chakra_data.get('modules', {}))
                    logger.info(f"✅ Chakra modules: {active_modules}/{total_modules} active")
                    logger.info(f"🕉️ System harmony: {chakra_data.get('system_harmony', 'unknown')}")
                    logger.info(f"🧘 Consciousness: {chakra_data.get('consciousness_level', 'unknown')}")
                    logger.info(f"⚖️ Dharma alignment: {chakra_data.get('dharma_alignment', 'unknown')}")
                else:
                    logger.warning("⚠️ Could not retrieve Chakra module status")
            else:
                logger.error("❌ Backend health check failed")
                return False
        except Exception as e:
            logger.error(f"❌ Backend health check error: {e}")
            return False
            
        return True
        
    def start_system(self):
        """Start the complete DharmaMind system"""
        self.print_banner()
        
        # Step 1: Prerequisites
        if not self.check_prerequisites():
            logger.error("❌ Prerequisites check failed")
            return False
            
        # Step 2: Environment setup
        self.setup_environment()
        
        # Step 3: Install dependencies
        if self.mode == "development":
            self.install_dependencies()
            
        # Step 4: Initialize components
        self.setup_vector_database()
        self.initialize_dharmallm()
        
        # Step 5: Start services
        if not self.start_backend():
            return False
            
        # Wait for backend to be ready
        if not self.wait_for_service("backend"):
            return False
            
        if not self.start_frontend():
            return False
            
        # Wait for frontend to be ready
        if not self.wait_for_service("frontend"):
            return False
            
        # Step 6: Health checks
        if not self.perform_health_checks():
            logger.warning("⚠️ Some health checks failed, but system is running")
            
        # Success message
        logger.info("=" * 70)
        logger.info("🎉 DharmaMind Complete System Successfully Started!")
        logger.info("=" * 70)
        logger.info(f"🌐 Frontend: http://localhost:3000")
        logger.info(f"⚡ Backend API: http://localhost:8000")
        logger.info(f"📚 API Documentation: http://localhost:8000/docs")
        logger.info(f"🔍 System Health: http://localhost:8000/health")
        logger.info(f"🕉️ Chakra Status: http://localhost:8000/chakra/status")
        logger.info("=" * 70)
        logger.info("🙏 May this serve all beings with wisdom and compassion")
        
        return True
        
    def stop_system(self):
        """Stop all system processes"""
        logger.info("🛑 Stopping DharmaMind system...")
        
        for service_name, process in self.processes.items():
            try:
                logger.info(f"🛑 Stopping {service_name}...")
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"✅ {service_name} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️ Force killing {service_name}...")
                process.kill()
            except Exception as e:
                logger.error(f"❌ Error stopping {service_name}: {e}")
                
        logger.info("🕉️ System shutdown complete")
        
    def monitor_system(self):
        """Monitor system processes"""
        logger.info("👁️ Starting system monitoring...")
        
        try:
            while True:
                time.sleep(30)  # Check every 30 seconds
                
                # Check if processes are still running
                for service_name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        logger.error(f"❌ {service_name} process died unexpectedly")
                        # Could implement auto-restart logic here
                        
                # Perform periodic health checks
                try:
                    response = requests.get("http://localhost:8000/health", timeout=5)
                    if response.status_code != 200:
                        logger.warning(f"⚠️ Backend health check failed: {response.status_code}")
                except requests.exceptions.RequestException:
                    logger.warning("⚠️ Backend not responding to health checks")
                    
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring interrupted by user")
            self.stop_system()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DharmaMind Complete System Manager")
    parser.add_argument(
        "--mode", 
        choices=["development", "production"], 
        default="development",
        help="System mode (default: development)"
    )
    parser.add_argument(
        "--no-monitor", 
        action="store_true",
        help="Don't monitor system after startup"
    )
    
    args = parser.parse_args()
    
    system_manager = DharmaMindSystemManager(mode=args.mode)
    
    try:
        if system_manager.start_system():
            if not args.no_monitor:
                system_manager.monitor_system()
        else:
            logger.error("❌ Failed to start system")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
        system_manager.stop_system()
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        system_manager.stop_system()
        sys.exit(1)

if __name__ == "__main__":
    main()
