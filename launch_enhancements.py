#!/usr/bin/env python3
"""
🚀 DharmaMind Quick Enhancement Launcher
Launch all three major enhancements simultaneously
"""

import subprocess
import time
import signal
import sys
import threading
from typing import List
import socket

class DharmaMindEnhancementLauncher:
    """Launch and manage DharmaMind enhancement services"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.services = [
            {
                "name": "📊 Analytics Dashboard",
                "script": "advanced_analytics_dashboard.py",
                "port": 8080,
                "url": "http://localhost:8080"
            },
            {
                "name": "🎮 Spiritual Gamification",
                "script": "spiritual_gamification.py",
                "port": 8081,
                "url": "http://localhost:8081"
            },
            {
                "name": "🎭 AI Personality Engine",
                "script": "ai_personality_engine.py",
                "port": 8082,
                "url": "http://localhost:8082"
            }
        ]
    
    def check_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result != 0  # Port is available if connection failed
        except:
            return True
    
    def launch_service(self, service: dict) -> subprocess.Popen:
        """Launch a single service"""
        try:
            print(f"🚀 Starting {service['name']} on port {service['port']}...")
            
            if not self.check_port_available(service['port']):
                print(f"⚠️  Port {service['port']} is already in use!")
                return None
            
            process = subprocess.Popen([
                sys.executable, service['script']
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            time.sleep(2)  # Give service time to start
            
            if process.poll() is None:  # Process is running
                print(f"✅ {service['name']} started successfully!")
                print(f"   URL: {service['url']}")
                return process
            else:
                print(f"❌ Failed to start {service['name']}")
                return None
                
        except Exception as e:
            print(f"❌ Error starting {service['name']}: {e}")
            return None
    
    def launch_all_services(self):
        """Launch all enhancement services"""
        print("🕉️ ========================================")
        print("🚀 DHARMAMIND ENHANCEMENT LAUNCHER")
        print("🕉️ ========================================")
        print()
        
        for service in self.services:
            process = self.launch_service(service)
            if process:
                self.processes.append(process)
        
        print()
        print("🌟 ENHANCEMENT SERVICES STATUS:")
        print("=" * 50)
        
        for i, service in enumerate(self.services):
            if i < len(self.processes) and self.processes[i]:
                print(f"✅ {service['name']}: {service['url']}")
            else:
                print(f"❌ {service['name']}: Failed to start")
        
        print()
        print("🎯 QUICK ACCESS URLS:")
        print("=" * 50)
        print("📊 Analytics Dashboard:    http://localhost:8080")
        print("   - Personal Journey:     http://localhost:8080/user/your_user_id")
        print("   - Community Stats:      http://localhost:8080/api/community")
        print()
        print("🎮 Spiritual Gamification: http://localhost:8081")
        print("   - User Profile:         http://localhost:8081/profile/your_user_id")
        print("   - Achievements:         http://localhost:8081/achievements")
        print("   - Leaderboard:          http://localhost:8081/leaderboard")
        print()
        print("🎭 AI Personality Engine:  http://localhost:8082")
        print("   - Emotion Analysis:     http://localhost:8082/analyze-emotion")
        print("   - Personality Types:    http://localhost:8082/personality-templates")
        print()
        print("🔧 Integration Examples:")
        print("=" * 50)
        print("# Track meditation session")
        print("curl -X POST 'http://localhost:8081/track/user123' \\")
        print("  -H 'Content-Type: application/json' \\")
        print("  -d '{\"activity_type\": \"meditation_session\", \"activity_data\": {\"duration_minutes\": 20, \"quality_score\": 85}}'")
        print()
        print("# Analyze user emotion")
        print("curl -X POST 'http://localhost:8082/analyze-emotion' \\")
        print("  -H 'Content-Type: application/json' \\")
        print("  -d '{\"user_input\": \"I am feeling anxious about my future\"}'")
        print()
        
        if self.processes:
            print("⭐ All services are ready for integration with your DharmaMind backend!")
            print("💫 Check the ENHANCEMENT_ROADMAP.md for detailed implementation guides")
            print()
            print("Press Ctrl+C to stop all services...")
            
            # Wait for interrupt
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop_all_services()
        else:
            print("❌ No services started successfully. Check for errors above.")
    
    def stop_all_services(self):
        """Stop all running services"""
        print("\n🛑 Stopping all enhancement services...")
        
        for i, process in enumerate(self.processes):
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    print(f"✅ Stopped {self.services[i]['name']}")
                except subprocess.TimeoutExpired:
                    process.kill()
                    print(f"🔥 Force stopped {self.services[i]['name']}")
                except Exception as e:
                    print(f"⚠️  Error stopping {self.services[i]['name']}: {e}")
        
        print("🙏 All enhancement services stopped gracefully.")
        print("May your spiritual AI continue to serve all beings with wisdom and compassion!")

def main():
    """Main launcher function"""
    launcher = DharmaMindEnhancementLauncher()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        launcher.stop_all_services()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Launch all services
    launcher.launch_all_services()

if __name__ == "__main__":
    main()
