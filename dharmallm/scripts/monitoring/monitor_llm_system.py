#!/usr/bin/env python3
"""
🕉️ DharmaLLM System Monitor Dashboard
======================================

Comprehensive monitoring for the DharmaLLM production system:
- API health and response times
- Request rates and throughput
- Model performance metrics
- Cache statistics
- Database connections
- Memory and resource usage
- Error rates and logs
- Active users and sessions
"""

import os
import sys
import time
import json
import psutil
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque

# Colors
class Color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class LLMSystemMonitor:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.response_times = deque(maxlen=100)
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        self.services_dir = Path("services")
        self.models_dir = Path("model")
        self.cache_dir = Path("engines/cache")
        
    def check_api_health(self) -> Dict:
        """Check API health endpoint"""
        try:
            start = time.time()
            response = requests.get(f"{self.api_url}/health", timeout=5)
            elapsed = time.time() - start
            
            self.response_times.append(elapsed)
            self.request_count += 1
            
            return {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'status_code': response.status_code,
                'response_time': elapsed,
                'data': response.json() if response.ok else None
            }
        except requests.exceptions.ConnectionError:
            self.error_count += 1
            return {'status': 'offline', 'error': 'Connection refused'}
        except Exception as e:
            self.error_count += 1
            return {'status': 'error', 'error': str(e)}
    
    def get_api_stats(self) -> Dict:
        """Get API statistics"""
        try:
            response = requests.get(f"{self.api_url}/api/stats", timeout=5)
            if response.ok:
                return response.json()
        except:
            pass
        
        return {
            'total_requests': self.request_count,
            'error_rate': self.error_count / max(self.request_count, 1) * 100,
            'avg_response_time': sum(self.response_times) / len(self.response_times) if self.response_times else 0,
            'uptime': time.time() - self.start_time
        }
    
    def find_llm_processes(self) -> List[psutil.Process]:
        """Find all DharmaLLM related processes"""
        processes = []
        keywords = ['uvicorn', 'fastapi', 'llm', 'dharma', 'api/main']
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info.get('cmdline', []))
                if any(kw in cmdline.lower() for kw in keywords):
                    processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return processes
    
    def get_model_info(self) -> Dict:
        """Get loaded model information"""
        info = {
            'models_available': [],
            'models_loaded': 0,
            'total_size_gb': 0
        }
        
        if self.models_dir.exists():
            checkpoints = list(self.models_dir.glob("**/*.pt"))
            for ckpt in checkpoints:
                size_gb = ckpt.stat().st_size / 1024**3
                info['models_available'].append({
                    'name': ckpt.name,
                    'path': str(ckpt.relative_to(self.models_dir)),
                    'size_gb': size_gb,
                    'modified': datetime.fromtimestamp(ckpt.stat().st_mtime)
                })
                info['total_size_gb'] += size_gb
        
        return info
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        stats = {
            'cache_enabled': False,
            'cache_size_mb': 0,
            'cache_entries': 0
        }
        
        if self.cache_dir.exists():
            stats['cache_enabled'] = True
            cache_files = list(self.cache_dir.glob("**/*"))
            total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
            stats['cache_size_mb'] = total_size / 1024 / 1024
            stats['cache_entries'] = len([f for f in cache_files if f.is_file()])
        
        return stats
    
    def get_service_status(self) -> Dict:
        """Check status of various services"""
        services = {}
        
        service_files = {
            'llm_router': 'services/llm_router.py',
            'dharma_processor': 'services/dharmic_llm_processor.py',
            'rishi_engine': 'services/enhanced_rishi_engine.py',
            'memory_manager': 'services/memory_manager.py',
            'evaluator': 'services/evaluator.py'
        }
        
        for name, path in service_files.items():
            file_path = Path(path)
            services[name] = {
                'exists': file_path.exists(),
                'size_kb': file_path.stat().st_size / 1024 if file_path.exists() else 0
            }
        
        return services
    
    def get_database_stats(self) -> Dict:
        """Get database statistics (if applicable)"""
        # Placeholder for database monitoring
        return {
            'connected': False,
            'total_users': 0,
            'active_sessions': 0
        }
    
    def draw_sparkline(self, values: List[float], width: int = 40) -> str:
        """Draw ASCII sparkline"""
        if not values:
            return "─" * width
        
        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val if max_val > min_val else 1
        
        chars = "▁▂▃▄▅▆▇█"
        line = []
        
        for val in values[-width:]:
            normalized = (val - min_val) / val_range
            idx = int(normalized * (len(chars) - 1))
            line.append(chars[idx])
        
        return ''.join(line)
    
    def draw_gauge(self, value: float, max_value: float = 100, width: int = 20) -> str:
        """Draw ASCII gauge"""
        if max_value == 0:
            return "[" + "─" * width + "]"
        
        percent = min(value / max_value, 1.0)
        filled = int(width * percent)
        
        if percent < 0.5:
            color = Color.OKGREEN
        elif percent < 0.8:
            color = Color.WARNING
        else:
            color = Color.FAIL
        
        bar = "█" * filled + "░" * (width - filled)
        return f"{color}[{bar}]{Color.ENDC} {percent*100:.1f}%"
    
    def display_dashboard(self):
        """Display the main dashboard"""
        os.system('clear')
        
        # Header
        print(f"\n{Color.HEADER}{'═' * 80}{Color.ENDC}")
        print(f"{Color.HEADER}{Color.BOLD}🕉️  DharmaLLM System Monitor{Color.ENDC}")
        print(f"{Color.HEADER}{'═' * 80}{Color.ENDC}\n")
        print(f"{Color.OKCYAN}⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Color.ENDC}\n")
        
        # API Health
        health = self.check_api_health()
        print(f"{Color.OKGREEN}{'─' * 80}{Color.ENDC}")
        print(f"{Color.OKGREEN}{Color.BOLD}🌐 API HEALTH{Color.ENDC}")
        print(f"{Color.OKGREEN}{'─' * 80}{Color.ENDC}")
        
        if health['status'] == 'healthy':
            print(f"  Status: {Color.OKGREEN}✅ HEALTHY{Color.ENDC}")
            print(f"  Endpoint: {self.api_url}")
            print(f"  Response Time: {health['response_time']*1000:.0f}ms")
        elif health['status'] == 'offline':
            print(f"  Status: {Color.FAIL}🔴 OFFLINE{Color.ENDC}")
            print(f"  Endpoint: {self.api_url}")
            print(f"  Error: Connection refused")
            print(f"\n  {Color.WARNING}Start the API server:{Color.ENDC}")
            print(f"    cd api && uvicorn main:app --reload")
        else:
            print(f"  Status: {Color.WARNING}⚠️  ERROR{Color.ENDC}")
            print(f"  Error: {health.get('error', 'Unknown')}")
        
        # API Statistics
        api_stats = self.get_api_stats()
        print(f"\n{Color.OKBLUE}{'─' * 80}{Color.ENDC}")
        print(f"{Color.OKBLUE}{Color.BOLD}📊 API STATISTICS{Color.ENDC}")
        print(f"{Color.OKBLUE}{'─' * 80}{Color.ENDC}")
        print(f"  Total Requests: {api_stats['total_requests']:,}")
        print(f"  Error Rate: {api_stats['error_rate']:.2f}%")
        print(f"  Avg Response: {api_stats['avg_response_time']*1000:.0f}ms")
        
        if self.response_times:
            print(f"  Response Times: {self.draw_sparkline(list(self.response_times))}")
        
        uptime = str(timedelta(seconds=int(api_stats['uptime'])))
        print(f"  Uptime: {uptime}")
        
        # Running Processes
        processes = self.find_llm_processes()
        print(f"\n{Color.OKCYAN}{'─' * 80}{Color.ENDC}")
        print(f"{Color.OKCYAN}{Color.BOLD}⚙️  RUNNING PROCESSES{Color.ENDC}")
        print(f"{Color.OKCYAN}{'─' * 80}{Color.ENDC}")
        
        if processes:
            for proc in processes:
                try:
                    print(f"  • PID {proc.pid}: {proc.name()} - CPU: {proc.cpu_percent():.1f}% - MEM: {proc.memory_percent():.1f}%")
                except:
                    pass
        else:
            print(f"  {Color.WARNING}No DharmaLLM processes found{Color.ENDC}")
        
        # Model Information
        model_info = self.get_model_info()
        print(f"\n{Color.WARNING}{'─' * 80}{Color.ENDC}")
        print(f"{Color.WARNING}{Color.BOLD}🤖 MODEL INFORMATION{Color.ENDC}")
        print(f"{Color.WARNING}{'─' * 80}{Color.ENDC}")
        print(f"  Available Models: {len(model_info['models_available'])}")
        print(f"  Total Size: {model_info['total_size_gb']:.2f} GB")
        
        if model_info['models_available']:
            latest = sorted(model_info['models_available'], 
                          key=lambda x: x['modified'], reverse=True)[0]
            print(f"  Latest: {latest['name']} ({latest['size_gb']:.2f} GB)")
            print(f"  Modified: {latest['modified'].strftime('%Y-%m-%d %H:%M')}")
        
        # Cache Statistics
        cache_stats = self.get_cache_stats()
        print(f"\n{Color.OKGREEN}{'─' * 80}{Color.ENDC}")
        print(f"{Color.OKGREEN}{Color.BOLD}💾 CACHE STATISTICS{Color.ENDC}")
        print(f"{Color.OKGREEN}{'─' * 80}{Color.ENDC}")
        
        if cache_stats['cache_enabled']:
            print(f"  Status: {Color.OKGREEN}✅ Enabled{Color.ENDC}")
            print(f"  Size: {cache_stats['cache_size_mb']:.1f} MB")
            print(f"  Entries: {cache_stats['cache_entries']:,}")
        else:
            print(f"  Status: {Color.WARNING}⚠️  Disabled{Color.ENDC}")
        
        # Service Status
        services = self.get_service_status()
        print(f"\n{Color.OKBLUE}{'─' * 80}{Color.ENDC}")
        print(f"{Color.OKBLUE}{Color.BOLD}🔧 SERVICE STATUS{Color.ENDC}")
        print(f"{Color.OKBLUE}{'─' * 80}{Color.ENDC}")
        
        for name, status in services.items():
            icon = "✅" if status['exists'] else "❌"
            print(f"  {icon} {name}: {status['size_kb']:.1f} KB" if status['exists'] else f"  {icon} {name}: Not found")
        
        # System Resources
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        print(f"\n{Color.HEADER}{'─' * 80}{Color.ENDC}")
        print(f"{Color.HEADER}{Color.BOLD}💻 SYSTEM RESOURCES{Color.ENDC}")
        print(f"{Color.HEADER}{'─' * 80}{Color.ENDC}")
        print(f"  CPU: {self.draw_gauge(cpu_percent, 100)}")
        print(f"  Memory: {self.draw_gauge(memory.percent, 100)}")
        print(f"  Memory Used: {memory.used / 1024**3:.1f}/{memory.total / 1024**3:.1f} GB")
        
        # GPU if available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  GPU Memory: {self.draw_gauge(gpu_mem, gpu_total)}")
        except:
            pass
        
        print(f"\n{Color.HEADER}{'═' * 80}{Color.ENDC}")
        print(f"{Color.OKCYAN}Press Ctrl+C to exit{Color.ENDC}")
        print(f"{Color.HEADER}{'═' * 80}{Color.ENDC}\n")
    
    def run(self, refresh_interval: int = 3):
        """Run the monitoring dashboard"""
        print(f"{Color.OKGREEN}Starting LLM System Monitor...{Color.ENDC}")
        time.sleep(1)
        
        try:
            while True:
                self.display_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print(f"\n\n{Color.WARNING}Monitor stopped.{Color.ENDC}\n")


def main():
    """Main function"""
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    refresh = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    monitor = LLMSystemMonitor(api_url)
    monitor.run(refresh_interval=refresh)


if __name__ == "__main__":
    main()
