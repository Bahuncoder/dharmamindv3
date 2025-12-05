#!/usr/bin/env python3
"""
🕉️ DharmaLLM Training Monitor Dashboard
========================================

Real-time comprehensive monitoring for Sanskrit model training:
- Training progress (epochs, batches, loss)
- System resources (CPU, Memory, GPU)
- Model checkpoints
- Corpus statistics
- Time estimates
- Loss graphs (ASCII)
"""

import os
import sys
import time
import json
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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
    UNDERLINE = '\033[4m'


class TrainingMonitor:
    def __init__(self):
        self.training_process = None
        self.checkpoint_dir = Path("model/checkpoints")
        self.log_file = Path("training_log.txt")
        self.corpus_file = Path("data/master_corpus/MASTER_SANSKRIT_CORPUS.json")
        self.start_time = None
        self.loss_history = []
        
    def find_training_process(self) -> Optional[psutil.Process]:
        """Find the training process"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('train_master_corpus' in str(cmd) for cmd in cmdline):
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def get_process_stats(self, proc: psutil.Process) -> Dict:
        """Get process resource usage"""
        try:
            return {
                'pid': proc.pid,
                'cpu_percent': proc.cpu_percent(interval=0.1),
                'memory_mb': proc.memory_info().rss / 1024 / 1024,
                'memory_percent': proc.memory_percent(),
                'num_threads': proc.num_threads(),
                'status': proc.status(),
                'create_time': proc.create_time(),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}
    
    def get_system_stats(self) -> Dict:
        """Get overall system stats"""
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        stats = {
            'cpu_cores': psutil.cpu_count(),
            'cpu_percent_avg': sum(cpu_percent) / len(cpu_percent),
            'cpu_percent_per_core': cpu_percent,
            'memory_total_gb': memory.total / 1024**3,
            'memory_used_gb': memory.used / 1024**3,
            'memory_percent': memory.percent,
            'disk_total_gb': disk.total / 1024**3,
            'disk_used_gb': disk.used / 1024**3,
            'disk_percent': disk.percent,
        }
        
        # GPU stats if available
        try:
            import torch
            if torch.cuda.is_available():
                stats['gpu_available'] = True
                stats['gpu_name'] = torch.cuda.get_device_name(0)
                stats['gpu_memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**3
                stats['gpu_memory_reserved'] = torch.cuda.memory_reserved(0) / 1024**3
                stats['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except:
            stats['gpu_available'] = False
        
        return stats
    
    def get_checkpoint_stats(self) -> Dict:
        """Get checkpoint information"""
        if not self.checkpoint_dir.exists():
            return {'count': 0, 'latest': None}
        
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        stats = {
            'count': len(checkpoints),
            'checkpoints': []
        }
        
        for ckpt in checkpoints:
            stat = ckpt.stat()
            stats['checkpoints'].append({
                'name': ckpt.name,
                'size_mb': stat.st_size / 1024 / 1024,
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%H:%M:%S'),
                'age_minutes': (time.time() - stat.st_mtime) / 60
            })
        
        return stats
    
    def parse_log_file(self) -> Dict:
        """Parse training log for progress"""
        if not self.log_file.exists():
            return {'epochs_complete': 0, 'latest_loss': None}
        
        try:
            with open(self.log_file, 'r') as f:
                content = f.read()
            
            # Try to extract epoch info
            import re
            epoch_matches = re.findall(r'Epoch (\d+)/(\d+)', content)
            loss_matches = re.findall(r'Loss[:\s]+(\d+\.\d+)', content)
            
            latest_epoch = max([int(m[0]) for m in epoch_matches]) if epoch_matches else 0
            total_epochs = int(epoch_matches[0][1]) if epoch_matches else 20
            latest_loss = float(loss_matches[-1]) if loss_matches else None
            
            return {
                'epochs_complete': latest_epoch,
                'total_epochs': total_epochs,
                'latest_loss': latest_loss,
                'loss_history': [float(l) for l in loss_matches[-10:]]
            }
        except:
            return {'epochs_complete': 0, 'latest_loss': None}
    
    def get_corpus_stats(self) -> Dict:
        """Get corpus statistics"""
        if not self.corpus_file.exists():
            return {'texts': 0, 'verses': 0}
        
        try:
            with open(self.corpus_file, 'r') as f:
                data = json.load(f)
            
            total_chars = sum(len(str(item.get('text', ''))) for item in data)
            
            return {
                'texts': len(data),
                'total_chars': total_chars,
                'estimated_verses': total_chars // 80,
                'avg_text_length': total_chars // len(data) if data else 0
            }
        except:
            return {'texts': 0, 'verses': 0}
    
    def draw_progress_bar(self, current: int, total: int, width: int = 40) -> str:
        """Draw ASCII progress bar"""
        if total == 0:
            return "[" + "?" * width + "]"
        
        percent = current / total
        filled = int(width * percent)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {percent*100:.1f}%"
    
    def draw_loss_graph(self, losses: List[float], width: int = 60, height: int = 10) -> str:
        """Draw ASCII loss graph"""
        if not losses or len(losses) < 2:
            return "No loss data yet"
        
        min_loss = min(losses)
        max_loss = max(losses)
        loss_range = max_loss - min_loss if max_loss > min_loss else 1
        
        graph = []
        for i in range(height):
            line = []
            threshold = max_loss - (i / height) * loss_range
            
            for loss in losses[-width:]:
                if loss >= threshold:
                    line.append("█")
                else:
                    line.append(" ")
            
            graph.append(f"{threshold:6.4f} │{''.join(line)}")
        
        graph.append("       └" + "─" * width)
        return "\n".join(graph)
    
    def display_dashboard(self):
        """Display the main dashboard"""
        # Clear screen
        os.system('clear')
        
        # Header
        print(f"\n{Color.HEADER}{'═' * 80}{Color.ENDC}")
        print(f"{Color.HEADER}{Color.BOLD}🕉️  DharmaLLM Training Monitor Dashboard{Color.ENDC}")
        print(f"{Color.HEADER}{'═' * 80}{Color.ENDC}\n")
        print(f"{Color.OKCYAN}⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Color.ENDC}\n")
        
        # Find training process
        proc = self.find_training_process()
        
        if not proc:
            print(f"{Color.WARNING}⚠️  No training process found{Color.ENDC}\n")
            print("Looking for process with 'train_master_corpus' in command...")
            print("\nTo start training:")
            print("  python3 train_master_corpus.py > training_log.txt 2>&1 &")
            return
        
        # Process stats
        proc_stats = self.get_process_stats(proc)
        runtime = time.time() - proc_stats['create_time']
        runtime_str = str(timedelta(seconds=int(runtime)))
        
        print(f"{Color.OKGREEN}{'─' * 80}{Color.ENDC}")
        print(f"{Color.OKGREEN}{Color.BOLD}📊 TRAINING PROCESS{Color.ENDC}")
        print(f"{Color.OKGREEN}{'─' * 80}{Color.ENDC}")
        print(f"  Status: {Color.OKGREEN}✅ RUNNING{Color.ENDC}")
        print(f"  PID: {proc_stats['pid']}")
        print(f"  Runtime: {runtime_str}")
        print(f"  CPU Usage: {proc_stats['cpu_percent']:.1f}%")
        print(f"  Memory: {proc_stats['memory_mb']:.1f} MB ({proc_stats['memory_percent']:.1f}%)")
        print(f"  Threads: {proc_stats['num_threads']}")
        
        # Training progress
        log_stats = self.parse_log_file()
        print(f"\n{Color.OKBLUE}{'─' * 80}{Color.ENDC}")
        print(f"{Color.OKBLUE}{Color.BOLD}🎓 TRAINING PROGRESS{Color.ENDC}")
        print(f"{Color.OKBLUE}{'─' * 80}{Color.ENDC}")
        
        if log_stats['epochs_complete'] > 0:
            total_epochs = log_stats.get('total_epochs', 20)
            progress_bar = self.draw_progress_bar(log_stats['epochs_complete'], total_epochs)
            print(f"  Epochs: {log_stats['epochs_complete']}/{total_epochs}")
            print(f"  {progress_bar}")
            
            if log_stats['latest_loss']:
                print(f"  Latest Loss: {Color.WARNING}{log_stats['latest_loss']:.6f}{Color.ENDC}")
            
            # Estimate completion
            if log_stats['epochs_complete'] > 0:
                time_per_epoch = runtime / log_stats['epochs_complete']
                remaining_epochs = total_epochs - log_stats['epochs_complete']
                eta_seconds = time_per_epoch * remaining_epochs
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                print(f"  Time per epoch: ~{time_per_epoch/60:.1f} minutes")
                print(f"  ETA: {eta_str}")
        else:
            print("  Initializing... (log may be buffered)")
        
        # Loss graph
        if log_stats.get('loss_history') and len(log_stats['loss_history']) > 1:
            print(f"\n  Loss History (last 10 readings):")
            print(self.draw_loss_graph(log_stats['loss_history']))
        
        # Checkpoints
        ckpt_stats = self.get_checkpoint_stats()
        print(f"\n{Color.OKCYAN}{'─' * 80}{Color.ENDC}")
        print(f"{Color.OKCYAN}{Color.BOLD}💾 CHECKPOINTS{Color.ENDC}")
        print(f"{Color.OKCYAN}{'─' * 80}{Color.ENDC}")
        print(f"  Total: {ckpt_stats['count']}")
        
        if ckpt_stats['checkpoints']:
            for ckpt in ckpt_stats['checkpoints'][:3]:
                age = f"{ckpt['age_minutes']:.0f}m ago" if ckpt['age_minutes'] < 60 else f"{ckpt['age_minutes']/60:.1f}h ago"
                print(f"  • {ckpt['name']}: {ckpt['size_mb']:.0f} MB ({age})")
        
        # Corpus stats
        corpus_stats = self.get_corpus_stats()
        print(f"\n{Color.WARNING}{'─' * 80}{Color.ENDC}")
        print(f"{Color.WARNING}{Color.BOLD}📚 CORPUS STATISTICS{Color.ENDC}")
        print(f"{Color.WARNING}{'─' * 80}{Color.ENDC}")
        print(f"  Total texts: {corpus_stats.get('texts', 0):,}")
        print(f"  Total characters: {corpus_stats.get('total_chars', 0):,}")
        print(f"  Estimated verses: ~{corpus_stats.get('estimated_verses', 0):,}")
        print(f"  Average text length: {corpus_stats.get('avg_text_length', 0)} chars")
        
        # System resources
        sys_stats = self.get_system_stats()
        print(f"\n{Color.HEADER}{'─' * 80}{Color.ENDC}")
        print(f"{Color.HEADER}{Color.BOLD}💻 SYSTEM RESOURCES{Color.ENDC}")
        print(f"{Color.HEADER}{'─' * 80}{Color.ENDC}")
        print(f"  CPU: {sys_stats['cpu_percent_avg']:.1f}% ({sys_stats['cpu_cores']} cores)")
        print(f"  Memory: {sys_stats['memory_used_gb']:.1f}/{sys_stats['memory_total_gb']:.1f} GB ({sys_stats['memory_percent']:.1f}%)")
        print(f"  Disk: {sys_stats['disk_used_gb']:.0f}/{sys_stats['disk_total_gb']:.0f} GB ({sys_stats['disk_percent']:.1f}%)")
        
        if sys_stats.get('gpu_available'):
            print(f"  GPU: {sys_stats['gpu_name']}")
            print(f"  GPU Memory: {sys_stats['gpu_memory_allocated']:.1f}/{sys_stats['gpu_memory_total']:.1f} GB")
        
        print(f"\n{Color.HEADER}{'═' * 80}{Color.ENDC}")
        print(f"{Color.OKCYAN}Press Ctrl+C to exit (training will continue){Color.ENDC}")
        print(f"{Color.HEADER}{'═' * 80}{Color.ENDC}\n")
    
    def run(self, refresh_interval: int = 5):
        """Run the monitoring dashboard"""
        print(f"{Color.OKGREEN}Starting Training Monitor...{Color.ENDC}")
        time.sleep(1)
        
        try:
            while True:
                self.display_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print(f"\n\n{Color.WARNING}Monitor stopped. Training continues in background.{Color.ENDC}\n")


def main():
    """Main function"""
    monitor = TrainingMonitor()
    refresh = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    monitor.run(refresh_interval=refresh)


if __name__ == "__main__":
    main()
