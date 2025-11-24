import time
import threading
import psutil
import csv
import os
from datetime import datetime

class ResourceMonitor:
    def __init__(self, log_file="system_metrics.csv", interval=1.0):
        self.log_file = log_file
        self.interval = interval
        self.running = False
        self.thread = None
        
        # Initialize CSV
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'cpu_percent', 'memory_percent', 'memory_used_gb', 'net_sent_mb', 'net_recv_mb'])
            
        self.last_net_io = psutil.net_io_counters()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitor_loop(self):
        while self.running:
            timestamp = datetime.now().isoformat()
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            net_io = psutil.net_io_counters()
            
            # Calculate network diff
            sent_diff = (net_io.bytes_sent - self.last_net_io.bytes_sent) / (1024 * 1024) # MB
            recv_diff = (net_io.bytes_recv - self.last_net_io.bytes_recv) / (1024 * 1024) # MB
            self.last_net_io = net_io

            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, 
                    cpu_percent, 
                    memory.percent, 
                    memory.used / (1024**3), # GB
                    sent_diff,
                    recv_diff
                ])
            
            time.sleep(self.interval)

class TrainingLogger:
    def __init__(self, log_file="training_metrics.csv"):
        self.log_file = log_file
        
        # Initialize CSV
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'epoch', 'batch', 'loss', 'accuracy', 'learning_rate'])

    def log_step(self, epoch, batch, loss, learning_rate, accuracy=None):
        timestamp = datetime.now().isoformat()
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, epoch, batch, loss, accuracy if accuracy is not None else '', learning_rate])
