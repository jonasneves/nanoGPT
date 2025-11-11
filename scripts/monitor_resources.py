#!/usr/bin/env python3
"""
Monitor system resource usage during training.
Tracks CPU, memory, and disk usage.
"""

import os
import sys
import time
import psutil
from pathlib import Path
from datetime import datetime
from collections import deque


class ResourceMonitor:
    def __init__(self, output_file, update_interval=15, history_size=20):
        self.output_file = Path(output_file)
        self.update_interval = update_interval
        self.history_size = history_size
        self.cpu_history = deque(maxlen=history_size)
        self.mem_history = deque(maxlen=history_size)
        self.start_time = time.time()
        self.training_pid = None

    def find_training_process(self):
        """Find the training Python process."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline') or []
                if 'python' in proc.info['name'].lower() and any('train.py' in cmd for cmd in cmdline):
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return None

    def get_metrics(self):
        """Get current system metrics."""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=0.5),
            'mem_percent': psutil.virtual_memory().percent,
            'mem_used_gb': psutil.virtual_memory().used / (1024**3),
            'mem_available_gb': psutil.virtual_memory().available / (1024**3),
        }

        # Try to get process-specific metrics
        if self.training_pid is None:
            proc = self.find_training_process()
            if proc:
                self.training_pid = proc.pid
                print(f"[ResourceMonitor] Found training process: PID {self.training_pid}", file=sys.stderr)

        if self.training_pid:
            try:
                proc = psutil.Process(self.training_pid)
                metrics['proc_cpu_percent'] = proc.cpu_percent()
                metrics['proc_mem_mb'] = proc.memory_info().rss / (1024**2)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.training_pid = None

        return metrics

    def create_ascii_graph(self, data, height=8, width=40):
        """Create a simple ASCII graph."""
        if not data or len(data) < 2:
            return ["No data yet"]

        min_val = min(data)
        max_val = max(data)
        val_range = max_val - min_val if max_val != min_val else 1

        graph = []
        for row in range(height):
            line = []
            threshold = max_val - (row / (height - 1)) * val_range

            for i in range(min(len(data), width)):
                idx = max(0, len(data) - width) + i
                if data[idx] >= threshold:
                    line.append('█')
                else:
                    line.append(' ')

            graph.append(''.join(line))

        return graph

    def update_display(self):
        """Update the display with current metrics."""
        metrics = self.get_metrics()

        # Store history
        self.cpu_history.append(metrics['cpu_percent'])
        self.mem_history.append(metrics['mem_percent'])

        elapsed = time.time() - self.start_time

        # Build output
        output = [
            "# 📊 System Resource Monitor",
            "",
            f"**Monitoring Duration:** {int(elapsed)}s | **Last Update:** {datetime.now().strftime('%H:%M:%S UTC')}",
            "",
            "## Current Usage",
            "",
            "```",
            f"CPU:    {metrics['cpu_percent']:5.1f}% {'█' * int(metrics['cpu_percent']/5)}",
            f"Memory: {metrics['mem_percent']:5.1f}% {'█' * int(metrics['mem_percent']/5)}",
            f"        {metrics['mem_used_gb']:.2f} GB used / {metrics['mem_available_gb']:.2f} GB available",
        ]

        if 'proc_cpu_percent' in metrics:
            output.append("")
            output.append("Training Process:")
            output.append(f"  CPU:    {metrics['proc_cpu_percent']:5.1f}%")
            output.append(f"  Memory: {metrics['proc_mem_mb']:.1f} MB")

            # Calculate per-core usage estimate
            cpu_cores = psutil.cpu_count()
            per_core = metrics['proc_cpu_percent'] / cpu_cores if cpu_cores else 0
            output.append(f"  Per-Core: {per_core:.1f}% (across {cpu_cores} cores)")

        output.append("```")

        # Add CPU history graph
        if len(self.cpu_history) > 5:
            output.append("")
            output.append("## CPU Usage History")
            output.append("")
            output.append("```")
            cpu_graph = self.create_ascii_graph(list(self.cpu_history), height=6, width=50)
            output.append(f"100% |{'─' * 50}")
            for line in cpu_graph:
                output.append(f"     |{line}")
            output.append(f"  0% |{'─' * 50}")
            output.append(f"      ← {len(self.cpu_history)} samples (older...newer →)")
            output.append("```")

        # Add memory history graph
        if len(self.mem_history) > 5:
            output.append("")
            output.append("## Memory Usage History")
            output.append("")
            output.append("```")
            mem_graph = self.create_ascii_graph(list(self.mem_history), height=6, width=50)
            output.append(f"100% |{'─' * 50}")
            for line in mem_graph:
                output.append(f"     |{line}")
            output.append(f"  0% |{'─' * 50}")
            output.append(f"      ← {len(self.mem_history)} samples (older...newer →)")
            output.append("```")

        # Statistics
        if len(self.cpu_history) > 1:
            output.append("")
            output.append("## Statistics")
            output.append("")
            output.append("| Metric | Avg | Min | Max |")
            output.append("|--------|-----|-----|-----|")

            cpu_data = list(self.cpu_history)
            mem_data = list(self.mem_history)

            output.append(f"| CPU % | {sum(cpu_data)/len(cpu_data):.1f} | {min(cpu_data):.1f} | {max(cpu_data):.1f} |")
            output.append(f"| Mem % | {sum(mem_data)/len(mem_data):.1f} | {min(mem_data):.1f} | {max(mem_data):.1f} |")

            # Add efficiency insights
            avg_cpu = sum(cpu_data) / len(cpu_data)
            output.append("")
            output.append("**Compute Insights:**")
            if avg_cpu < 30:
                output.append("- ⚠️ Low CPU utilization - training may be I/O bound")
            elif avg_cpu > 80:
                output.append("- ✅ Good CPU utilization - compute bound")
            else:
                output.append("- ℹ️ Moderate CPU utilization")

        output.append("")
        output.append("*Updates every 15 seconds*")

        # Write to file
        with open(self.output_file, 'w') as f:
            f.write('\n'.join(output))

        print(f"[ResourceMonitor] CPU: {metrics['cpu_percent']:.1f}% | Mem: {metrics['mem_percent']:.1f}%", file=sys.stderr)

    def monitor(self, duration=None):
        """Main monitoring loop."""
        print(f"[ResourceMonitor] Starting resource monitor...", file=sys.stderr)
        print(f"[ResourceMonitor] Output: {self.output_file}", file=sys.stderr)
        print(f"[ResourceMonitor] Update interval: {self.update_interval}s", file=sys.stderr)

        start = time.time()

        while True:
            self.update_display()

            if duration and (time.time() - start) > duration:
                print("[ResourceMonitor] Duration limit reached", file=sys.stderr)
                break

            time.sleep(self.update_interval)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Monitor system resources during training')
    parser.add_argument('--output-file', default='resource_monitor.md',
                       help='Output file for resource metrics')
    parser.add_argument('--update-interval', type=int, default=15,
                       help='Seconds between updates')
    parser.add_argument('--duration', type=int, default=None,
                       help='Maximum monitoring duration in seconds (optional)')

    args = parser.parse_args()

    monitor = ResourceMonitor(args.output_file, args.update_interval)

    try:
        monitor.monitor(duration=args.duration)
    except KeyboardInterrupt:
        print("\n[ResourceMonitor] Stopped by user", file=sys.stderr)
        sys.exit(0)


if __name__ == '__main__':
    main()
