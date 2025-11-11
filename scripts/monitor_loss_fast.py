#!/usr/bin/env python3
"""
Fast loss monitor - Updates every 10s with quick metrics.
Writes to a separate section of the step summary for rapid feedback.
"""

import os
import sys
import time
import re
from pathlib import Path
from datetime import datetime


class FastLossMonitor:
    def __init__(self, log_file, output_file, update_interval=10):
        self.log_file = Path(log_file)
        self.output_file = Path(output_file)
        self.update_interval = update_interval
        self.last_position = 0
        self.start_time = time.time()

    def parse_latest_metrics(self):
        """Read log file and get the latest metrics."""
        if not self.log_file.exists():
            return None

        latest = None
        try:
            with open(self.log_file, 'r') as f:
                f.seek(self.last_position)
                lines = f.readlines()
                self.last_position = f.tell()

                # Parse from end to find latest
                for line in reversed(lines):
                    iter_match = re.search(r'iter (\d+):', line)
                    if iter_match:
                        latest = {'iter': int(iter_match.group(1))}

                        loss_match = re.search(r'loss ([\d.]+)', line)
                        if loss_match:
                            latest['loss'] = float(loss_match.group(1))

                        time_match = re.search(r'time ([\d.]+)ms', line)
                        if time_match:
                            latest['time'] = float(time_match.group(1))

                        # Found the latest training line
                        break

                # Also check for latest validation
                for line in reversed(lines):
                    val_match = re.search(r'val loss ([\d.]+)', line)
                    if val_match:
                        latest['val_loss'] = float(val_match.group(1))
                        break

        except Exception as e:
            print(f"Error parsing log: {e}", file=sys.stderr)
            return None

        return latest

    def update_display(self, metrics, max_iter=2000):
        """Update the fast display with latest metrics."""
        if not metrics:
            return

        elapsed = time.time() - self.start_time
        current_iter = metrics.get('iter', 0)
        progress_pct = (current_iter / max_iter) * 100

        # Create compact display
        output = [
            "# 🚀 Live Training Metrics (Fast Update)",
            "",
            f"**Last Update:** {datetime.now().strftime('%H:%M:%S')} UTC | **Elapsed:** {int(elapsed)}s",
            "",
            "```",
            f"Iteration: {current_iter:,} / {max_iter:,} ({progress_pct:.1f}%)",
        ]

        if 'loss' in metrics:
            output.append(f"Train Loss: {metrics['loss']:.4f}")

        if 'val_loss' in metrics:
            output.append(f"Val Loss:   {metrics['val_loss']:.4f}")

        if 'time' in metrics:
            iters_per_sec = 1000.0 / metrics['time'] if metrics['time'] > 0 else 0
            output.append(f"Speed:      {metrics['time']:.2f}ms/iter ({iters_per_sec:.2f} iter/s)")

        # Simple progress bar
        bar_width = 40
        filled = int(bar_width * current_iter / max_iter)
        bar = '█' * filled + '░' * (bar_width - filled)
        output.append(f"\n{bar}")

        output.append("```")
        output.append("")
        output.append("*Updates every 10 seconds - See detailed monitor below for full analysis*")

        # Write to file
        with open(self.output_file, 'w') as f:
            f.write('\n'.join(output))

        print(f"[FastMonitor] Updated - Iter {current_iter} | Loss {metrics.get('loss', 'N/A')}", file=sys.stderr)

    def monitor(self, max_iter=2000):
        """Main monitoring loop."""
        print(f"[FastMonitor] Starting fast loss monitor...", file=sys.stderr)
        print(f"[FastMonitor] Log file: {self.log_file}", file=sys.stderr)
        print(f"[FastMonitor] Output: {self.output_file}", file=sys.stderr)
        print(f"[FastMonitor] Update interval: {self.update_interval}s", file=sys.stderr)

        while True:
            metrics = self.parse_latest_metrics()

            if metrics:
                self.update_display(metrics, max_iter)

                # Check if training is complete
                if metrics.get('iter', 0) >= max_iter:
                    print(f"[FastMonitor] Training complete!", file=sys.stderr)
                    return

            time.sleep(self.update_interval)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fast loss monitor for training')
    parser.add_argument('--log-file', default='training.log',
                       help='Path to training log file')
    parser.add_argument('--output-file', default='fast_monitor.md',
                       help='Output file for fast updates')
    parser.add_argument('--update-interval', type=int, default=10,
                       help='Seconds between updates')
    parser.add_argument('--max-iter', type=int, default=2000,
                       help='Maximum training iterations')

    args = parser.parse_args()

    monitor = FastLossMonitor(args.log_file, args.output_file, args.update_interval)

    try:
        monitor.monitor(max_iter=args.max_iter)
    except KeyboardInterrupt:
        print("\n[FastMonitor] Stopped by user", file=sys.stderr)
        sys.exit(0)


if __name__ == '__main__':
    main()
