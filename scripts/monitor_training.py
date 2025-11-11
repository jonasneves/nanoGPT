#!/usr/bin/env python3
"""
Real-time training monitor for GitHub Actions.
Watches training.log and updates GitHub Step Summary with progress.
"""

import os
import sys
import time
import re
from pathlib import Path
from datetime import datetime, timedelta


class TrainingMonitor:
    def __init__(self, log_file, summary_file, update_interval=30):
        self.log_file = Path(log_file)
        self.summary_file = Path(summary_file)
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_position = 0

    def parse_log_line(self, line):
        """Parse a training log line and extract metrics."""
        # Example: iter 100: loss 4.5678, time 123.45ms, mfu 12.34%
        patterns = {
            'iter': r'iter (\d+):',
            'loss': r'loss ([\d.]+)',
            'time': r'time ([\d.]+)ms',
            'mfu': r'mfu ([\d.]+)%',
            'val_loss': r'val loss ([\d.]+)',
        }

        result = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                result[key] = match.group(1)

        return result if result else None

    def read_new_lines(self):
        """Read new lines from log file since last position."""
        if not self.log_file.exists():
            return []

        with open(self.log_file, 'r') as f:
            f.seek(self.last_position)
            new_lines = f.readlines()
            self.last_position = f.tell()

        return new_lines

    def format_time(self, seconds):
        """Format seconds into human-readable time."""
        return str(timedelta(seconds=int(seconds)))

    def estimate_completion(self, current_iter, max_iter, elapsed):
        """Estimate time to completion."""
        if current_iter == 0:
            return "Calculating..."

        iters_per_sec = current_iter / elapsed
        remaining_iters = max_iter - current_iter
        remaining_seconds = remaining_iters / iters_per_sec

        return self.format_time(remaining_seconds)

    def update_summary(self, metrics_history, max_iter=2000):
        """Update GitHub Step Summary with current progress."""
        if not metrics_history:
            return

        latest = metrics_history[-1]
        current_iter = int(latest.get('iter', 0))

        # Calculate progress
        progress_pct = (current_iter / max_iter) * 100
        elapsed = time.time() - self.start_time
        eta = self.estimate_completion(current_iter, max_iter, elapsed)

        # Get recent validation losses
        val_losses = [m for m in metrics_history if 'val_loss' in m]
        recent_val = val_losses[-5:] if val_losses else []

        # Create progress bar
        bar_length = 30
        filled = int(bar_length * current_iter / max_iter)
        bar = '█' * filled + '░' * (bar_length - filled)

        # Build summary
        summary = f"""# Training Progress Monitor

## Status: {'✅ Complete' if current_iter >= max_iter else '🔄 Training'}

### Overall Progress
{bar} **{progress_pct:.1f}%** ({current_iter:,} / {max_iter:,} iterations)

### Timing
- **Elapsed:** {self.format_time(elapsed)}
- **ETA:** {eta}
- **Started:** {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S UTC')}

### Latest Metrics
"""

        if 'loss' in latest:
            summary += f"- **Current Loss:** {float(latest['loss']):.4f}\n"
        if 'time' in latest:
            summary += f"- **Iteration Time:** {latest['time']}ms\n"
        if 'mfu' in latest:
            summary += f"- **MFU:** {latest['mfu']}%\n"

        # Add validation loss table
        if recent_val:
            summary += "\n### Recent Validation Losses\n\n"
            summary += "| Iteration | Val Loss | Trend |\n"
            summary += "|-----------|----------|-------|\n"

            for i, m in enumerate(recent_val):
                trend = ""
                if i > 0:
                    prev_loss = float(recent_val[i-1]['val_loss'])
                    curr_loss = float(m['val_loss'])
                    if curr_loss < prev_loss:
                        trend = "📉 Improving"
                    elif curr_loss > prev_loss:
                        trend = "📈 Increasing"
                    else:
                        trend = "➡️ Stable"

                summary += f"| {m['iter']} | {float(m['val_loss']):.4f} | {trend} |\n"

        # Add quick stats
        if len(metrics_history) > 1:
            train_losses = [float(m['loss']) for m in metrics_history if 'loss' in m]
            if train_losses:
                summary += f"\n### Training Loss Statistics\n"
                summary += f"- **Min:** {min(train_losses):.4f}\n"
                summary += f"- **Max:** {max(train_losses):.4f}\n"
                summary += f"- **Current:** {train_losses[-1]:.4f}\n"

                # Show trend
                if len(train_losses) > 10:
                    recent_avg = sum(train_losses[-10:]) / 10
                    older_avg = sum(train_losses[-20:-10]) / 10 if len(train_losses) > 20 else recent_avg
                    if recent_avg < older_avg:
                        summary += f"- **Trend:** 📉 Improving (recent avg: {recent_avg:.4f})\n"
                    else:
                        summary += f"- **Trend:** 📈 Recent avg: {recent_avg:.4f}\n"

        summary += f"\n---\n*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*\n"

        # Write to summary file
        with open(self.summary_file, 'w') as f:
            f.write(summary)

        print(f"✓ Summary updated - Iter {current_iter}/{max_iter} ({progress_pct:.1f}%)")

    def monitor(self, max_iter=2000, duration=None):
        """Main monitoring loop."""
        print(f"Starting training monitor...")
        print(f"Log file: {self.log_file}")
        print(f"Summary file: {self.summary_file}")
        print(f"Update interval: {self.update_interval}s")
        print("-" * 60)

        metrics_history = []
        last_update = 0
        start = time.time()

        while True:
            # Check if we should stop
            if duration and (time.time() - start) > duration:
                print("Duration limit reached, stopping monitor")
                break

            # Read new log lines
            new_lines = self.read_new_lines()

            for line in new_lines:
                metrics = self.parse_log_line(line)
                if metrics:
                    metrics_history.append(metrics)

                    # Check if training is complete
                    if 'iter' in metrics and int(metrics['iter']) >= max_iter:
                        print(f"Training complete! Reached {max_iter} iterations")
                        self.update_summary(metrics_history, max_iter)
                        return

            # Update summary periodically
            current_time = time.time()
            if current_time - last_update >= self.update_interval:
                self.update_summary(metrics_history, max_iter)
                last_update = current_time

            # Sleep before next check
            time.sleep(5)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('--log-file', default='training.log', help='Path to training log file')
    parser.add_argument('--summary-file', default=os.environ.get('GITHUB_STEP_SUMMARY', 'summary.md'),
                       help='Path to GitHub step summary file')
    parser.add_argument('--update-interval', type=int, default=30,
                       help='Seconds between summary updates')
    parser.add_argument('--max-iter', type=int, default=2000,
                       help='Maximum training iterations')
    parser.add_argument('--duration', type=int, default=None,
                       help='Maximum monitoring duration in seconds (optional)')

    args = parser.parse_args()

    monitor = TrainingMonitor(args.log_file, args.summary_file, args.update_interval)

    try:
        monitor.monitor(max_iter=args.max_iter, duration=args.duration)
    except KeyboardInterrupt:
        print("\nMonitor stopped by user")
        sys.exit(0)


if __name__ == '__main__':
    main()
