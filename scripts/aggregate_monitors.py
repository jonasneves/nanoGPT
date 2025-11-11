#!/usr/bin/env python3
"""
Aggregate all monitor outputs and write to GitHub Step Summary.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime


def aggregate_monitors(step_summary_file, monitor_files):
    """Combine all monitor outputs into step summary."""

    output = [
        "# 🔄 Live Training Dashboard",
        "",
        f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "---",
        ""
    ]

    # Add each monitor's output
    for label, filepath in monitor_files.items():
        if Path(filepath).exists():
            try:
                with open(filepath, 'r') as f:
                    content = f.read().strip()
                    if content:
                        output.append(content)
                        output.append("")
                        output.append("---")
                        output.append("")
            except Exception as e:
                output.append(f"Error reading {label}: {e}")
                output.append("")

    # Write to step summary
    try:
        with open(step_summary_file, 'w') as f:
            f.write('\n'.join(output))
        return True
    except Exception as e:
        print(f"Error writing to step summary: {e}", file=sys.stderr)
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Aggregate monitor outputs')
    parser.add_argument('--interval', type=int, default=20,
                       help='Update interval in seconds')
    parser.add_argument('--duration', type=int, default=None,
                       help='Max duration in seconds')

    args = parser.parse_args()

    step_summary = os.environ.get('GITHUB_STEP_SUMMARY', 'step_summary.md')

    monitor_files = {
        'fast': 'fast_monitor.md',
        'resource': 'resource_monitor.md',
        'detailed': 'detailed_monitor.md',
        'samples': 'intermediate_samples.md'
    }

    print(f"[Aggregator] Starting monitor aggregator", file=sys.stderr)
    print(f"[Aggregator] Step summary file: {step_summary}", file=sys.stderr)
    print(f"[Aggregator] Update interval: {args.interval}s", file=sys.stderr)

    start = time.time()
    updates = 0

    while True:
        if aggregate_monitors(step_summary, monitor_files):
            updates += 1
            print(f"[Aggregator] Update #{updates} written to step summary", file=sys.stderr)

        if args.duration and (time.time() - start) > args.duration:
            break

        time.sleep(args.interval)


if __name__ == '__main__':
    main()
