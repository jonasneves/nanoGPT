#!/usr/bin/env python3
"""
Extract key metrics from training logs for workflow outputs.
"""

import re
import json
import argparse
import sys
from pathlib import Path


def extract_metrics(log_file):
    """Extract all key metrics from training log."""
    metrics = {
        'iterations_completed': 0,
        'final_train_loss': None,
        'final_val_loss': None,
        'best_val_loss': None,
        'avg_iteration_time_ms': None,
        'avg_mfu_percent': None,
        'total_train_time_seconds': None,
        'validation_count': 0,
    }

    train_losses = []
    val_losses = []
    times = []
    mfus = []
    max_iter = 0

    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Extract all training iterations
        for match in re.finditer(r'iter (\d+):.*?loss ([\d.]+)', content):
            iteration = int(match.group(1))
            loss = float(match.group(2))
            max_iter = max(max_iter, iteration)
            train_losses.append(loss)

        # Extract validation losses
        for match in re.finditer(r'val loss ([\d.]+)', content):
            val_losses.append(float(match.group(1)))

        # Extract iteration times
        for match in re.finditer(r'time ([\d.]+)ms', content):
            times.append(float(match.group(1)))

        # Extract MFU values
        for match in re.finditer(r'mfu ([\d.]+)%', content):
            mfus.append(float(match.group(1)))

        # Calculate metrics
        metrics['iterations_completed'] = max_iter
        metrics['final_train_loss'] = train_losses[-1] if train_losses else None
        metrics['final_val_loss'] = val_losses[-1] if val_losses else None
        metrics['best_val_loss'] = min(val_losses) if val_losses else None
        metrics['avg_iteration_time_ms'] = sum(times) / len(times) if times else None
        metrics['avg_mfu_percent'] = sum(mfus) / len(mfus) if mfus else None
        metrics['validation_count'] = len(val_losses)

        # Try to extract total training time
        time_match = re.search(r'total time: ([\d.]+)s', content)
        if time_match:
            metrics['total_train_time_seconds'] = float(time_match.group(1))

    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error parsing log file: {e}", file=sys.stderr)
        return None

    return metrics


def format_for_github_output(metrics):
    """Format metrics as GitHub Actions output variables."""
    if not metrics:
        return ""

    output = []
    for key, value in metrics.items():
        if value is not None:
            # Format floats to 4 decimal places
            if isinstance(value, float):
                output.append(f"{key}={value:.4f}")
            else:
                output.append(f"{key}={value}")

    return '\n'.join(output)


def create_badge_data(metrics):
    """Create data for GitHub badges."""
    badges = {}

    if metrics.get('final_val_loss'):
        loss = metrics['final_val_loss']
        # Determine color based on loss value
        if loss < 2.0:
            color = 'brightgreen'
        elif loss < 3.0:
            color = 'green'
        elif loss < 4.0:
            color = 'yellow'
        else:
            color = 'orange'

        badges['val_loss'] = {
            'label': 'val loss',
            'message': f"{loss:.4f}",
            'color': color
        }

    if metrics.get('iterations_completed'):
        badges['iterations'] = {
            'label': 'iterations',
            'message': f"{metrics['iterations_completed']}",
            'color': 'blue'
        }

    return badges


def main():
    parser = argparse.ArgumentParser(description='Extract metrics from training log')
    parser.add_argument('--log-file', default='training.log',
                       help='Path to training log file')
    parser.add_argument('--output-json', default='extracted_metrics.json',
                       help='Output JSON file')
    parser.add_argument('--github-output', action='store_true',
                       help='Format output for GitHub Actions')
    parser.add_argument('--summary', action='store_true',
                       help='Print human-readable summary')

    args = parser.parse_args()

    metrics = extract_metrics(args.log_file)

    if metrics is None:
        return 1

    # Save JSON
    output_data = {
        'metrics': metrics,
        'badges': create_badge_data(metrics)
    }

    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Metrics extracted and saved to {args.output_json}", file=sys.stderr)

    # GitHub Actions output format
    if args.github_output:
        print(format_for_github_output(metrics))

    # Human-readable summary
    if args.summary:
        print("\nTraining Metrics Summary:", file=sys.stderr)
        print("=" * 50, file=sys.stderr)

        if metrics['iterations_completed']:
            print(f"Iterations Completed: {metrics['iterations_completed']}", file=sys.stderr)

        if metrics['final_train_loss']:
            print(f"Final Training Loss: {metrics['final_train_loss']:.4f}", file=sys.stderr)

        if metrics['final_val_loss']:
            print(f"Final Validation Loss: {metrics['final_val_loss']:.4f}", file=sys.stderr)

        if metrics['best_val_loss']:
            print(f"Best Validation Loss: {metrics['best_val_loss']:.4f}", file=sys.stderr)

        if metrics['avg_iteration_time_ms']:
            print(f"Avg Iteration Time: {metrics['avg_iteration_time_ms']:.2f}ms", file=sys.stderr)

        if metrics['avg_mfu_percent']:
            print(f"Avg MFU: {metrics['avg_mfu_percent']:.2f}%", file=sys.stderr)

        if metrics['total_train_time_seconds']:
            hours = int(metrics['total_train_time_seconds'] // 3600)
            minutes = int((metrics['total_train_time_seconds'] % 3600) // 60)
            seconds = int(metrics['total_train_time_seconds'] % 60)
            print(f"Total Training Time: {hours}h {minutes}m {seconds}s", file=sys.stderr)

        print("=" * 50, file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
