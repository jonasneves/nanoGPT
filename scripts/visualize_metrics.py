#!/usr/bin/env python3
"""
Generate visualizations from training logs.
Creates loss curve plots and metric summaries.
"""

import re
import json
import argparse
from pathlib import Path
from datetime import datetime


def parse_training_log(log_file):
    """Parse training log and extract all metrics."""
    metrics = {
        'iterations': [],
        'train_losses': [],
        'val_losses': [],
        'val_iterations': [],
        'times': [],
        'mfu': []
    }

    with open(log_file, 'r') as f:
        for line in f:
            # Parse training iteration
            iter_match = re.search(r'iter (\d+):', line)
            if iter_match:
                iteration = int(iter_match.group(1))

                # Training loss
                loss_match = re.search(r'loss ([\d.]+)', line)
                if loss_match:
                    metrics['iterations'].append(iteration)
                    metrics['train_losses'].append(float(loss_match.group(1)))

                # Time
                time_match = re.search(r'time ([\d.]+)ms', line)
                if time_match:
                    metrics['times'].append(float(time_match.group(1)))

                # MFU
                mfu_match = re.search(r'mfu ([\d.]+)%', line)
                if mfu_match:
                    metrics['mfu'].append(float(mfu_match.group(1)))

            # Parse validation loss
            val_match = re.search(r'val loss ([\d.]+)', line)
            if val_match and iter_match:
                metrics['val_iterations'].append(int(iter_match.group(1)))
                metrics['val_losses'].append(float(val_match.group(1)))

    return metrics


def create_ascii_plot(values, width=60, height=15, title="Loss Curve"):
    """Create an ASCII plot of values."""
    if not values or len(values) < 2:
        return "Insufficient data for plot"

    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val if max_val != min_val else 1

    # Normalize values to plot height
    normalized = [int((v - min_val) / val_range * (height - 1)) for v in values]

    # Create plot grid
    plot = []
    for row in range(height):
        line = []
        y_val = max_val - (row / (height - 1)) * val_range

        # Y-axis label
        label = f"{y_val:6.3f} |"
        line.append(label)

        # Plot points
        for col in range(min(len(normalized), width)):
            idx = int(col * len(normalized) / width)
            if normalized[idx] == (height - 1 - row):
                line.append('●')
            elif abs(normalized[idx] - (height - 1 - row)) <= 1:
                line.append('·')
            else:
                line.append(' ')

        plot.append(''.join(line))

    # Add title and x-axis
    result = [f"\n{title}", "=" * (width + 10)]
    result.extend(plot)
    result.append(" " * 8 + "+" + "-" * width)
    result.append(" " * 8 + f"0{' ' * (width-10)}iter {len(values)}")

    return '\n'.join(result)


def create_summary_table(metrics):
    """Create a markdown summary table."""
    if not metrics['train_losses']:
        return "No training data available"

    summary = ["## Training Summary\n"]

    # Overall statistics
    summary.append("| Metric | Value |")
    summary.append("|--------|-------|")
    summary.append(f"| Total Iterations | {len(metrics['train_losses'])} |")
    summary.append(f"| Final Train Loss | {metrics['train_losses'][-1]:.4f} |")

    if metrics['val_losses']:
        summary.append(f"| Final Val Loss | {metrics['val_losses'][-1]:.4f} |")
        summary.append(f"| Best Val Loss | {min(metrics['val_losses']):.4f} |")

    if metrics['times']:
        avg_time = sum(metrics['times']) / len(metrics['times'])
        summary.append(f"| Avg Iteration Time | {avg_time:.2f}ms |")

    if metrics['mfu']:
        avg_mfu = sum(metrics['mfu']) / len(metrics['mfu'])
        summary.append(f"| Avg MFU | {avg_mfu:.2f}% |")

    return '\n'.join(summary)


def create_loss_comparison(metrics):
    """Create a comparison of train vs validation loss."""
    if not metrics['val_losses']:
        return ""

    comparison = ["\n## Loss Comparison\n"]
    comparison.append("| Iteration | Train Loss | Val Loss | Delta |")
    comparison.append("|-----------|------------|----------|-------|")

    # Find matching iterations
    val_dict = dict(zip(metrics['val_iterations'], metrics['val_losses']))

    for val_iter in metrics['val_iterations'][-10:]:  # Last 10 validations
        if val_iter in metrics['iterations']:
            idx = metrics['iterations'].index(val_iter)
            train_loss = metrics['train_losses'][idx]
            val_loss = val_dict[val_iter]
            delta = val_loss - train_loss

            comparison.append(
                f"| {val_iter} | {train_loss:.4f} | {val_loss:.4f} | "
                f"{delta:+.4f} {'⚠️' if abs(delta) > 1.0 else ''} |"
            )

    return '\n'.join(comparison)


def create_progress_milestones(metrics):
    """Create milestone markers for training progress."""
    if not metrics['train_losses']:
        return ""

    milestones = ["\n## Training Milestones\n"]

    total_iters = len(metrics['train_losses'])
    checkpoints = [int(total_iters * p) for p in [0.25, 0.5, 0.75, 1.0]]

    for checkpoint in checkpoints:
        if checkpoint < len(metrics['train_losses']):
            loss = metrics['train_losses'][checkpoint]
            iteration = metrics['iterations'][checkpoint]
            pct = (checkpoint / total_iters) * 100

            milestone_emoji = "🎯" if checkpoint == total_iters - 1 else "📍"
            milestones.append(f"- {milestone_emoji} **{pct:.0f}%** (iter {iteration}): loss = {loss:.4f}")

    return '\n'.join(milestones)


def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('--log-file', default='training.log',
                       help='Path to training log file')
    parser.add_argument('--output', default='metrics_report.md',
                       help='Output markdown file')
    parser.add_argument('--json-output', default='metrics.json',
                       help='Output JSON file with raw metrics')

    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return 1

    print(f"Parsing training log: {log_path}")
    metrics = parse_training_log(log_path)

    # Generate report
    report_sections = [
        f"# Training Metrics Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"\n---\n",
        create_summary_table(metrics),
        create_progress_milestones(metrics),
    ]

    # Add ASCII plots
    if metrics['train_losses']:
        # Sample data for plotting (max 200 points for readability)
        sample_rate = max(1, len(metrics['train_losses']) // 200)
        sampled_train = metrics['train_losses'][::sample_rate]

        report_sections.append("\n## Training Loss Curve\n")
        report_sections.append("```")
        report_sections.append(create_ascii_plot(sampled_train, title="Training Loss"))
        report_sections.append("```")

    if metrics['val_losses']:
        report_sections.append("\n## Validation Loss Curve\n")
        report_sections.append("```")
        report_sections.append(create_ascii_plot(metrics['val_losses'], title="Validation Loss"))
        report_sections.append("```")

    report_sections.append(create_loss_comparison(metrics))

    # Write markdown report
    report = '\n'.join(report_sections)
    with open(args.output, 'w') as f:
        f.write(report)

    print(f"✓ Report written to: {args.output}")

    # Write JSON metrics
    json_data = {
        'generated_at': datetime.now().isoformat(),
        'total_iterations': len(metrics['train_losses']),
        'metrics': {
            'train_loss': {
                'values': metrics['train_losses'],
                'iterations': metrics['iterations'],
                'final': metrics['train_losses'][-1] if metrics['train_losses'] else None,
                'min': min(metrics['train_losses']) if metrics['train_losses'] else None,
                'max': max(metrics['train_losses']) if metrics['train_losses'] else None,
            },
            'val_loss': {
                'values': metrics['val_losses'],
                'iterations': metrics['val_iterations'],
                'final': metrics['val_losses'][-1] if metrics['val_losses'] else None,
                'best': min(metrics['val_losses']) if metrics['val_losses'] else None,
            },
            'performance': {
                'avg_iteration_time_ms': sum(metrics['times']) / len(metrics['times']) if metrics['times'] else None,
                'avg_mfu_percent': sum(metrics['mfu']) / len(metrics['mfu']) if metrics['mfu'] else None,
            }
        }
    }

    with open(args.json_output, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"✓ JSON metrics written to: {args.json_output}")

    # Print summary to console
    print("\n" + "=" * 60)
    print(create_summary_table(metrics))
    print("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
