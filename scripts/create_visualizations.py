#!/usr/bin/env python3
"""
Create PNG visualizations from training metrics.
Generates actual charts that can be embedded in GitHub Step Summary.
"""

import re
import json
import argparse
import sys
from pathlib import Path


def parse_training_log(log_file):
    """Parse training log and extract metrics."""
    metrics = {
        'iterations': [],
        'train_losses': [],
        'val_losses': [],
        'val_iterations': [],
    }

    try:
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

                # Parse validation loss
                val_match = re.search(r'val loss ([\d.]+)', line)
                if val_match and iter_match:
                    metrics['val_iterations'].append(int(iter_match.group(1)))
                    metrics['val_losses'].append(float(val_match.group(1)))

    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file}", file=sys.stderr)
        return None

    return metrics


def create_loss_plot(metrics, output_file='loss_curve.png'):
    """Create loss curve visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot training loss
        if metrics['train_losses']:
            # Sample if too many points
            max_points = 500
            step = max(1, len(metrics['train_losses']) // max_points)
            sampled_iters = metrics['iterations'][::step]
            sampled_losses = metrics['train_losses'][::step]

            ax.plot(sampled_iters, sampled_losses,
                   label='Training Loss', color='#3498db', linewidth=2, alpha=0.7)

        # Plot validation loss
        if metrics['val_losses']:
            ax.plot(metrics['val_iterations'], metrics['val_losses'],
                   label='Validation Loss', color='#e74c3c',
                   linewidth=2, marker='o', markersize=6)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Loss curve saved to {output_file}", file=sys.stderr)
        return True

    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error creating plot: {e}", file=sys.stderr)
        return False


def create_progress_chart(metrics, output_file='progress.png'):
    """Create progress/milestone chart."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not metrics['train_losses']:
            return False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Loss progression with milestones
        total = len(metrics['train_losses'])
        milestones = [int(total * p) for p in [0.25, 0.5, 0.75, 1.0]]

        ax1.plot(metrics['iterations'], metrics['train_losses'],
                color='#3498db', linewidth=2, alpha=0.7)

        for idx in milestones:
            if idx < len(metrics['train_losses']):
                ax1.axvline(x=metrics['iterations'][idx],
                           color='red', linestyle='--', alpha=0.5, linewidth=1)
                ax1.text(metrics['iterations'][idx], ax1.get_ylim()[1],
                        f"{(idx/total)*100:.0f}%",
                        rotation=90, va='top', fontsize=9)

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Progress with Milestones')
        ax1.grid(True, alpha=0.3)

        # Right: Distribution
        ax2.hist(metrics['train_losses'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        ax2.axvline(x=metrics['train_losses'][-1], color='red', linestyle='--', linewidth=2, label='Final Loss')
        ax2.set_xlabel('Loss Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Loss Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Progress chart saved to {output_file}", file=sys.stderr)
        return True

    except ImportError:
        return False
    except Exception as e:
        print(f"Error creating progress chart: {e}", file=sys.stderr)
        return False


def create_summary_with_images(metrics, output_file='visual_summary.md'):
    """Create markdown summary with embedded images."""
    summary = [
        "# 📈 Visual Training Summary",
        "",
        "## Loss Curves",
        "",
    ]

    # Check if images exist
    if Path('loss_curve.png').exists():
        summary.append("![Loss Curves](loss_curve.png)")
        summary.append("")

    if Path('progress.png').exists():
        summary.append("## Training Progress")
        summary.append("")
        summary.append("![Training Progress](progress.png)")
        summary.append("")

    # Add metrics table
    if metrics['train_losses']:
        summary.append("## Key Statistics")
        summary.append("")
        summary.append("| Metric | Value |")
        summary.append("|--------|-------|")
        summary.append(f"| Total Iterations | {len(metrics['train_losses']):,} |")
        summary.append(f"| Final Train Loss | {metrics['train_losses'][-1]:.4f} |")

        if metrics['val_losses']:
            summary.append(f"| Final Val Loss | {metrics['val_losses'][-1]:.4f} |")
            summary.append(f"| Best Val Loss | {min(metrics['val_losses']):.4f} |")

        summary.append("")

    with open(output_file, 'w') as f:
        f.write('\n'.join(summary))

    print(f"✓ Visual summary saved to {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Create visual izations from training log')
    parser.add_argument('--log-file', default='training.log',
                       help='Path to training log file')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for images')

    args = parser.parse_args()

    print("Parsing training log...", file=sys.stderr)
    metrics = parse_training_log(args.log_file)

    if not metrics:
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    success = False
    if create_loss_plot(metrics, output_dir / 'loss_curve.png'):
        success = True

    if create_progress_chart(metrics, output_dir / 'progress.png'):
        success = True

    if success:
        create_summary_with_images(metrics, output_dir / 'visual_summary.md')
        print("\n✅ All visualizations created successfully", file=sys.stderr)
        return 0
    else:
        print("\n⚠️  Could not create visualizations (matplotlib may not be installed)", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
