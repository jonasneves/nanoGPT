#!/usr/bin/env python3
"""
Generate samples during training at regular intervals.
Monitors the checkpoint file and generates samples when updated.
"""

import os
import sys
import time
import re
import subprocess
from pathlib import Path
from datetime import datetime


class IntermediateSampler:
    def __init__(self, log_file, checkpoint_dir, output_file, interval=500):
        self.log_file = Path(log_file)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_file = Path(output_file)
        self.interval = interval
        self.last_sampled_iter = -1
        self.samples_generated = []
        self.last_position = 0

    def get_current_iteration(self):
        """Get the current training iteration from log."""
        if not self.log_file.exists():
            return None

        try:
            with open(self.log_file, 'r') as f:
                f.seek(self.last_position)
                lines = f.readlines()
                self.last_position = f.tell()

                # Find latest iteration
                for line in reversed(lines):
                    match = re.search(r'iter (\d+):', line)
                    if match:
                        return int(match.group(1))
        except Exception as e:
            print(f"[Sampler] Error reading log: {e}", file=sys.stderr)

        return None

    def checkpoint_exists(self):
        """Check if checkpoint file exists."""
        checkpoint_path = self.checkpoint_dir / "ckpt.pt"
        return checkpoint_path.exists()

    def generate_sample(self, iteration):
        """Generate a sample from the current checkpoint."""
        if not self.checkpoint_exists():
            print(f"[Sampler] Checkpoint not found, skipping sample at iter {iteration}", file=sys.stderr)
            return None

        prompts = [
            "Breaking news: ",
            "In a recent study, ",
            "The future of technology "
        ]

        samples = []
        for i, prompt in enumerate(prompts):
            try:
                print(f"[Sampler] Generating sample {i+1}/3 at iteration {iteration}...", file=sys.stderr)

                result = subprocess.run(
                    [
                        'python', 'sample.py',
                        f'--out_dir={self.checkpoint_dir}',
                        f'--start={prompt}',
                        '--num_samples=1',
                        '--max_new_tokens=100'
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode == 0:
                    sample_text = result.stdout.strip()
                    samples.append({
                        'prompt': prompt,
                        'text': sample_text,
                        'iteration': iteration
                    })
                else:
                    print(f"[Sampler] Sample generation failed: {result.stderr}", file=sys.stderr)

            except subprocess.TimeoutExpired:
                print(f"[Sampler] Sample generation timeout", file=sys.stderr)
            except Exception as e:
                print(f"[Sampler] Error generating sample: {e}", file=sys.stderr)

        return samples if samples else None

    def update_output(self):
        """Update the output file with all samples."""
        output = [
            "# 🎨 Intermediate Sample Generations",
            "",
            f"Generated during training to show model evolution",
            f"**Total samples:** {len(self.samples_generated)} checkpoints",
            "",
        ]

        if self.samples_generated:
            # Show only the last 3 sample sets to avoid overflow
            recent_samples = self.samples_generated[-3:]

            for sample_set in recent_samples:
                iteration = sample_set[0]['iteration']
                output.append(f"## Iteration {iteration:,}")
                output.append("")

                for i, sample in enumerate(sample_set, 1):
                    output.append(f"**Prompt {i}:** `{sample['prompt']}`")
                    output.append("")
                    output.append("```")
                    # Truncate long samples
                    text = sample['text'][:300] + "..." if len(sample['text']) > 300 else sample['text']
                    output.append(text)
                    output.append("```")
                    output.append("")

            output.append("---")
            output.append(f"*Last updated: {datetime.now().strftime('%H:%M:%S UTC')}*")

        else:
            output.append("*Waiting for first checkpoint...*")

        # Write to file
        with open(self.output_file, 'w') as f:
            f.write('\n'.join(output))

    def monitor(self, max_iter=2000):
        """Main monitoring loop."""
        print(f"[Sampler] Starting intermediate sampler...", file=sys.stderr)
        print(f"[Sampler] Sample interval: every {self.interval} iterations", file=sys.stderr)
        print(f"[Sampler] Checkpoint dir: {self.checkpoint_dir}", file=sys.stderr)

        # Initial output
        self.update_output()

        while True:
            current_iter = self.get_current_iteration()

            if current_iter is not None:
                # Check if we should generate a sample
                if current_iter >= (self.last_sampled_iter + self.interval):
                    # Only sample if checkpoint exists and is recent
                    if self.checkpoint_exists():
                        print(f"[Sampler] Generating samples at iteration {current_iter}...", file=sys.stderr)
                        samples = self.generate_sample(current_iter)

                        if samples:
                            self.samples_generated.append(samples)
                            self.last_sampled_iter = current_iter
                            self.update_output()
                            print(f"[Sampler] ✓ Samples generated and saved", file=sys.stderr)

                # Check if training is complete
                if current_iter >= max_iter:
                    print(f"[Sampler] Training complete!", file=sys.stderr)
                    return

            time.sleep(30)  # Check every 30 seconds


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate samples during training')
    parser.add_argument('--log-file', default='training.log',
                       help='Path to training log file')
    parser.add_argument('--checkpoint-dir', default='out-custom',
                       help='Directory containing model checkpoints')
    parser.add_argument('--output-file', default='intermediate_samples.md',
                       help='Output file for samples')
    parser.add_argument('--interval', type=int, default=500,
                       help='Generate samples every N iterations')
    parser.add_argument('--max-iter', type=int, default=2000,
                       help='Maximum training iterations')

    args = parser.parse_args()

    sampler = IntermediateSampler(
        args.log_file,
        args.checkpoint_dir,
        args.output_file,
        args.interval
    )

    try:
        sampler.monitor(max_iter=args.max_iter)
    except KeyboardInterrupt:
        print("\n[Sampler] Stopped by user", file=sys.stderr)
        sys.exit(0)


if __name__ == '__main__':
    main()
