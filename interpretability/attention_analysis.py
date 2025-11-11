"""
Attention Pattern Analysis and Visualization

This module provides tools for extracting, analyzing, and visualizing attention
patterns in transformer models. Understanding where models "look" is crucial for
mechanistic interpretability.

Key Functions:
    - extract_attention: Get attention patterns from all layers/heads
    - visualize_attention_head: Create heatmaps for single heads
    - plot_all_heads: Grid visualization of all attention heads
    - compute_attention_stats: Calculate metrics (entropy, sparsity, etc.)
    - find_attention_patterns: Detect common patterns (induction, copying, etc.)

References:
    - "Attention is All You Need" (Vaswani et al., 2017)
    - "A Mathematical Framework for Transformer Circuits" (Elhage et al., 2021)
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from .utils import (
    HookManager,
    get_module_by_name,
    to_numpy,
    format_attention_scores
)


@dataclass
class AttentionPattern:
    """
    Container for attention patterns from a model.

    Attributes:
        patterns: Attention weights [n_layers, n_heads, seq_len, seq_len]
        tokens: List of token strings
        layer_names: Names of layers
        metadata: Additional information
    """
    patterns: torch.Tensor
    tokens: List[str]
    layer_names: List[str]
    metadata: Dict

    def __repr__(self) -> str:
        return (f"AttentionPattern(layers={len(self.layer_names)}, "
                f"heads={self.patterns.shape[1]}, "
                f"seq_len={len(self.tokens)})")


def extract_attention(
    model: nn.Module,
    inputs: torch.Tensor,
    tokens: Optional[List[str]] = None
) -> AttentionPattern:
    """
    Extract attention patterns from all layers and heads.

    Args:
        model: The GPT model
        inputs: Input tensor [batch_size, seq_len]
        tokens: Optional list of token strings for visualization

    Returns:
        AttentionPattern object containing all attention weights

    Example:
        >>> attn = extract_attention(model, inputs, tokens)
        >>> # Access layer 2, head 3
        >>> head_pattern = attn.patterns[2, 3]  # [seq_len, seq_len]
    """
    attention_patterns = []
    layer_names = []
    hook_manager = HookManager()

    def make_attention_hook(storage: list):
        def hook(module, input, output):
            # In nanoGPT's CausalSelfAttention, we need to extract attention
            # weights from the forward pass. This requires modifying the
            # module slightly or accessing internal state.
            # For now, we'll store the output and compute attention separately
            pass
        return hook

    # Register hooks on all attention layers
    for layer_idx in range(model.config.n_layer):
        layer_name = f"transformer.h.{layer_idx}.attn"
        layer_names.append(layer_name)

        # Note: nanoGPT's implementation doesn't expose attention weights
        # directly. We'll need to modify the model or use a workaround.
        # For now, we'll return a placeholder that shows the structure.

    model.eval()
    with torch.no_grad():
        # Forward pass
        output = model(inputs)

    hook_manager.remove_all_hooks()

    # Create dummy patterns for now (will be implemented properly)
    batch_size, seq_len = inputs.shape
    n_heads = model.config.n_head
    n_layers = model.config.n_layer

    # Placeholder: uniform attention (will be replaced with actual extraction)
    patterns = torch.ones(n_layers, n_heads, seq_len, seq_len) / seq_len

    # Generate token list if not provided
    if tokens is None:
        tokens = [f"tok_{i}" for i in range(seq_len)]

    return AttentionPattern(
        patterns=patterns,
        tokens=tokens,
        layer_names=layer_names,
        metadata={"model": model.__class__.__name__}
    )


def visualize_attention_head(
    attention_pattern: torch.Tensor,
    tokens: List[str],
    layer_idx: int,
    head_idx: int,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Visualize a single attention head as a heatmap.

    Args:
        attention_pattern: Attention weights [seq_len, seq_len]
        tokens: List of token strings
        layer_idx: Layer index (for title)
        head_idx: Head index (for title)
        title: Custom title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure

    Example:
        >>> attn = extract_attention(model, inputs, tokens)
        >>> fig = visualize_attention_head(
        ...     attn.patterns[2, 3],
        ...     attn.tokens,
        ...     layer_idx=2,
        ...     head_idx=3
        ... )
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy
    pattern_np = to_numpy(attention_pattern)

    # Create heatmap
    sns.heatmap(
        pattern_np,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        cbar=True,
        square=True,
        linewidths=0.5,
        ax=ax
    )

    # Set title
    if title is None:
        title = f"Attention Pattern - Layer {layer_idx}, Head {head_idx}"
    ax.set_title(title, fontsize=14, pad=20)

    ax.set_xlabel("Source Token (Key)", fontsize=12)
    ax.set_ylabel("Target Token (Query)", fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_all_heads(
    attention_patterns: AttentionPattern,
    layer_idx: int,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 15)
) -> plt.Figure:
    """
    Plot all attention heads from a single layer in a grid.

    Args:
        attention_patterns: AttentionPattern object
        layer_idx: Which layer to visualize
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure

    Example:
        >>> attn = extract_attention(model, inputs, tokens)
        >>> fig = plot_all_heads(attn, layer_idx=2)
        >>> plt.show()
    """
    n_heads = attention_patterns.patterns.shape[1]
    layer_patterns = attention_patterns.patterns[layer_idx]

    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(n_heads)))
    n_rows = int(np.ceil(n_heads / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_heads > 1 else [axes]

    for head_idx in range(n_heads):
        ax = axes[head_idx]
        pattern_np = to_numpy(layer_patterns[head_idx])

        sns.heatmap(
            pattern_np,
            cmap="viridis",
            cbar=True,
            square=True,
            ax=ax,
            xticklabels=False,
            yticklabels=False
        )

        ax.set_title(f"Head {head_idx}", fontsize=10)

    # Hide unused subplots
    for idx in range(n_heads, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(
        f"All Attention Heads - Layer {layer_idx}",
        fontsize=16,
        y=0.995
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def compute_attention_stats(
    attention_pattern: torch.Tensor
) -> Dict[str, float]:
    """
    Compute statistics about an attention pattern.

    Args:
        attention_pattern: Attention weights [seq_len, seq_len]

    Returns:
        Dictionary of statistics

    Metrics:
        - entropy: Average entropy of attention distributions (higher = more uniform)
        - sparsity: Fraction of near-zero attention weights
        - max_attention: Maximum attention weight
        - diagonal_strength: How much model attends to same position
        - prev_token_strength: How much model attends to previous token

    Example:
        >>> stats = compute_attention_stats(attn.patterns[2, 3])
        >>> print(f"Entropy: {stats['entropy']:.3f}")
    """
    pattern = attention_pattern

    # Entropy: -sum(p * log(p))
    entropy = -torch.sum(
        pattern * torch.log(pattern + 1e-10),
        dim=-1
    ).mean().item()

    # Sparsity: fraction of weights below threshold
    threshold = 0.01
    sparsity = (pattern < threshold).float().mean().item()

    # Max attention
    max_attention = pattern.max().item()

    # Diagonal strength (attending to same position)
    diagonal = torch.diagonal(pattern, dim1=-2, dim2=-1)
    diagonal_strength = diagonal.mean().item()

    # Previous token strength
    if pattern.shape[-1] > 1:
        prev_token_attn = torch.diagonal(pattern, offset=-1, dim1=-2, dim2=-1)
        prev_token_strength = prev_token_attn.mean().item()
    else:
        prev_token_strength = 0.0

    return {
        "entropy": entropy,
        "sparsity": sparsity,
        "max_attention": max_attention,
        "diagonal_strength": diagonal_strength,
        "prev_token_strength": prev_token_strength
    }


def find_induction_heads(
    attention_patterns: AttentionPattern,
    threshold: float = 0.5
) -> List[Tuple[int, int]]:
    """
    Detect potential induction heads using pattern matching.

    Induction heads attend to tokens that previously appeared after the
    current token, enabling in-context learning.

    Pattern: [A][B]...[A] -> attends to [B]

    Args:
        attention_patterns: AttentionPattern object
        threshold: Minimum attention strength to consider

    Returns:
        List of (layer_idx, head_idx) tuples for potential induction heads

    Example:
        >>> attn = extract_attention(model, inputs, tokens)
        >>> induction_heads = find_induction_heads(attn)
        >>> print(f"Found {len(induction_heads)} potential induction heads")
    """
    induction_heads = []

    n_layers, n_heads, seq_len, _ = attention_patterns.patterns.shape

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            pattern = attention_patterns.patterns[layer_idx, head_idx]

            # Check for characteristic induction pattern
            # This is a simplified heuristic
            # Real detection requires testing on specific sequences

            # Look for offset diagonal pattern (attending to previous occurrence)
            stats = compute_attention_stats(pattern)

            # Heuristic: high prev_token_strength might indicate induction
            if stats['prev_token_strength'] > threshold:
                induction_heads.append((layer_idx, head_idx))

    return induction_heads


def categorize_attention_heads(
    attention_patterns: AttentionPattern
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Categorize attention heads by their behavior patterns.

    Categories:
        - previous_token: Strongly attends to previous token
        - positional: Shows clear positional patterns
        - uniform: Roughly uniform attention
        - sparse: Very sparse attention (few strong connections)
        - induction: Potential induction head behavior

    Args:
        attention_patterns: AttentionPattern object

    Returns:
        Dictionary mapping categories to lists of (layer, head) tuples

    Example:
        >>> attn = extract_attention(model, inputs, tokens)
        >>> categories = categorize_attention_heads(attn)
        >>> print(f"Previous token heads: {categories['previous_token']}")
    """
    categories = {
        "previous_token": [],
        "positional": [],
        "uniform": [],
        "sparse": [],
        "induction": []
    }

    n_layers, n_heads, _, _ = attention_patterns.patterns.shape

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            pattern = attention_patterns.patterns[layer_idx, head_idx]
            stats = compute_attention_stats(pattern)

            # Categorize based on statistics
            if stats['prev_token_strength'] > 0.5:
                categories['previous_token'].append((layer_idx, head_idx))

            if stats['entropy'] < 1.0:
                categories['sparse'].append((layer_idx, head_idx))

            if stats['entropy'] > 3.0:
                categories['uniform'].append((layer_idx, head_idx))

            # Check for positional patterns (simplified)
            if stats['diagonal_strength'] > 0.3:
                categories['positional'].append((layer_idx, head_idx))

    # Find induction heads
    categories['induction'] = find_induction_heads(attention_patterns)

    return categories


# Example usage
if __name__ == "__main__":
    print("Attention Analysis Module")
    print("=" * 50)
    print("\nThis module provides tools for analyzing and")
    print("visualizing attention patterns in transformers.")
    print("\nKey functions:")
    print("  - extract_attention: Get all attention patterns")
    print("  - visualize_attention_head: Plot single head")
    print("  - plot_all_heads: Grid of all heads in layer")
    print("  - compute_attention_stats: Calculate metrics")
    print("  - find_induction_heads: Detect induction heads")
    print("  - categorize_attention_heads: Taxonomize heads by behavior")
