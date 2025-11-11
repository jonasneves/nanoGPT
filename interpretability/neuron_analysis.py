"""
Neuron Analysis

Tools for analyzing individual neurons and understanding what features they
represent. This module helps discover what individual model components "detect"
or "compute".

Key Functions:
    - get_neuron_activations: Extract activation values for specific neurons
    - find_max_activating_examples: Find inputs that maximally activate neurons
    - ablate_neuron: Test importance by removing neuron
    - visualize_neuron_behavior: Plot activation distributions
    - find_feature_direction: Identify feature directions in activation space

References:
    - "Zoom In: An Introduction to Circuits" (Olah et al., 2020)
    - "Toy Models of Superposition" (Elhage et al., 2022)
"""

from typing import Dict, List, Tuple, Optional, Callable
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from .utils import (
    HookManager,
    ActivationCache,
    get_module_by_name,
    to_numpy
)


@dataclass
class NeuronActivation:
    """
    Container for neuron activation data.

    Attributes:
        neuron_id: (layer_idx, neuron_idx) identifying the neuron
        activations: Activation values across dataset
        max_examples: Top examples that activate this neuron
        statistics: Mean, std, etc.
    """
    neuron_id: Tuple[int, int]
    activations: torch.Tensor
    max_examples: List[Tuple[str, float]]
    statistics: Dict[str, float]

    def __repr__(self) -> str:
        layer, neuron = self.neuron_id
        return (f"NeuronActivation(layer={layer}, neuron={neuron}, "
                f"mean={self.statistics.get('mean', 0):.3f})")


def get_neuron_activations(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_idx: int,
    neuron_idx: Optional[int] = None
) -> torch.Tensor:
    """
    Extract activation values for specific neurons.

    Args:
        model: The GPT model
        inputs: Input tensor [batch_size, seq_len]
        layer_idx: Which layer to analyze
        neuron_idx: Specific neuron (None = all neurons)

    Returns:
        Activations tensor [batch_size, seq_len, n_neurons] or
        [batch_size, seq_len] if neuron_idx specified

    Example:
        >>> # Get all MLP neuron activations from layer 2
        >>> acts = get_neuron_activations(model, inputs, layer_idx=2)
        >>> # Get specific neuron
        >>> neuron_acts = get_neuron_activations(model, inputs, 2, neuron_idx=42)
    """
    cache = ActivationCache()
    hook_manager = HookManager()

    # Hook the MLP output
    layer_name = f"transformer.h.{layer_idx}.mlp"
    module = get_module_by_name(model, layer_name)

    def hook(module, input, output):
        cache.store("mlp_output", output)

    hook_manager.register_hook(module, hook)

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(inputs)

    hook_manager.remove_all_hooks()

    activations = cache.get("mlp_output")

    if neuron_idx is not None:
        activations = activations[..., neuron_idx]

    return activations


def find_max_activating_examples(
    model: nn.Module,
    dataset: List[torch.Tensor],
    layer_idx: int,
    neuron_idx: int,
    top_k: int = 10,
    tokens_list: Optional[List[List[str]]] = None
) -> List[Tuple[int, int, float]]:
    """
    Find dataset examples that maximally activate a neuron.

    Args:
        model: The GPT model
        dataset: List of input tensors
        layer_idx: Layer index
        neuron_idx: Neuron index
        top_k: Number of top examples to return
        tokens_list: Optional list of token strings for each example

    Returns:
        List of (example_idx, position, activation_value) tuples

    Example:
        >>> max_examples = find_max_activating_examples(
        ...     model, dataset, layer_idx=2, neuron_idx=42, top_k=10
        ... )
        >>> print(f"Neuron 42 most activated by: {max_examples[0]}")
    """
    all_activations = []

    for example_idx, inputs in enumerate(dataset):
        acts = get_neuron_activations(
            model, inputs.unsqueeze(0), layer_idx, neuron_idx
        )

        # Get max activation for each position
        for pos in range(acts.shape[1]):
            activation_value = acts[0, pos].item()
            all_activations.append((example_idx, pos, activation_value))

    # Sort by activation value
    all_activations.sort(key=lambda x: x[2], reverse=True)

    return all_activations[:top_k]


def ablate_neuron(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_idx: int,
    neuron_idx: int,
    ablation_value: float = 0.0
) -> torch.Tensor:
    """
    Test neuron importance by ablating (zeroing or setting to mean).

    Args:
        model: The GPT model
        inputs: Input tensor
        layer_idx: Layer index
        neuron_idx: Neuron to ablate
        ablation_value: Value to set neuron to (default: 0.0)

    Returns:
        Model output after ablation

    Example:
        >>> normal_output = model(inputs)
        >>> ablated_output = ablate_neuron(model, inputs, layer=2, neuron=42)
        >>> importance = (normal_output - ablated_output).abs().mean()
    """
    hook_manager = HookManager()

    layer_name = f"transformer.h.{layer_idx}.mlp"
    module = get_module_by_name(model, layer_name)

    def ablation_hook(module, input, output):
        # Set specific neuron to ablation value
        output_copy = output.clone()
        output_copy[..., neuron_idx] = ablation_value
        return output_copy

    hook_manager.register_hook(module, ablation_hook)

    model.eval()
    with torch.no_grad():
        output = model(inputs)

    hook_manager.remove_all_hooks()

    return output


def visualize_neuron_activations(
    activations: torch.Tensor,
    layer_idx: int,
    neuron_idx: int,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Visualize distribution of neuron activations.

    Args:
        activations: Activation tensor
        layer_idx: Layer index (for title)
        neuron_idx: Neuron index (for title)
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Convert to numpy
    acts_np = to_numpy(activations).flatten()

    # Histogram
    axes[0].hist(acts_np, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Activation Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Activation Distribution - L{layer_idx}N{neuron_idx}")
    axes[0].axvline(acts_np.mean(), color='red', linestyle='--',
                    label=f'Mean: {acts_np.mean():.3f}')
    axes[0].legend()

    # Box plot
    axes[1].boxplot(acts_np, vert=True)
    axes[1].set_ylabel("Activation Value")
    axes[1].set_title(f"Activation Statistics - L{layer_idx}N{neuron_idx}")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def compute_neuron_statistics(activations: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics about neuron activations.

    Args:
        activations: Activation tensor

    Returns:
        Dictionary of statistics

    Example:
        >>> acts = get_neuron_activations(model, inputs, layer_idx=2, neuron_idx=42)
        >>> stats = compute_neuron_statistics(acts)
        >>> print(f"Mean: {stats['mean']:.3f}, Sparsity: {stats['sparsity']:.2%}")
    """
    acts_flat = activations.flatten()

    return {
        "mean": acts_flat.mean().item(),
        "std": acts_flat.std().item(),
        "min": acts_flat.min().item(),
        "max": acts_flat.max().item(),
        "sparsity": (acts_flat.abs() < 0.1).float().mean().item(),
        "positive_fraction": (acts_flat > 0).float().mean().item()
    }


# Placeholder for future advanced features
def find_feature_directions(
    model: nn.Module,
    dataset: List[torch.Tensor],
    layer_idx: int,
    method: str = "pca"
) -> torch.Tensor:
    """
    Find important feature directions in activation space.

    This is a placeholder for future implementation of techniques like:
    - PCA to find principal components
    - Sparse autoencoders to find interpretable features
    - Probing to find task-relevant directions

    Args:
        model: The GPT model
        dataset: Dataset to analyze
        layer_idx: Which layer to analyze
        method: Method to use ("pca", "sparse_autoencoder")

    Returns:
        Feature direction vectors
    """
    # Placeholder implementation
    raise NotImplementedError(
        "Feature direction finding will be implemented in future versions. "
        "This will include PCA, sparse autoencoders, and probing methods."
    )


# Example usage
if __name__ == "__main__":
    print("Neuron Analysis Module")
    print("=" * 50)
    print("\nThis module provides tools for analyzing individual")
    print("neurons and understanding what features they detect.")
    print("\nKey functions:")
    print("  - get_neuron_activations: Extract neuron activations")
    print("  - find_max_activating_examples: Find what activates neurons")
    print("  - ablate_neuron: Test neuron importance")
    print("  - visualize_neuron_activations: Plot activation distributions")
    print("  - compute_neuron_statistics: Calculate metrics")
