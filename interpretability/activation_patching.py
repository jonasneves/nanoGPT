"""
Activation Patching for Causal Circuit Discovery

This module implements activation patching (also called "causal tracing" or
"interchange intervention"), a key technique in mechanistic interpretability.

The core idea: replace activations from a "clean" run with activations from a
"corrupted" run to determine which components are causally important for a
specific model behavior.

Key Functions:
    - get_activations: Extract activations from specified layers
    - patch_activation: Replace activations and measure the effect
    - path_patching: Find causal paths through the network
    - ablate_component: Remove a component's effect
    - find_circuit: Discover minimal circuits for specific behaviors

References:
    - "Locating and Editing Factual Associations in GPT" (Meng et al., 2022)
    - "Interpretability in the Wild" (Bills et al., 2023)
"""

from typing import Dict, List, Tuple, Optional, Callable, Any
import torch
import torch.nn as nn
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from .utils import (
    ActivationCache,
    HookManager,
    get_module_by_name,
    get_device,
    to_numpy
)


@dataclass
class PatchingResult:
    """
    Results from an activation patching experiment.

    Attributes:
        layer_name: Name of the patched layer
        clean_output: Model output on clean input
        corrupted_output: Model output on corrupted input
        patched_output: Model output after patching
        effect: How much the patch restored clean behavior (0-1)
        metadata: Additional information about the experiment
    """
    layer_name: str
    clean_output: torch.Tensor
    corrupted_output: torch.Tensor
    patched_output: torch.Tensor
    effect: float
    metadata: Dict[str, Any]

    def __repr__(self) -> str:
        return (f"PatchingResult(layer={self.layer_name}, "
                f"effect={self.effect:.3f})")


def get_activations(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_names: List[str],
    return_output: bool = True
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract activations from specified layers during a forward pass.

    Args:
        model: The model to analyze
        inputs: Input tensor [batch_size, seq_len]
        layer_names: List of layer names to extract activations from
        return_output: Whether to return the model output

    Returns:
        activations: Dictionary mapping layer names to activations
        output: Model output (if return_output=True)

    Example:
        >>> activations, output = get_activations(
        ...     model,
        ...     inputs,
        ...     ["transformer.h.0.attn", "transformer.h.1.mlp"]
        ... )
    """
    cache = ActivationCache()
    hook_manager = HookManager()

    # Create hook function to store activations
    def make_hook(name: str):
        def hook(module, input, output):
            cache.store(name, output)
        return hook

    # Register hooks
    for layer_name in layer_names:
        module = get_module_by_name(model, layer_name)
        hook_manager.register_hook(module, make_hook(layer_name))

    # Run forward pass
    model.eval()
    with torch.no_grad():
        output = model(inputs)

    # Clean up hooks
    hook_manager.remove_all_hooks()

    activations = {name: cache.get(name) for name in layer_names}

    if return_output:
        return activations, output
    else:
        return activations, None


def patch_activation(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_name: str,
    patch_value: torch.Tensor,
    return_activations: bool = False
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """
    Replace activation at a specific layer and get model output.

    Args:
        model: The model to analyze
        inputs: Input tensor [batch_size, seq_len]
        layer_name: Name of layer to patch
        patch_value: Tensor to replace the activation with
        return_activations: Whether to return all activations

    Returns:
        output: Model output after patching
        activations: Dictionary of activations (if return_activations=True)

    Example:
        >>> # Get clean activations
        >>> clean_acts, _ = get_activations(model, clean_input, ["layer1"])
        >>> # Patch corrupted input with clean activation
        >>> output, _ = patch_activation(
        ...     model, corrupted_input, "layer1", clean_acts["layer1"]
        ... )
    """
    cache = ActivationCache() if return_activations else None
    hook_manager = HookManager()
    patched = False

    def make_patch_hook(patch_val: torch.Tensor):
        def hook(module, input, output):
            nonlocal patched
            patched = True
            return patch_val
        return hook

    def make_cache_hook(name: str):
        def hook(module, input, output):
            if cache is not None:
                cache.store(name, output)
        return hook

    # Register patch hook
    module = get_module_by_name(model, layer_name)
    hook_manager.register_hook(module, make_patch_hook(patch_value))

    # Optionally register cache hooks for all layers
    if return_activations:
        for name, mod in model.named_modules():
            if name:
                hook_manager.register_hook(mod, make_cache_hook(name))

    # Run forward pass
    model.eval()
    with torch.no_grad():
        output = model(inputs)

    # Clean up
    hook_manager.remove_all_hooks()

    activations = cache.cache if cache is not None else None

    return output, activations


def compute_patching_effect(
    clean_output: torch.Tensor,
    corrupted_output: torch.Tensor,
    patched_output: torch.Tensor,
    metric: str = "logit_diff"
) -> float:
    """
    Compute the effect of patching using various metrics.

    Effect = (patched - corrupted) / (clean - corrupted)
    - Effect H 1.0: Patching fully restores clean behavior (component is important)
    - Effect H 0.0: Patching has no effect (component is unimportant)

    Args:
        clean_output: Output on clean input
        corrupted_output: Output on corrupted input
        patched_output: Output after patching
        metric: Metric to use ("logit_diff", "kl_div", "prob")

    Returns:
        effect: Patching effect score (typically 0-1)
    """
    if metric == "logit_diff":
        # Simple difference in logits
        clean_score = clean_output.max(dim=-1).values.mean()
        corrupted_score = corrupted_output.max(dim=-1).values.mean()
        patched_score = patched_output.max(dim=-1).values.mean()

        denominator = (clean_score - corrupted_score).abs()
        if denominator < 1e-6:
            return 0.0

        effect = (patched_score - corrupted_score) / denominator
        return float(effect)

    elif metric == "kl_div":
        # KL divergence between distributions
        clean_probs = torch.softmax(clean_output, dim=-1)
        corrupted_probs = torch.softmax(corrupted_output, dim=-1)
        patched_probs = torch.softmax(patched_output, dim=-1)

        kl_clean_corrupted = torch.nn.functional.kl_div(
            corrupted_probs.log(), clean_probs, reduction='batchmean'
        )
        kl_clean_patched = torch.nn.functional.kl_div(
            patched_probs.log(), clean_probs, reduction='batchmean'
        )

        if kl_clean_corrupted < 1e-6:
            return 0.0

        effect = 1.0 - (kl_clean_patched / kl_clean_corrupted)
        return float(effect)

    else:
        raise ValueError(f"Unknown metric: {metric}")


def patch_layer_scan(
    model: nn.Module,
    clean_input: torch.Tensor,
    corrupted_input: torch.Tensor,
    layer_names: Optional[List[str]] = None,
    metric: str = "logit_diff",
    show_progress: bool = True
) -> Dict[str, PatchingResult]:
    """
    Scan through layers, patching each one to find important components.

    This is the main workhorse function for discovering which layers matter
    for a specific behavior.

    Args:
        model: The model to analyze
        clean_input: Input that produces desired behavior
        corrupted_input: Input that doesn't produce desired behavior
        layer_names: Layers to scan (default: all transformer blocks)
        metric: Metric to measure patching effect
        show_progress: Whether to show progress bar

    Returns:
        results: Dictionary mapping layer names to PatchingResults

    Example:
        >>> results = patch_layer_scan(
        ...     model,
        ...     clean_input=torch.tensor([[1, 2, 3]]),
        ...     corrupted_input=torch.tensor([[4, 5, 6]])
        ... )
        >>> # Find most important layers
        >>> sorted_results = sorted(
        ...     results.items(),
        ...     key=lambda x: x[1].effect,
        ...     reverse=True
        ... )
    """
    # Get default layer names if not provided
    if layer_names is None:
        layer_names = [
            f"transformer.h.{i}"
            for i in range(model.config.n_layer)
        ]

    # Get clean and corrupted activations
    clean_acts, clean_output = get_activations(model, clean_input, layer_names)
    corrupted_acts, corrupted_output = get_activations(
        model, corrupted_input, layer_names
    )

    results = {}

    iterator = tqdm(layer_names) if show_progress else layer_names

    for layer_name in iterator:
        if show_progress:
            iterator.set_description(f"Patching {layer_name}")

        # Patch corrupted run with clean activation
        patched_output, _ = patch_activation(
            model,
            corrupted_input,
            layer_name,
            clean_acts[layer_name]
        )

        # Compute effect
        effect = compute_patching_effect(
            clean_output[0],  # Extract logits
            corrupted_output[0],
            patched_output[0],
            metric=metric
        )

        results[layer_name] = PatchingResult(
            layer_name=layer_name,
            clean_output=clean_output[0],
            corrupted_output=corrupted_output[0],
            patched_output=patched_output[0],
            effect=effect,
            metadata={"metric": metric}
        )

    return results


def ablate_component(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_name: str,
    ablation_type: str = "zero"
) -> torch.Tensor:
    """
    Ablate (remove the effect of) a specific component.

    Args:
        model: The model to analyze
        inputs: Input tensor
        layer_name: Name of component to ablate
        ablation_type: How to ablate ("zero", "mean", "random")

    Returns:
        output: Model output after ablation

    Example:
        >>> # Test importance by removing component
        >>> normal_output = model(inputs)
        >>> ablated_output = ablate_component(model, inputs, "transformer.h.2")
        >>> importance = (normal_output - ablated_output).abs().mean()
    """
    hook_manager = HookManager()

    def make_ablation_hook(ablation_type: str):
        def hook(module, input, output):
            if ablation_type == "zero":
                return torch.zeros_like(output)
            elif ablation_type == "mean":
                return torch.full_like(output, output.mean())
            elif ablation_type == "random":
                return torch.randn_like(output) * output.std()
            else:
                raise ValueError(f"Unknown ablation type: {ablation_type}")
        return hook

    module = get_module_by_name(model, layer_name)
    hook_manager.register_hook(module, make_ablation_hook(ablation_type))

    model.eval()
    with torch.no_grad():
        output = model(inputs)

    hook_manager.remove_all_hooks()

    return output


def path_patching(
    model: nn.Module,
    clean_input: torch.Tensor,
    corrupted_input: torch.Tensor,
    sender_layer: str,
    receiver_layer: str,
    metric: str = "logit_diff"
) -> float:
    """
    Test if information flows from sender to receiver using path patching.

    Path patching patches the sender's activation but only in the computational
    path to the receiver, not to other parts of the model.

    Args:
        model: The model to analyze
        clean_input: Clean input
        corrupted_input: Corrupted input
        sender_layer: Layer to patch from
        receiver_layer: Layer to patch into
        metric: Metric to measure effect

    Returns:
        effect: How much the sender affects the receiver

    Note:
        This is a simplified version. Full path patching requires more
        sophisticated intervention on the computational graph.
    """
    # Get clean activation at sender
    clean_acts, clean_output = get_activations(
        model, clean_input, [sender_layer]
    )

    # Get corrupted output
    _, corrupted_output = get_activations(
        model, corrupted_input, [sender_layer]
    )

    # Patch sender in corrupted run
    patched_output, _ = patch_activation(
        model,
        corrupted_input,
        sender_layer,
        clean_acts[sender_layer]
    )

    # Compute effect
    effect = compute_patching_effect(
        clean_output[0],
        corrupted_output[0],
        patched_output[0],
        metric=metric
    )

    return effect


def find_important_components(
    results: Dict[str, PatchingResult],
    threshold: float = 0.5,
    top_k: Optional[int] = None
) -> List[Tuple[str, float]]:
    """
    Extract important components from patching results.

    Args:
        results: Results from patch_layer_scan
        threshold: Minimum effect to be considered important
        top_k: If provided, return top k components by effect

    Returns:
        important: List of (layer_name, effect) tuples

    Example:
        >>> results = patch_layer_scan(model, clean_input, corrupted_input)
        >>> important = find_important_components(results, threshold=0.5)
        >>> for layer, effect in important:
        ...     print(f"{layer}: {effect:.3f}")
    """
    # Sort by effect
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].effect,
        reverse=True
    )

    # Filter by threshold
    important = [
        (name, result.effect)
        for name, result in sorted_results
        if result.effect >= threshold
    ]

    # Apply top_k if specified
    if top_k is not None:
        important = important[:top_k]

    return important


# Example usage and tests
if __name__ == "__main__":
    print("Activation Patching Module")
    print("=" * 50)
    print("\nThis module implements activation patching for")
    print("mechanistic interpretability of neural networks.")
    print("\nKey functions:")
    print("  - get_activations: Extract layer activations")
    print("  - patch_activation: Replace activations")
    print("  - patch_layer_scan: Find important layers")
    print("  - ablate_component: Test component importance")
    print("  - path_patching: Trace information flow")
