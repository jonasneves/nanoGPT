"""
Utility functions for interpretability analysis.

This module provides shared utilities used across all interpretability tools,
including hook management, activation storage, tokenization helpers, and
visualization utilities.
"""

from typing import Dict, List, Tuple, Callable, Optional, Any
import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np


class ActivationCache:
    """
    Cache for storing model activations during forward passes.

    This class provides a convenient way to store and retrieve activations
    from specific layers and components during model inference.

    Attributes:
        cache: Dictionary mapping layer names to their activations
    """

    def __init__(self):
        self.cache: Dict[str, torch.Tensor] = {}

    def store(self, name: str, activation: torch.Tensor) -> None:
        """Store an activation tensor."""
        self.cache[name] = activation.detach().clone()

    def get(self, name: str) -> Optional[torch.Tensor]:
        """Retrieve a stored activation."""
        return self.cache.get(name)

    def clear(self) -> None:
        """Clear all stored activations."""
        self.cache.clear()

    def keys(self) -> List[str]:
        """Get all stored activation names."""
        return list(self.cache.keys())

    def __repr__(self) -> str:
        return f"ActivationCache(n_stored={len(self.cache)})"


class HookManager:
    """
    Manager for PyTorch forward hooks.

    This class simplifies the process of registering, managing, and removing
    forward hooks on model layers. It's essential for intercepting and
    modifying activations during model execution.

    Example:
        >>> manager = HookManager()
        >>> cache = ActivationCache()
        >>> hook_fn = lambda module, input, output: cache.store("layer1", output)
        >>> manager.register_hook(model.layer1, hook_fn)
        >>> # ... run model forward pass ...
        >>> manager.remove_all_hooks()
    """

    def __init__(self):
        self.hooks = []

    def register_hook(
        self,
        module: nn.Module,
        hook_fn: Callable,
        hook_type: str = "forward"
    ) -> None:
        """
        Register a hook on a module.

        Args:
            module: The PyTorch module to attach the hook to
            hook_fn: The hook function to call
            hook_type: Type of hook ("forward" or "backward")
        """
        if hook_type == "forward":
            handle = module.register_forward_hook(hook_fn)
        elif hook_type == "backward":
            handle = module.register_backward_hook(hook_fn)
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")

        self.hooks.append(handle)

    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_all_hooks()

    def __repr__(self) -> str:
        return f"HookManager(n_hooks={len(self.hooks)})"


def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    """
    Get a module from a model by its name.

    Args:
        model: The model to search
        module_name: Dot-separated path to the module (e.g., "transformer.h.0.attn")

    Returns:
        The requested module

    Example:
        >>> attn_module = get_module_by_name(model, "transformer.h.0.attn")
    """
    parts = module_name.split('.')
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def get_device(model: nn.Module) -> torch.device:
    """Get the device a model is on."""
    return next(model.parameters()).device


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def to_tensor(array: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert numpy array to tensor."""
    tensor = torch.from_numpy(array)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def get_tokens_with_context(
    tokens: List[str],
    position: int,
    context_size: int = 5
) -> str:
    """
    Get a token with surrounding context for visualization.

    Args:
        tokens: List of token strings
        position: Position of the target token
        context_size: Number of tokens to show on each side

    Returns:
        String showing the token in context
    """
    start = max(0, position - context_size)
    end = min(len(tokens), position + context_size + 1)

    context_tokens = tokens[start:end]
    target_idx = position - start

    # Highlight the target token
    context_tokens[target_idx] = f"**{context_tokens[target_idx]}**"

    return " ".join(context_tokens)


def batch_data(data: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Batch a list of data into chunks.

    Args:
        data: List of items to batch
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def format_attention_scores(
    attention_scores: torch.Tensor,
    src_tokens: List[str],
    tgt_tokens: List[str],
    threshold: float = 0.1
) -> str:
    """
    Format attention scores as a readable string.

    Args:
        attention_scores: Attention weight matrix [seq_len, seq_len]
        src_tokens: Source tokens
        tgt_tokens: Target tokens
        threshold: Minimum attention weight to display

    Returns:
        Formatted string representation
    """
    lines = []
    for i, tgt_token in enumerate(tgt_tokens):
        attended = []
        for j, src_token in enumerate(src_tokens):
            score = attention_scores[i, j].item()
            if score >= threshold:
                attended.append(f"{src_token}({score:.2f})")
        if attended:
            lines.append(f"{tgt_token} <- {', '.join(attended)}")
    return "\n".join(lines)


def get_layer_names(model: nn.Module, layer_type: Optional[type] = None) -> List[str]:
    """
    Get names of all layers in a model, optionally filtered by type.

    Args:
        model: The model to inspect
        layer_type: Optional layer type to filter by (e.g., nn.Linear)

    Returns:
        List of layer names
    """
    layer_names = []
    for name, module in model.named_modules():
        if layer_type is None or isinstance(module, layer_type):
            if name:  # Skip empty names (root module)
                layer_names.append(name)
    return layer_names


def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: The model to count parameters for
        only_trainable: If True, only count trainable parameters

    Returns:
        Number of parameters
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_info(model: nn.Module) -> None:
    """Print useful information about a model."""
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model, only_trainable=True):,}")
    print(f"Device: {get_device(model)}")

    # Count layer types
    layer_counts = defaultdict(int)
    for module in model.modules():
        layer_counts[module.__class__.__name__] += 1

    print("\nLayer composition:")
    for layer_type, count in sorted(layer_counts.items()):
        if count > 1:  # Only show layers that appear multiple times
            print(f"  {layer_type}: {count}")


class dotdict(dict):
    """
    Dictionary that supports dot notation access.

    Example:
        >>> d = dotdict({'a': 1, 'b': 2})
        >>> d.a  # Returns 1
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
