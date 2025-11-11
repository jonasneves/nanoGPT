"""
Logit Lens Analysis

The logit lens technique applies the final unembedding matrix at intermediate
layers to see what the model "thinks" at each stage of computation. This reveals
how predictions form and evolve throughout the forward pass.

Key insight: Transformer residual streams can be "decoded" at any layer to see
partial predictions, showing when the model becomes confident about its output.

Key Functions:
    - logit_lens: Extract predictions at each layer
    - plot_prediction_evolution: Visualize how top predictions change
    - measure_convergence: Determine when model becomes confident
    - compare_tokens: Compare prediction formation for different inputs

References:
    - "Interpreting GPT: The Logit Lens" (nostalgebraist, 2020)
    - "Eliciting Latent Predictions from Transformers" (Belrose et al., 2023)
"""

from typing import Dict, List, Tuple, Optional
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
class LogitLensResult:
    """
    Results from logit lens analysis.

    Attributes:
        logits_by_layer: List of logits for each layer [n_layers, vocab_size]
        predictions_by_layer: Top-k predictions per layer
        probabilities_by_layer: Probabilities for top predictions
        tokens: Input tokens
        target_position: Which position was analyzed
        metadata: Additional information
    """
    logits_by_layer: List[torch.Tensor]
    predictions_by_layer: List[List[Tuple[str, float]]]
    probabilities_by_layer: List[torch.Tensor]
    tokens: List[str]
    target_position: int
    metadata: Dict

    def __repr__(self) -> str:
        return (f"LogitLensResult(n_layers={len(self.logits_by_layer)}, "
                f"target_pos={self.target_position})")


def logit_lens(
    model: nn.Module,
    inputs: torch.Tensor,
    target_position: int = -1,
    top_k: int = 10,
    layer_range: Optional[Tuple[int, int]] = None
) -> LogitLensResult:
    """
    Apply logit lens to see predictions at each layer.

    Args:
        model: The GPT model
        inputs: Input tensor [batch_size, seq_len]
        target_position: Position to analyze (-1 for last token)
        top_k: Number of top predictions to track
        layer_range: Optional (start, end) layer range to analyze

    Returns:
        LogitLensResult containing predictions at each layer

    Example:
        >>> result = logit_lens(model, inputs, target_position=-1, top_k=5)
        >>> # See how top prediction evolves
        >>> for layer, preds in enumerate(result.predictions_by_layer):
        ...     top_pred, prob = preds[0]
        ...     print(f"Layer {layer}: {top_pred} ({prob:.2%})")
    """
    # Determine layer range
    if layer_range is None:
        start_layer, end_layer = 0, model.config.n_layer
    else:
        start_layer, end_layer = layer_range

    # Storage for results
    logits_by_layer = []
    predictions_by_layer = []
    probabilities_by_layer = []

    cache = ActivationCache()
    hook_manager = HookManager()

    # Get the unembedding matrix (lm_head)
    lm_head = model.lm_head

    # Create hooks to capture layer outputs
    for layer_idx in range(start_layer, end_layer):
        layer_name = f"transformer.h.{layer_idx}"
        module = get_module_by_name(model, layer_name)

        def make_hook(idx):
            def hook(module, input, output):
                cache.store(f"layer_{idx}", output)
            return hook

        hook_manager.register_hook(module, make_hook(layer_idx))

    # Forward pass
    model.eval()
    with torch.no_grad():
        # Run model
        final_output = model(inputs)

        # Also capture final layer norm output
        final_hidden = model.transformer.ln_f(
            get_module_by_name(model, f"transformer.h.{model.config.n_layer - 1}")(
                inputs
            )[0] if hasattr(model, 'transformer') else inputs
        )

    hook_manager.remove_all_hooks()

    # Get tokenizer if available (for decoding predictions)
    # For now, we'll just use token IDs
    vocab_size = model.config.vocab_size

    # Apply logit lens at each layer
    for layer_idx in range(start_layer, end_layer):
        # Get hidden state at this layer
        hidden_state = cache.get(f"layer_{layer_idx}")

        if hidden_state is None:
            continue

        # Apply layer norm (important for getting good predictions)
        # Note: This is a simplification. Ideally we'd apply the exact
        # layer norm that would be applied at this point
        normed_hidden = model.transformer.ln_f(hidden_state)

        # Get logits for target position
        if target_position == -1:
            pos = hidden_state.shape[1] - 1
        else:
            pos = target_position

        hidden_at_pos = normed_hidden[0, pos, :]  # [hidden_size]

        # Apply unembedding
        logits = lm_head(hidden_at_pos.unsqueeze(0))  # [1, vocab_size]
        logits = logits.squeeze(0)  # [vocab_size]

        # Get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k)

        # Store results
        logits_by_layer.append(logits)
        probabilities_by_layer.append(top_probs)

        # Convert to token predictions (for now just use indices)
        predictions = [
            (f"token_{idx.item()}", prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
        predictions_by_layer.append(predictions)

    # Create result object
    result = LogitLensResult(
        logits_by_layer=logits_by_layer,
        predictions_by_layer=predictions_by_layer,
        probabilities_by_layer=probabilities_by_layer,
        tokens=[f"tok_{i}" for i in range(inputs.shape[1])],
        target_position=target_position,
        metadata={
            "model": model.__class__.__name__,
            "n_layers": end_layer - start_layer,
            "vocab_size": vocab_size
        }
    )

    return result


def plot_prediction_evolution(
    result: LogitLensResult,
    num_predictions: int = 5,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Visualize how top predictions evolve across layers.

    Args:
        result: LogitLensResult from logit_lens()
        num_predictions: Number of top predictions to show
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure

    Example:
        >>> result = logit_lens(model, inputs)
        >>> fig = plot_prediction_evolution(result, num_predictions=5)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_layers = len(result.predictions_by_layer)
    layers = list(range(n_layers))

    # Track probability of each prediction across layers
    prediction_tracks = {}

    for layer_idx, predictions in enumerate(result.predictions_by_layer):
        for token, prob in predictions[:num_predictions]:
            if token not in prediction_tracks:
                prediction_tracks[token] = [0.0] * n_layers
            prediction_tracks[token][layer_idx] = prob

    # Plot each prediction's evolution
    for token, probs in prediction_tracks.items():
        ax.plot(layers, probs, marker='o', label=token, linewidth=2)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("Prediction Evolution Across Layers (Logit Lens)", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_probability_heatmap(
    result: LogitLensResult,
    num_predictions: int = 10,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a heatmap showing top predictions at each layer.

    Args:
        result: LogitLensResult from logit_lens()
        num_predictions: Number of predictions to show per layer
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    import seaborn as sns

    n_layers = len(result.predictions_by_layer)

    # Build matrix of probabilities
    # Rows: layers, Columns: top predictions
    prob_matrix = []
    prediction_labels = []

    for layer_idx, predictions in enumerate(result.predictions_by_layer):
        layer_probs = [prob for _, prob in predictions[:num_predictions]]

        # Pad if necessary
        while len(layer_probs) < num_predictions:
            layer_probs.append(0.0)

        prob_matrix.append(layer_probs)

        # Collect unique predictions for labels
        if layer_idx == 0:
            prediction_labels = [
                token for token, _ in predictions[:num_predictions]
            ]

    prob_matrix = np.array(prob_matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        prob_matrix.T,  # Transpose so predictions are rows
        yticklabels=prediction_labels,
        xticklabels=[f"L{i}" for i in range(n_layers)],
        cmap="YlOrRd",
        cbar_kws={'label': 'Probability'},
        annot=False,
        fmt='.2f',
        ax=ax
    )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Prediction", fontsize=12)
    ax.set_title("Top Predictions by Layer (Logit Lens)", fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def measure_convergence(result: LogitLensResult, threshold: float = 0.5) -> Dict:
    """
    Measure when the model's prediction converges (becomes confident).

    Args:
        result: LogitLensResult from logit_lens()
        threshold: Probability threshold for "convergence"

    Returns:
        Dictionary with convergence information

    Example:
        >>> result = logit_lens(model, inputs)
        >>> convergence = measure_convergence(result, threshold=0.5)
        >>> print(f"Model confident by layer {convergence['convergence_layer']}")
    """
    convergence_layer = None
    final_prediction = None

    # Find first layer where top prediction exceeds threshold
    for layer_idx, predictions in enumerate(result.predictions_by_layer):
        if len(predictions) > 0:
            top_token, top_prob = predictions[0]

            if convergence_layer is None and top_prob >= threshold:
                convergence_layer = layer_idx
                final_prediction = top_token

    # Calculate stability (how much top prediction changes)
    prediction_changes = 0
    prev_top = None

    for predictions in result.predictions_by_layer:
        if len(predictions) > 0:
            curr_top = predictions[0][0]
            if prev_top is not None and curr_top != prev_top:
                prediction_changes += 1
            prev_top = curr_top

    return {
        "convergence_layer": convergence_layer,
        "convergence_prediction": final_prediction,
        "total_layers": len(result.predictions_by_layer),
        "prediction_changes": prediction_changes,
        "stability": 1.0 - (prediction_changes / len(result.predictions_by_layer))
    }


def compare_positions(
    model: nn.Module,
    inputs: torch.Tensor,
    positions: List[int],
    top_k: int = 5
) -> Dict[int, LogitLensResult]:
    """
    Compare logit lens results across multiple positions.

    Args:
        model: The GPT model
        inputs: Input tensor
        positions: List of positions to analyze
        top_k: Number of top predictions to track

    Returns:
        Dictionary mapping positions to LogitLensResults

    Example:
        >>> results = compare_positions(model, inputs, positions=[5, 10, 15])
        >>> for pos, result in results.items():
        ...     conv = measure_convergence(result)
        ...     print(f"Position {pos}: converges at layer {conv['convergence_layer']}")
    """
    results = {}

    for pos in positions:
        result = logit_lens(model, inputs, target_position=pos, top_k=top_k)
        results[pos] = result

    return results


# Example usage
if __name__ == "__main__":
    print("Logit Lens Module")
    print("=" * 50)
    print("\nThis module implements the logit lens technique for")
    print("tracking how predictions form across transformer layers.")
    print("\nKey functions:")
    print("  - logit_lens: Get predictions at each layer")
    print("  - plot_prediction_evolution: Visualize prediction formation")
    print("  - plot_probability_heatmap: Heatmap of predictions")
    print("  - measure_convergence: When does model become confident?")
    print("  - compare_positions: Compare across positions")
