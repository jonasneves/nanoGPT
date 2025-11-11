"""
Unit tests for activation patching module.

Run with: pytest tests/test_activation_patching.py
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import GPT, GPTConfig
from interpretability import activation_patching
from interpretability.utils import ActivationCache, HookManager


@pytest.fixture
def small_model():
    """Create a small GPT model for testing."""
    config = GPTConfig(
        block_size=64,
        vocab_size=50,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False
    )
    model = GPT(config)
    model.eval()
    return model


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randint(0, 50, (2, 20))  # batch_size=2, seq_len=20


class TestActivationCache:
    """Test ActivationCache functionality."""

    def test_store_and_retrieve(self):
        cache = ActivationCache()
        test_tensor = torch.randn(2, 3, 4)

        cache.store("test_layer", test_tensor)
        retrieved = cache.get("test_layer")

        assert retrieved is not None
        assert torch.allclose(retrieved, test_tensor)

    def test_clear(self):
        cache = ActivationCache()
        cache.store("layer1", torch.randn(2, 3))
        cache.store("layer2", torch.randn(2, 3))

        assert len(cache.keys()) == 2

        cache.clear()
        assert len(cache.keys()) == 0

    def test_get_nonexistent(self):
        cache = ActivationCache()
        result = cache.get("nonexistent")
        assert result is None


class TestHookManager:
    """Test HookManager functionality."""

    def test_hook_registration(self, small_model):
        manager = HookManager()
        hook_called = []

        def test_hook(module, input, output):
            hook_called.append(True)

        # Register hook on first layer
        layer = small_model.transformer.h[0]
        manager.register_hook(layer, test_hook)

        # Run forward pass
        inputs = torch.randint(0, 50, (1, 10))
        with torch.no_grad():
            _ = small_model(inputs)

        assert len(hook_called) > 0
        manager.remove_all_hooks()

    def test_context_manager(self, small_model):
        hook_called = []

        def test_hook(module, input, output):
            hook_called.append(True)

        with HookManager() as manager:
            layer = small_model.transformer.h[0]
            manager.register_hook(layer, test_hook)

            inputs = torch.randint(0, 50, (1, 10))
            with torch.no_grad():
                _ = small_model(inputs)

        # Hooks should be automatically removed
        assert len(manager.hooks) == 0


class TestGetActivations:
    """Test get_activations function."""

    def test_basic_functionality(self, small_model, sample_input):
        layer_names = ["transformer.h.0", "transformer.h.1"]

        activations, output = activation_patching.get_activations(
            small_model,
            sample_input,
            layer_names,
            return_output=True
        )

        assert len(activations) == 2
        assert "transformer.h.0" in activations
        assert "transformer.h.1" in activations
        assert output is not None

    def test_without_output(self, small_model, sample_input):
        layer_names = ["transformer.h.0"]

        activations, output = activation_patching.get_activations(
            small_model,
            sample_input,
            layer_names,
            return_output=False
        )

        assert len(activations) == 1
        assert output is None

    def test_activation_shapes(self, small_model, sample_input):
        layer_names = ["transformer.h.0"]

        activations, _ = activation_patching.get_activations(
            small_model,
            sample_input,
            layer_names
        )

        # Check shape matches input shape with embedding dimension
        act = activations["transformer.h.0"]
        batch_size, seq_len = sample_input.shape

        assert act.shape[0] == batch_size
        assert act.shape[1] == seq_len
        assert act.shape[2] == small_model.config.n_embd


class TestPatchActivation:
    """Test patch_activation function."""

    def test_patching_changes_output(self, small_model, sample_input):
        # Get original output
        with torch.no_grad():
            original_output, _ = small_model(sample_input)

        # Get activation to patch with
        layer_name = "transformer.h.0"
        activations, _ = activation_patching.get_activations(
            small_model,
            sample_input,
            [layer_name]
        )

        # Create modified activation (zero it out)
        modified_activation = torch.zeros_like(activations[layer_name])

        # Patch with modified activation
        patched_output, _ = activation_patching.patch_activation(
            small_model,
            sample_input,
            layer_name,
            modified_activation
        )

        # Outputs should be different
        assert not torch.allclose(original_output[0], patched_output[0], rtol=1e-3)


class TestAblateComponent:
    """Test ablate_component function."""

    def test_zero_ablation(self, small_model, sample_input):
        # Get original output
        with torch.no_grad():
            original_output, _ = small_model(sample_input)

        # Ablate first layer
        ablated_output = activation_patching.ablate_component(
            small_model,
            sample_input,
            "transformer.h.0",
            ablation_type="zero"
        )

        # Outputs should be different
        assert not torch.allclose(
            original_output[0],
            ablated_output[0],
            rtol=1e-3
        )

    def test_invalid_ablation_type(self, small_model, sample_input):
        with pytest.raises(ValueError):
            activation_patching.ablate_component(
                small_model,
                sample_input,
                "transformer.h.0",
                ablation_type="invalid"
            )


class TestComputePatchingEffect:
    """Test compute_patching_effect function."""

    def test_perfect_restoration(self):
        clean = torch.tensor([[1.0, 2.0, 3.0]])
        corrupted = torch.tensor([[0.0, 0.0, 0.0]])
        patched = clean.clone()

        effect = activation_patching.compute_patching_effect(
            clean, corrupted, patched, metric="logit_diff"
        )

        # Perfect restoration should give effect H 1.0
        assert effect > 0.9

    def test_no_effect(self):
        clean = torch.tensor([[1.0, 2.0, 3.0]])
        corrupted = torch.tensor([[0.0, 0.0, 0.0]])
        patched = corrupted.clone()

        effect = activation_patching.compute_patching_effect(
            clean, corrupted, patched, metric="logit_diff"
        )

        # No restoration should give effect H 0.0
        assert effect < 0.1


class TestFindImportantComponents:
    """Test find_important_components function."""

    def test_threshold_filtering(self):
        # Create mock results
        from interpretability.activation_patching import PatchingResult

        results = {
            "layer0": PatchingResult("layer0", None, None, None, 0.8, {}),
            "layer1": PatchingResult("layer1", None, None, None, 0.3, {}),
            "layer2": PatchingResult("layer2", None, None, None, 0.6, {}),
        }

        important = activation_patching.find_important_components(
            results,
            threshold=0.5
        )

        assert len(important) == 2
        assert important[0][0] == "layer0"  # Highest effect
        assert important[1][0] == "layer2"

    def test_top_k_filtering(self):
        from interpretability.activation_patching import PatchingResult

        results = {
            f"layer{i}": PatchingResult(f"layer{i}", None, None, None, i * 0.1, {})
            for i in range(10)
        }

        important = activation_patching.find_important_components(
            results,
            top_k=3
        )

        assert len(important) == 3
        assert important[0][0] == "layer9"  # Highest effect


# Integration tests
class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_patching_workflow(self, small_model):
        # Create test sequences
        clean_input = torch.randint(0, 50, (1, 20))
        corrupted_input = torch.randint(0, 50, (1, 20))

        # Run layer scan
        results = activation_patching.patch_layer_scan(
            small_model,
            clean_input,
            corrupted_input,
            layer_names=["transformer.h.0", "transformer.h.1"],
            show_progress=False
        )

        assert len(results) == 2
        assert all(isinstance(r.effect, float) for r in results.values())

        # Find important components
        important = activation_patching.find_important_components(
            results,
            threshold=0.0
        )

        assert len(important) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
