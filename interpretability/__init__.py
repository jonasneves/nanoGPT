"""
nanoGPT Interpretability Toolkit

A suite of mechanistic interpretability tools for understanding transformer internals.

Modules:
    - activation_patching: Causal intervention and circuit discovery
    - attention_analysis: Attention pattern visualization and analysis
    - logit_lens: Layer-wise prediction tracking
    - neuron_analysis: Individual neuron and feature analysis
    - utils: Shared utilities and helpers
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from . import activation_patching
from . import attention_analysis
from . import logit_lens
from . import neuron_analysis
from . import utils

__all__ = [
    "activation_patching",
    "attention_analysis",
    "logit_lens",
    "neuron_analysis",
    "utils",
]
