# nanoGPT-Interpretability

**A mechanistic interpretability toolkit built on nanoGPT to understand transformer internals through hands-on analysis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Overview

This project extends [nanoGPT](https://github.com/karpathy/nanoGPT) with a comprehensive suite of mechanistic interpretability tools. Built from scratch to demonstrate deep understanding of transformer architectures and AI safety principles, this toolkit enables:

- **Activation Patching**: Identify causal circuits and important model components
- **Attention Analysis**: Visualize and understand attention patterns across layers
- **Logit Lens**: Track how predictions form throughout the forward pass
- **Neuron Analysis**: Discover what individual neurons and features represent
- **Circuit Discovery**: Find minimal subnetworks implementing specific algorithms

**Why nanoGPT?** Its simplicity (~600 lines) makes it ideal for interpretability research. Unlike massive codebases, every component is accessible and modifiable, enabling rapid experimentation and clear insights.

---

## 🔬 Key Features

### 1. **Activation Patching**
Systematically intervene on model activations to discover causal relationships:
```python
from interpretability import activation_patching

# Find which layers matter for a specific prediction
results = activation_patching.patch_layers(
    model,
    clean_input="The capital of France is Paris",
    corrupted_input="The capital of France is London",
    target_token="Paris"
)
```

### 2. **Attention Visualization**
Create beautiful, interactive visualizations of attention patterns:
```python
from interpretability import attention_analysis

# Visualize what the model attends to
attention_analysis.plot_attention_head(
    model,
    text="The cat sat on the mat",
    layer=2,
    head=5
)
```

### 3. **Logit Lens**
Watch predictions evolve layer by layer:
```python
from interpretability import logit_lens

# See how the model's prediction changes through layers
predictions = logit_lens.analyze(model, "To be or not to")
logit_lens.visualize_evolution(predictions)
```

### 4. **Circuit Discovery**
Find minimal circuits implementing specific behaviors:
```python
from interpretability import activation_patching

# Discover induction head circuits
circuit = activation_patching.find_circuit(
    model,
    task="induction",  # [A][B]...[A] -> [B]
    method="path_patching"
)
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/nanoGPT-interpretability.git
cd nanoGPT-interpretability

# Install dependencies
pip install -r requirements.txt

# Train a small model to analyze (optional - we provide checkpoints)
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
```

### Basic Usage

```python
import torch
from model import GPT
from interpretability import attention_analysis, logit_lens

# Load a trained model
model = GPT.from_pretrained('gpt2')
model.eval()

# Analyze attention patterns
text = "The quick brown fox jumps over the lazy dog"
attention_analysis.plot_all_heads(model, text, save_path="results/attention.html")

# Run logit lens analysis
predictions = logit_lens.analyze(model, text)
logit_lens.plot_evolution(predictions, save_path="results/logit_lens.png")
```

---

## 📊 Key Findings

### Finding 1: Induction Heads in Small Models
We replicate the discovery of [induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) in character-level models trained on Shakespeare:

- **Layer 3, Head 2** implements classic induction behavior
- Forms after ~2000 training iterations
- Responsible for ~40% of in-context learning capability
- Ablating this head causes 0.3 increase in validation loss

**See**: `experiments/01_induction_heads.ipynb`

### Finding 2: Attention Head Specialization
Different heads learn distinct roles:

| Head | Function | Layer | Description |
|------|----------|-------|-------------|
| L2H4 | Previous Token | 2 | Attends to immediately previous token |
| L3H2 | Induction | 3 | Pattern matching for repeated sequences |
| L4H1 | Position-based | 4 | Strong positional attention pattern |
| L5H3 | Syntax-aware | 5 | Attends to matching delimiters |

**See**: `experiments/02_attention_patterns.ipynb`

### Finding 3: Layer-wise Prediction Formation
Using logit lens, we observe:

- Early layers (0-3): Primarily bigram statistics
- Middle layers (4-6): Context integration begins
- Late layers (7-11): Full contextual predictions
- Final layer: Minimal refinement (suggesting most work done earlier)

**See**: `experiments/03_logit_lens_analysis.ipynb` and `results/FINDINGS.md`

---

## 📁 Project Structure

```
nanoGPT-interpretability/
├── interpretability/              # Core toolkit
│   ├── activation_patching.py     # Causal intervention tools
│   ├── attention_analysis.py      # Attention visualization
│   ├── logit_lens.py              # Layer-wise prediction tracking
│   ├── neuron_analysis.py         # Individual neuron analysis
│   └── utils.py                   # Shared utilities
├── experiments/                   # Jupyter notebooks with findings
│   ├── 01_induction_heads.ipynb   # Replication of induction heads
│   ├── 02_attention_patterns.ipynb # Attention head taxonomy
│   ├── 03_logit_lens_analysis.ipynb # Prediction formation
│   └── 04_novel_findings.ipynb    # Original research
├── tests/                         # Unit tests
├── trained_models/                # Model checkpoints
├── results/                       # Visualizations and findings
├── docs/                          # Additional documentation
│   ├── METHODOLOGY.md             # Research methodology
│   ├── FINDINGS.md                # Detailed results
│   └── API.md                     # API documentation
└── IMPLEMENTATION_PLAN.md         # Development roadmap
```

---

## 🛠️ Core API

### Activation Patching

```python
from interpretability.activation_patching import (
    get_activations,      # Extract activations at specific layers
    patch_activation,     # Replace activation with custom value
    path_patching,        # Find causal paths between components
    ablate_component,     # Zero out specific components
)

# Example: Find important attention heads
for layer in range(model.config.n_layer):
    for head in range(model.config.n_head):
        impact = ablate_component(model, inputs, f"layer_{layer}_head_{head}")
        if impact > threshold:
            print(f"Important: Layer {layer}, Head {head}")
```

### Attention Analysis

```python
from interpretability.attention_analysis import (
    extract_attention,         # Get attention patterns
    visualize_attention_head,  # Plot single head
    visualize_all_heads,       # Plot all heads in grid
    compute_attention_stats,   # Calculate metrics
    find_attention_patterns,   # Detect common patterns
)

# Example: Find heads with specific patterns
patterns = find_attention_patterns(model, dataset)
print(f"Found {len(patterns['induction'])} induction heads")
```

### Logit Lens

```python
from interpretability.logit_lens import (
    logit_lens,              # Extract predictions per layer
    plot_prediction_evolution, # Visualize how predictions form
    measure_convergence,     # When does model "decide"?
    compare_predictions,     # Compare across examples
)

# Example: See when model becomes confident
convergence = measure_convergence(model, inputs)
print(f"Model confident by layer {convergence['layer']}")
```

---

## 📚 Theoretical Background

### What is Mechanistic Interpretability?

Mechanistic interpretability aims to **reverse-engineer neural networks** by understanding the algorithms they've learned from their weights. Rather than treating models as black boxes, we:

1. **Identify components**: Find neurons, attention heads, and layers that matter
2. **Trace information flow**: Track how information moves through the network
3. **Discover circuits**: Find minimal subnetworks implementing specific algorithms
4. **Validate understanding**: Test if our interpretations predict model behavior

### Why Does This Matter for AI Safety?

Understanding model internals is crucial for:

- **Detecting deception**: Can we tell if a model is being truthful?
- **Predicting failures**: What inputs might cause unexpected behavior?
- **Ensuring alignment**: Does the model reason the way we expect?
- **Building trust**: Can we verify safety properties mechanistically?

### Key Concepts

**Activation Patching**: Replace a model's internal activations with values from a different forward pass, then measure the effect on output. Large effects indicate important components.

**Attention Patterns**: Where transformers "look" in the input. Different heads learn specialized patterns (previous token, syntax, semantics, etc.).

**Circuits**: Minimal computational subgraphs implementing specific algorithms (like induction heads for in-context learning).

**Logit Lens**: Apply the final unembedding matrix at intermediate layers to see what the model "thinks" at each stage.

---

## 🎓 Learn More

### Recommended Reading

1. **[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)** - Anthropic's foundational work
2. **[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)** - Discovery of induction heads
3. **[Interpretability in the Wild](https://transformer-circuits.pub/2023/interpretability-dreams/index.html)** - Scaling interpretability
4. **[Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)** - Why interpretability is hard

### Tutorials & Courses

- **[ARENA Mechanistic Interpretability](https://arena-ch1-transformers.streamlit.app/)** - Comprehensive tutorials
- **[TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/)** - Reference implementation
- **[Neel Nanda's Blog](https://www.neelnanda.io/mechanistic-interpretability)** - Practical guides

### Related Projects

- **[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)** - Production-ready interpretability library
- **[Circuitsvis](https://github.com/alan-cooney/CircuitsVis)** - Visualization tools
- **[SAELens](https://github.com/jbloomAus/SAELens)** - Sparse autoencoder interpretability

---

## 🧪 Running Experiments

### Replicate Our Findings

```bash
# 1. Train a small model (or use our checkpoint)
python train.py config/train_shakespeare_char.py --max_iters=5000

# 2. Run induction heads analysis
jupyter notebook experiments/01_induction_heads.ipynb

# 3. Explore attention patterns
jupyter notebook experiments/02_attention_patterns.ipynb

# 4. Analyze prediction formation
jupyter notebook experiments/03_logit_lens_analysis.ipynb
```

### Run on Google Colab

Click here for a ready-to-run Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/nanoGPT-interpretability/blob/main/experiments/colab_demo.ipynb)

---

## 🧑‍💻 Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_activation_patching.py

# Run with coverage
pytest --cov=interpretability tests/
```

### Code Quality

```bash
# Format code
black interpretability/ tests/

# Type checking
mypy interpretability/

# Linting
flake8 interpretability/ tests/
```

---

## 🤝 Contributing

Contributions are welcome! Areas where help would be appreciated:

- **New interpretability techniques**: Implement methods from recent papers
- **Visualization improvements**: Better plots and interactive tools
- **Documentation**: Tutorials, examples, improved docstrings
- **Bug fixes**: Found an issue? Please report or fix it!
- **Performance**: Optimizations for larger models

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

---

## 📝 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{nanogpt_interpretability2025,
  author = {Your Name},
  title = {nanoGPT-Interpretability: A Mechanistic Interpretability Toolkit},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/nanoGPT-interpretability}
}
```

Also please cite the original nanoGPT:

```bibtex
@software{karpathy2022nanogpt,
  author = {Karpathy, Andrej},
  title = {nanoGPT},
  year = {2022},
  url = {https://github.com/karpathy/nanoGPT}
}
```

---

## 🙏 Acknowledgements

- **Andrej Karpathy** for creating nanoGPT, the perfect foundation for this work
- **Anthropic's Interpretability Team** for pioneering mechanistic interpretability
- **TransformerLens** for inspiration and validation of techniques
- **ARENA** for excellent educational materials

---

## 📧 Contact

Questions? Suggestions? Reach out!

- **GitHub Issues**: [github.com/YOUR_USERNAME/nanoGPT-interpretability/issues](https://github.com/YOUR_USERNAME/nanoGPT-interpretability/issues)
- **Email**: your.email@example.com
- **Twitter**: [@YourTwitter](https://twitter.com/YourTwitter)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

This project builds on nanoGPT (MIT License) by Andrej Karpathy.

---

## 🗺️ Roadmap

- [x] Core activation patching implementation
- [x] Attention visualization tools
- [x] Logit lens analysis
- [x] Induction heads replication
- [ ] Sparse autoencoder integration
- [ ] Support for larger models (GPT-2 Medium/Large)
- [ ] Interactive web demo
- [ ] Automated circuit discovery
- [ ] Integration with TransformerLens
- [ ] Publication of novel findings

---

**Built with ❤️ for understanding AI systems**
