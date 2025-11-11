# Mechanistic Interpretability Findings

**Project**: nanoGPT Interpretability Toolkit
**Author**: [Your Name]
**Date**: November 2025
**Status**: In Progress

---

## Overview

This document summarizes key findings from applying mechanistic interpretability techniques to small GPT models trained on character-level Shakespeare data. Our goal is to understand the algorithms learned by these models and validate interpretability methods.

---

## Models Analyzed

### Model 1: Shakespeare Character-Level GPT

| Parameter | Value |
|-----------|-------|
| Architecture | GPT |
| Layers | 6 |
| Heads per layer | 6 |
| Embedding dimension | 384 |
| Context length | 128 |
| Vocabulary size | 65 (characters) |
| Total parameters | ~2.5M |
| Training iterations | 5,000 |
| Dataset | Shakespeare (character-level) |
| Final loss | TBD |

---

## Finding 1: Induction Heads Discovery

**Status**: 🔬 In Progress
**Notebook**: `experiments/01_induction_heads.ipynb`

### Hypothesis
Small character-level models should learn induction head circuits similar to those found in larger models (Anthropic, 2022).

### Methodology
1. Created test sequences with repeated patterns: `[A][B]...[A][?]`
2. Used activation patching to identify important layers
3. Analyzed attention patterns for characteristic induction behavior
4. Measured prediction formation using logit lens

### Results

#### Induction Capability
- **Test**: Does model predict [B] after seeing `[A][B]...[A]`?
- **Result**: TBD (pending trained model)
- **Accuracy on induction tasks**: TBD
- **Baseline (random)**: 1.5% (1/65 characters)

#### Layer Localization
*Activation patching results showing which layers are important for induction:*

TBD - Add graph: `results/induction_patching_effects.png`

**Key layers identified**:
- Layer X: Effect = Y.YY
- Layer X: Effect = Y.YY

#### Attention Head Analysis
*Which specific heads implement induction?*

TBD - Add attention visualizations

**Suspected induction heads**:
- Layer X, Head Y: [Description of behavior]

#### Prediction Formation
*Logit lens analysis showing when model becomes confident:*

TBD - Add graph: `results/induction_logit_lens.png`

**Convergence point**: Layer X (model becomes >50% confident)

### Comparison to Literature

| Metric | Our Finding | Anthropic (2022) |
|--------|-------------|------------------|
| Emergence iteration | TBD | ~2000-5000 |
| Layer location | TBD | Middle-to-late layers |
| Number of heads | TBD | 1-2 per model |

### Interpretation

TBD - Discuss what we learned:
- Do induction heads form in small models?
- How similar are they to findings in larger models?
- What does this tell us about in-context learning?

### Implications for AI Safety

TBD - Connect to alignment/safety:
- Can we predict induction head formation?
- What does this tell us about model capabilities?
- How might this relate to deceptive alignment?

---

## Finding 2: Attention Head Specialization

**Status**: 📋 Planned
**Notebook**: `experiments/02_attention_patterns.ipynb`

### Hypothesis
Different attention heads specialize in different types of patterns (previous token, positional, syntactic, etc.).

### Methodology
TBD - Outline approach:
1. Extract all attention patterns
2. Categorize heads by behavior
3. Create taxonomy of head types
4. Measure specialization vs generalization

### Results
TBD - Pending experiment

---

## Finding 3: Layer-wise Prediction Formation

**Status**: 📋 Planned
**Notebook**: `experiments/03_logit_lens_analysis.ipynb`

### Hypothesis
Predictions form gradually across layers, with early layers capturing bigram statistics and late layers capturing long-range dependencies.

### Methodology
TBD - Outline approach:
1. Apply logit lens at each layer
2. Track prediction evolution for various sequence types
3. Measure convergence points
4. Compare across different input types

### Results
TBD - Pending experiment

---

## Finding 4: [Novel Discovery]

**Status**: 💡 Potential
**Notebook**: `experiments/04_novel_findings.ipynb`

### Research Question
TBD - What unique insight can we discover?

Ideas:
- How do circuits evolve during training?
- What's the minimal model size for specific capabilities?
- How robust are interpretability findings across random seeds?
- Can we predict model failures from circuit analysis?

### Results
TBD - This is where you make your mark!

---

## Methodological Insights

### Tool Validation

#### Activation Patching
**Effectiveness**: TBD
- Successfully identifies important components: Yes/No
- Computational cost: TBD ms per layer
- Robustness across random seeds: TBD

**Lessons learned**:
- TBD

#### Attention Analysis
**Effectiveness**: TBD
- Pattern extraction works reliably: Yes/No
- Visualization quality: TBD
- Categorization accuracy: TBD

**Lessons learned**:
- TBD

#### Logit Lens
**Effectiveness**: TBD
- Reveals prediction formation: Yes/No
- Convergence measurements meaningful: Yes/No
- Computational overhead: TBD

**Lessons learned**:
- TBD

### Challenges Encountered

1. **Challenge**: TBD
   - **Solution**: TBD
   - **Impact**: TBD

2. **Challenge**: TBD
   - **Solution**: TBD
   - **Impact**: TBD

---

## Reproducibility

### Replication Results

Tested across multiple random seeds:
- Seed 1: TBD
- Seed 2: TBD
- Seed 3: TBD

**Consistency**: TBD% of findings replicate across seeds

### Limitations

1. **Model Size**: Findings based on very small models (2.5M params) may not generalize to larger models
2. **Dataset**: Shakespeare is limited domain; findings may be task-specific
3. **Training**: Only 5K iterations; longer training may reveal different circuits
4. **Tools**: Some interpretability methods are approximate (e.g., logit lens assumes linear unembedding)

---

## Future Directions

### Immediate Next Steps
1. Train models with different random seeds
2. Vary model architecture (layers, heads, dimensions)
3. Test on different datasets (code, Wikipedia, etc.)
4. Implement additional interpretability techniques

### Research Questions
1. How do circuits evolve during training?
2. Can we predict emergence of capabilities?
3. What determines which layers learn which algorithms?
4. How do findings scale to larger models?

### Tool Improvements
1. Add sparse autoencoder (SAE) analysis
2. Implement causal scrubbing
3. Build interactive visualization dashboards
4. Create automated circuit discovery

---

## Visualizations

### Key Figures

1. **Induction Patching Effects** (`induction_patching_effects.png`)
   - Shows layer importance for induction via activation patching

2. **Logit Lens Evolution** (`induction_logit_lens.png`)
   - Tracks prediction formation across layers

3. **Attention Pattern Grid** (TBD)
   - Visualization of all attention heads in key layer

4. **Circuit Diagram** (TBD)
   - Minimal circuit implementing induction behavior

---

## Code & Data

### Artifacts
- **Trained models**: `trained_models/shakespeare_char_model.pt`
- **Experiment notebooks**: `experiments/01_induction_heads.ipynb`
- **Analysis results**: `results/induction_heads_analysis.json`
- **Figures**: `results/*.png`

### Reproduction Instructions
```bash
# 1. Train model
bash scripts/train_for_interpretability.sh

# 2. Run analysis
jupyter notebook experiments/01_induction_heads.ipynb

# 3. View results
cat results/FINDINGS.md
```

---

## References

### Key Papers
1. Elhage et al. (2021). "A Mathematical Framework for Transformer Circuits." Anthropic.
2. Olsson et al. (2022). "In-context Learning and Induction Heads." Anthropic.
3. Bills et al. (2023). "Interpretability in the Wild." Anthropic.
4. Olah et al. (2020). "Zoom In: An Introduction to Circuits." Distill.

### Tools & Libraries
1. nanoGPT: https://github.com/karpathy/nanoGPT
2. TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
3. ARENA Tutorials: https://arena-ch1-transformers.streamlit.app/

---

## Acknowledgements

- Andrej Karpathy for nanoGPT
- Anthropic's Interpretability Team for pioneering research
- TransformerLens for methodological inspiration

---

## Updates Log

### 2025-11-11
- Created findings template
- Set up experiment infrastructure
- Defined research questions

### [Date]
- TBD - Log major discoveries and updates

---

**Note**: This document will be updated as experiments are completed and findings emerge.
