#!/bin/bash

# Script to train small models for interpretability analysis
# This creates models suitable for mechanistic interpretability experiments

echo "========================================="
echo "Training Model for Interpretability"
echo "========================================="
echo ""

# Check if data is prepared
if [ ! -f "data/shakespeare_char/train.bin" ]; then
    echo "Preparing Shakespeare dataset..."
    python data/shakespeare_char/prepare.py
    echo "Dataset prepared!"
    echo ""
fi

# Train the model
echo "Training small model (6 layers, 384 dim, ~5000 iters)"
echo "This should take ~5-10 minutes on a GPU..."
echo ""

python train.py config/train_interpretability.py

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Training completed successfully!"
    echo "========================================="
    echo ""

    # Copy checkpoint to trained_models directory
    mkdir -p trained_models

    if [ -f "out-interpretability/ckpt.pt" ]; then
        cp out-interpretability/ckpt.pt trained_models/shakespeare_char_model.pt
        echo "✓ Checkpoint copied to: trained_models/shakespeare_char_model.pt"
        echo ""
        echo "You can now run the interpretability experiments:"
        echo "  jupyter notebook experiments/01_induction_heads.ipynb"
    else
        echo "Warning: Checkpoint not found at expected location"
    fi
else
    echo "Training failed. Check the error messages above."
    exit 1
fi

echo ""
echo "Next steps:"
echo "1. Open experiments/01_induction_heads.ipynb"
echo "2. Run the notebook to analyze induction heads"
echo "3. Explore other interpretability techniques"
