#!/bin/bash

# Run All Experiments Script
# This script reproduces the basic experiment pipeline

set -e

echo "=== Friend Recommendation GNN Project ==="
echo ""

# Step 1: Create synthetic dataset
echo "Step 1: Creating synthetic dataset..."
python scripts/download_and_prepare.py --dataset synthetic --preprocess --output data/processed
echo ""

# Step 2: Train GraphSAGE
echo "Step 2: Training GraphSAGE model..."
python scripts/train.py --model graphsage --dataset synthetic --config configs/graphsage_config.yaml --checkpoint_dir data/checkpoints
echo ""

# Step 3: Evaluate GraphSAGE
echo "Step 3: Evaluating GraphSAGE model..."
python scripts/evaluate.py --model graphsage --checkpoint data/checkpoints/graphsage/best_model.pt --dataset synthetic --config configs/graphsage_config.yaml
echo ""

# Step 4: Train GAT
echo "Step 4: Training GAT model..."
python scripts/train.py --model gat --dataset synthetic --config configs/gat_config.yaml --checkpoint_dir data/checkpoints
echo ""

# Step 5: Evaluate GAT
echo "Step 5: Evaluating GAT model..."
python scripts/evaluate.py --model gat --checkpoint data/checkpoints/gat/best_model.pt --dataset synthetic --config configs/gat_config.yaml
echo ""

echo "=== Experiments Complete ==="
echo ""
echo "To run the demo app:"
echo "  streamlit run demo/streamlit_app.py"
echo ""
echo "To run notebooks:"
echo "  jupyter notebook notebooks/"

