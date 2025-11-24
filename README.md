# Friend Recommendation in Social Networks using Graph Neural Networks (GNN)

A complete implementation of friend recommendation (link prediction) using Graph Neural Networks, including GraphSAGE, GAT, and SEAL models.

## Project Structure

```
.
├── data/                    # Dataset storage
│   ├── raw/                 # Raw downloaded datasets
│   ├── processed/           # Processed graph data
│   └── checkpoints/         # Model checkpoints
├── src/                     # Source code
│   ├── data/                # Data loading and preprocessing
│   ├── models/              # GNN models
│   ├── baselines/           # Baseline methods
│   ├── training/            # Training utilities
│   ├── evaluation/          # Evaluation metrics
│   └── utils/               # Utility functions
├── notebooks/               # Jupyter notebooks
├── scripts/                 # Utility scripts
├── tests/                   # Unit tests
├── demo/                    # Streamlit demo app
├── configs/                 # Configuration files
└── requirements.txt         # Python dependencies
```

## Installation

### 1. Create Conda Environment

```bash
conda create -n gnn-friend-recommendation python=3.10
conda activate gnn-friend-recommendation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install PyTorch Geometric

```bash
# For CUDA 11.8 (adjust based on your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## Quick Start

### Option 1: Using Synthetic Dataset (Fastest)

For quick testing, use the synthetic dataset:

```bash
# Create synthetic dataset
python scripts/download_and_prepare.py --dataset synthetic --preprocess

# Train GraphSAGE
python scripts/train.py --model graphsage --dataset synthetic --config configs/graphsage_config.yaml

# Run demo
streamlit run demo/streamlit_app.py
```

### Option 2: Using Facebook Dataset

```bash
# Download and preprocess Facebook dataset
python scripts/download_and_prepare.py --dataset facebook --download --preprocess

# Train models
python scripts/train.py --model graphsage --dataset facebook --config configs/graphsage_config.yaml
python scripts/train.py --model gat --dataset facebook --config configs/gat_config.yaml

# Evaluate
python scripts/evaluate.py --model graphsage --checkpoint data/checkpoints/graphsage/best_model.pt --dataset facebook --config configs/graphsage_config.yaml
```

### Option 3: Using Makefile

```bash
# Install dependencies
make install

# Prepare data
make data

# Train models
make train

# Evaluate
make evaluate

# Run demo
make demo
```

### Option 4: Using run_all.sh

```bash
chmod +x run_all.sh
./run_all.sh
```

Or use Jupyter notebooks:
```bash
jupyter notebook notebooks/data-preprocessing.ipynb
```

## Notebooks

1. **data-preprocessing.ipynb**: Download and preprocess datasets
2. **baselines.ipynb**: Run baseline methods (common neighbors, Node2Vec)
3. **training_graphsage_gat.ipynb**: Train GraphSAGE and GAT models
4. **training_seal.ipynb**: Train SEAL model
5. **evaluation_and_ablation.ipynb**: Comprehensive evaluation and ablation study

## Datasets

### SNAP Facebook Social Circles
- Small-scale ego-network dataset
- Includes node features and circles
- Good for quick experimentation

### OGB Link Prediction Datasets
- ogbl-collab: Collaboration network
- Robust train/val/test splits
- Large-scale evaluation

## Model Architecture

### GraphSAGE
- Mean and pooling aggregator variants
- Inductive learning capability
- Scalable to large graphs

### GAT (Graph Attention Network)
- Multi-head attention mechanism
- Adaptive neighborhood aggregation
- Better representation learning

### SEAL (Subgraph Embedding and Link prediction)
- Subgraph extraction around target links
- Double-radius node labeling
- State-of-the-art link prediction performance

## Evaluation Metrics

- **Link Classification**: AUC, AP
- **Top-K Recommendations**: Precision@K, Recall@K, NDCG@K, MAP@K
- **Runtime & Memory**: Training time, inference time, memory usage

## Explainability

For each recommended friend, we show:
- Number of mutual friends
- Common groups/interests
- Profile similarity score
- Path evidence (shortest path length)
- Confidence score (model prediction)

## Reproducibility

All experiments use fixed random seeds. Configuration files in `configs/` directory contain hyperparameters. See `project_report.md` for detailed experimental results.

## License

MIT License

## Author

Final Year Project - Friend Recommendation using GNN

