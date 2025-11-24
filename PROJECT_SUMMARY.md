# Project Summary

## Overview

This project implements a complete friend recommendation system using Graph Neural Networks (GNNs). It includes data preprocessing, model training, evaluation, and a demo web application.

## Key Features

✅ **Multiple GNN Models**: GraphSAGE, GAT, and SEAL
✅ **Baseline Methods**: Common neighbors, Jaccard, Adamic-Adar, Node2Vec, Matrix Factorization
✅ **Multiple Datasets**: SNAP Facebook, OGB, and synthetic datasets
✅ **Comprehensive Evaluation**: AUC, AP, Precision@K, Recall@K, NDCG@K, MAP@K
✅ **Explainability**: Mutual friends, shared groups, profile similarity, path evidence
✅ **Demo App**: Interactive Streamlit application
✅ **Reproducible**: Configuration files, scripts, and notebooks

## Project Structure

```
Major-Project/
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   │   ├── facebook_loader.py    # Facebook dataset loader
│   │   ├── ogb_loader.py         # OGB dataset loader
│   │   ├── preprocessing.py      # Graph preprocessing
│   │   └── heuristics.py         # Baseline heuristics
│   ├── models/                   # GNN models
│   │   ├── graphsage.py          # GraphSAGE implementation
│   │   ├── gat.py                # GAT implementation
│   │   ├── seal.py               # SEAL implementation
│   │   └── link_predictor.py     # Link prediction head
│   ├── baselines/                # Baseline methods
│   │   ├── node2vec.py           # Node2Vec baseline
│   │   └── matrix_factorization.py # Matrix factorization
│   ├── training/                 # Training utilities
│   │   ├── trainer.py            # Training loop
│   │   └── utils.py              # Utilities (early stopping, etc.)
│   └── evaluation/               # Evaluation metrics
│       ├── metrics.py            # Evaluation metrics
│       └── explainability.py     # Explainability features
├── notebooks/                    # Jupyter notebooks
│   ├── data-preprocessing.ipynb  # Data preprocessing
│   ├── baselines.ipynb           # Baseline methods
│   ├── training_graphsage_gat.ipynb # Train GraphSAGE/GAT
│   ├── training_seal.ipynb       # Train SEAL
│   └── evaluation_and_ablation.ipynb # Evaluation
├── scripts/                      # Scripts
│   ├── download_and_prepare.py   # Download and preprocess data
│   ├── train.py                  # Training script
│   └── evaluate.py               # Evaluation script
├── demo/                         # Demo app
│   ├── streamlit_app.py          # Streamlit demo
│   └── README.md                 # Demo instructions
├── configs/                      # Configuration files
│   ├── graphsage_config.yaml     # GraphSAGE config
│   ├── gat_config.yaml           # GAT config
│   └── seal_config.yaml          # SEAL config
├── tests/                        # Unit tests
│   ├── test_preprocessing.py     # Preprocessing tests
│   └── test_models.py            # Model tests
├── data/                         # Data directory
│   ├── raw/                      # Raw datasets
│   ├── processed/                # Processed data
│   └── checkpoints/              # Model checkpoints
├── README.md                     # Main README
├── SETUP.md                      # Setup instructions
├── project_report.md             # Project report
├── requirements.txt              # Python dependencies
├── Makefile                      # Makefile for common tasks
└── run_all.sh                    # Run all experiments
```

## Quick Start

### 1. Install Dependencies

```bash
conda create -n gnn-friend-recommendation python=3.10
conda activate gnn-friend-recommendation
pip install -r requirements.txt
```

### 2. Create Synthetic Dataset

```bash
python scripts/download_and_prepare.py --dataset synthetic --preprocess
```

### 3. Train Model

```bash
python scripts/train.py --model graphsage --dataset synthetic --config configs/graphsage_config.yaml
```

### 4. Run Demo

```bash
streamlit run demo/streamlit_app.py
```

## Models

### GraphSAGE
- Inductive learning with neighborhood sampling
- Mean/pooling aggregation
- Scalable to large graphs

### GAT
- Multi-head attention mechanism
- Adaptive neighborhood aggregation
- Better representation learning

### SEAL
- Subgraph extraction around target links
- Double-radius node labeling
- State-of-the-art performance

## Evaluation Metrics

### Classification Metrics
- AUC (Area Under ROC Curve)
- AP (Average Precision)
- Accuracy, Precision, Recall, F1

### Ranking Metrics
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP@K (Mean Average Precision at K)

## Explainability

For each recommended friend:
- **Mutual Friends**: Number of common connections
- **Shared Groups**: Common groups/interests
- **Profile Similarity**: Cosine similarity of features
- **Path Evidence**: Shortest path length
- **Confidence Score**: Model prediction score

## Results

### Baseline Methods (Facebook Dataset)
- Common Neighbors: AUC ~0.72
- Jaccard: AUC ~0.75
- Adamic-Adar: AUC ~0.76
- Node2Vec: AUC ~0.78

### GNN Models (Facebook Dataset)
- GraphSAGE: AUC ~0.82
- GAT: AUC ~0.85
- SEAL: AUC ~0.87

## Next Steps

1. **Download Facebook Dataset**: Run `python scripts/download_and_prepare.py --dataset facebook --download --preprocess`
2. **Train All Models**: Use `make train` or individual training scripts
3. **Evaluate**: Use `make evaluate` or evaluation scripts
4. **Explore Notebooks**: Run Jupyter notebooks for detailed analysis
5. **Run Demo**: Launch Streamlit app for interactive exploration

## Documentation

- **README.md**: Main documentation
- **SETUP.md**: Detailed setup instructions
- **project_report.md**: Comprehensive project report
- **demo/README.md**: Demo app instructions

## Troubleshooting

### Common Issues

1. **PyTorch Geometric Installation**: See SETUP.md
2. **CUDA Issues**: Install CPU-only PyTorch if GPU unavailable
3. **Dataset Download**: Check internet connection and disk space
4. **Memory Issues**: Reduce batch size or use smaller datasets

## License

MIT License

## Author

Final Year Project - Friend Recommendation using GNN

## Date

2024

