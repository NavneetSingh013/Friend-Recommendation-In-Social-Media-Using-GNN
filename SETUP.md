# Setup Instructions

## Prerequisites

- Python 3.10+
- CUDA (optional, for GPU training)
- conda or pip

## Step-by-Step Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd Major-Project
```

### 2. Create Conda Environment

```bash
conda create -n gnn-friend-recommendation python=3.10
conda activate gnn-friend-recommendation
```

### 3. Install PyTorch

```bash
# For CPU only
pip install torch torchvision torchaudio

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install PyTorch Geometric

```bash
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

Adjust the PyTorch version in the URL based on your installed version.

### 5. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 6. Verify Installation

```python
python -c "import torch; import torch_geometric; print('Installation successful!')"
```

## Quick Test

```bash
# Create synthetic dataset
python scripts/download_and_prepare.py --dataset synthetic --preprocess

# Verify data was created
ls data/processed/
```

## Troubleshooting

### PyTorch Geometric Installation Issues

If you encounter issues with PyTorch Geometric:

1. Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
2. Install matching PyG version from [official site](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

### CUDA Issues

If CUDA is not available:

1. Check CUDA installation: `nvidia-smi`
2. Install CPU-only PyTorch if GPU is not available
3. Models will run on CPU (slower but functional)

### Missing Dependencies

If you get import errors:

```bash
pip install -r requirements.txt --upgrade
```

## Next Steps

1. Run data preprocessing (see README.md)
2. Train models (see README.md)
3. Run demo app (see README.md)

