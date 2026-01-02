# Dataset File Locations Guide

This document shows you exactly where to find the Facebook dataset files in your project.

---

## 📁 Dataset File Locations

### **Processed Dataset Files (Ready to Use)**

These are the files you'll use directly in your code:

1. **Main Graph Data:**
   ```
   data/processed/facebook_combined.pt
   ```
   - Contains the graph structure (nodes, edges, features)
   - Format: PyTorch Geometric Data object
   - Includes: `x` (node features), `edge_index` (edges), `num_nodes`

2. **Link Prediction Data:**
   ```
   data/processed/facebook_link_data.pt
   ```
   - Contains train/validation/test splits for link prediction
   - Includes: `train_edges`, `val_edges`, `test_edges`, and their labels

3. **Heuristics (Optional):**
   ```
   data/processed/facebook_heuristics.pt
   ```
   - Contains precomputed heuristic features (if generated)
   - Includes: common neighbors, Jaccard coefficient, etc.

---

## 🔍 How to Access the Dataset in Code

### **Option 1: Using the Load Functions (Recommended)**

```python
import torch

# Load graph data
data = torch.load('data/processed/facebook_combined.pt', weights_only=False)

# Access components
print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.edge_index.size(1) // 2}")
print(f"Features: {data.x.size(1)}")

# Load link prediction data
link_data = torch.load('data/processed/facebook_link_data.pt', weights_only=False)

# Access splits
train_edges = link_data['train_edges']
train_labels = link_data['train_labels']
val_edges = link_data['val_edges']
test_edges = link_data['test_edges']
```

### **Option 2: Using the Project's Load Function**

The project has helper functions in `scripts/evaluate.py` and other scripts:

```python
import sys
sys.path.append('.')

from scripts.evaluate import load_data

# Load both graph and link data
data, link_data = load_data('facebook', data_dir='data/processed')
```

---

## 📂 Raw Dataset Files (Source Data)

If you want to see the original raw data:

```
data/raw/facebook/
```

This directory contains:
- `{ego_id}.edges` - Edge lists for each ego network
- `{ego_id}.feat` - Feature files
- `{ego_id}.featnames` - Feature names
- `{ego_id}.circles` - Social circles (optional)

**Note:** The raw files are only needed if you want to reprocess the dataset. The processed files in `data/processed/` are what you'll use.

---

## 🐍 Quick Access Script

Create a file `check_dataset.py` to quickly view the dataset:

```python
import torch
import os

# Check if files exist
files_to_check = [
    'data/processed/facebook_combined.pt',
    'data/processed/facebook_link_data.pt'
]

print("Facebook Dataset Files:")
print("=" * 60)

for file_path in files_to_check:
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"✓ {file_path}")
        print(f"  Size: {file_size:.2f} MB")
        
        # Load and show basic info
        if 'combined' in file_path:
            data = torch.load(file_path, weights_only=False)
            print(f"  Nodes: {data.num_nodes:,}")
            print(f"  Edges: {data.edge_index.size(1) // 2:,}")
            print(f"  Features: {data.x.size(1)}")
        elif 'link_data' in file_path:
            link_data = torch.load(file_path, weights_only=False)
            print(f"  Train: {link_data['train_edges'].size(1):,} examples")
            print(f"  Val: {link_data['val_edges'].size(1):,} examples")
            print(f"  Test: {link_data['test_edges'].size(1):,} examples")
    else:
        print(f"✗ {file_path} - NOT FOUND")
    print()
```

Run it:
```bash
python check_dataset.py
```

---

## 📍 File Structure

```
Friend-Recommendation-In-Social-Media-Using-GNN/
│
├── data/
│   ├── processed/
│   │   ├── facebook_combined.pt      ← Main dataset file (USE THIS)
│   │   ├── facebook_link_data.pt     ← Link prediction splits (USE THIS)
│   │   └── facebook_heuristics.pt    ← Optional heuristics
│   │
│   └── raw/
│       └── facebook/                  ← Raw source files (not needed for usage)
│           ├── 0.edges
│           ├── 0.feat
│           ├── 0.featnames
│           └── ...
│
└── scripts/
    └── evaluate.py                    ← Has load_data() function
```

---

## 💡 How to Load in Different Scripts

### **In Training Script (`scripts/train.py`):**
```python
data, link_data = load_data('facebook', data_dir='data/processed')
```

### **In Evaluation Script (`scripts/evaluate.py`):**
```python
data, link_data = load_data('facebook', data_dir='data/processed')
```

### **In Streamlit App (`demo/streamlit_app.py`):**
```python
data = torch.load("data/processed/facebook_combined.pt", weights_only=False)
link_data = torch.load("data/processed/facebook_link_data.pt", weights_only=False)
```

### **In Analysis Script (`scripts/analyze_dataset.py`):**
```python
data = torch.load('data/processed/facebook_combined.pt', weights_only=False)
link_data = torch.load('data/processed/facebook_link_data.pt', weights_only=False)
```

---

## ✅ Verification

To verify your dataset files are in place, run:

```bash
python -c "import torch, os; files = ['data/processed/facebook_combined.pt', 'data/processed/facebook_link_data.pt']; [print(f'{f}: {\"✓\" if os.path.exists(f) else \"✗\"}') for f in files]"
```

Or use the analyze script:
```bash
python scripts/analyze_dataset.py
```

---

## 🔧 If Files Are Missing

If the dataset files are not found:

1. **Download and process the dataset:**
   ```bash
   python scripts/download_and_prepare.py --dataset facebook --download --preprocess
   ```

2. **Or just process existing raw data:**
   ```bash
   python scripts/download_and_prepare.py --dataset facebook --preprocess
   ```

---

## 📝 Summary

**Main files to use:**
- ✅ `data/processed/facebook_combined.pt` - Graph structure and features
- ✅ `data/processed/facebook_link_data.pt` - Train/val/test splits

**Path format in code:**
```python
'data/processed/facebook_combined.pt'
'data/processed/facebook_link_data.pt'
```

**Relative to project root:** Use paths relative to the project root directory.

