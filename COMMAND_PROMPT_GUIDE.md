# Step-by-Step Command Prompt Guide

## Complete Project Execution Guide

Follow these steps in order to run the entire project from scratch.

---

## Step 1: Navigate to Project Directory

```cmd
cd C:\Users\navne\Desktop\Major-Project
```

---

## Step 2: Verify Python Installation

```cmd
python --version
```

Should show Python 3.10 or higher. If not, install Python first.

---

## Step 3: Install Dependencies

### Option A: Install from requirements.txt (Recommended)

```cmd
pip install -r requirements.txt
```

### Option B: Install Core Dependencies Manually

```cmd
pip install torch torchvision torchaudio
pip install torch-geometric
pip install numpy pandas scikit-learn networkx
pip install streamlit matplotlib seaborn
pip install pyyaml tqdm jupyter
```

**Note**: For PyTorch Geometric, you may need to install additional packages:
```cmd
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

---

## Step 4: Create Synthetic Dataset (Quick Start)

```cmd
python scripts/download_and_prepare.py --dataset synthetic --preprocess
```

**Expected Output:**
```
Creating synthetic dataset with 100 nodes...
Saved synthetic dataset to data/processed\synthetic.pt
Preparing link prediction data...
Saved link prediction data to data/processed\synthetic_link_data.pt
Synthetic dataset created successfully!
```

---

## Step 5: Train GraphSAGE Model

```cmd
python scripts/train.py --model graphsage --dataset synthetic --config configs/graphsage_config.yaml
```

**Expected Output:**
```
Using device: cpu
Loading dataset: synthetic
Data: 100 nodes, 200 edges
Model: graphsage
Parameters: 20737
Epoch 10/100, Train Loss: 0.xxxx, Val Loss: 0.xxxx
...
Training completed!
Best validation loss: 0.xxxx
```

**Time**: ~1-2 minutes on CPU

---

## Step 6: Train GAT Model (Optional)

```cmd
python scripts/train.py --model gat --dataset synthetic --config configs/gat_config.yaml
```

**Time**: ~1-2 minutes on CPU

---

## Step 7: Evaluate GraphSAGE Model

```cmd
python scripts/evaluate.py --model graphsage --checkpoint data/checkpoints/graphsage/best_model.pt --dataset synthetic --config configs/graphsage_config.yaml
```

**Expected Output:**
```
=== Classification Metrics ===
auc: 0.xxxx
ap: 0.xxxx
...
=== Ranking Metrics ===
precision@10: 0.xxxx
recall@10: 0.xxxx
...
```

---

## Step 8: Run Streamlit Demo App

```cmd
python -m streamlit run demo/streamlit_app.py
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

The app will automatically open in your browser. If not, copy the Local URL and paste it in your browser.

---

## Alternative: Using Facebook Dataset (Larger Dataset)

If you want to use the Facebook dataset instead of synthetic:

### Step 4a: Download Facebook Dataset

```cmd
python scripts/download_and_prepare.py --dataset facebook --download --preprocess
```

**Note**: This will download ~100MB of data and may take a few minutes.

### Step 5a: Train on Facebook Dataset

```cmd
python scripts/train.py --model graphsage --dataset facebook --config configs/graphsage_config.yaml
```

**Time**: ~5-10 minutes on CPU

---

## Quick Reference: All Commands in Sequence

Copy and paste these commands one by one:

```cmd
REM Step 1: Navigate to project
cd C:\Users\navne\Desktop\Major-Project

REM Step 2: Install dependencies
pip install -r requirements.txt

REM Step 3: Create dataset
python scripts/download_and_prepare.py --dataset synthetic --preprocess

REM Step 4: Train model
python scripts/train.py --model graphsage --dataset synthetic --config configs/graphsage_config.yaml

REM Step 5: Evaluate model
python scripts/evaluate.py --model graphsage --checkpoint data/checkpoints/graphsage/best_model.pt --dataset synthetic --config configs/graphsage_config.yaml

REM Step 6: Run demo
python -m streamlit run demo/streamlit_app.py
```

---

## Troubleshooting

### Issue: "python is not recognized"
**Solution**: Add Python to PATH or use full path:
```cmd
C:\Python313\python.exe --version
```

### Issue: "pip is not recognized"
**Solution**: Use python -m pip:
```cmd
python -m pip install -r requirements.txt
```

### Issue: "ModuleNotFoundError"
**Solution**: Install missing module:
```cmd
pip install <module_name>
```

### Issue: "Streamlit not found"
**Solution**: Use Python module syntax:
```cmd
python -m streamlit run demo/streamlit_app.py
```

### Issue: "CUDA out of memory" (if using GPU)
**Solution**: Reduce batch size in config file or use CPU:
```cmd
python scripts/train.py --model graphsage --dataset synthetic --config configs/graphsage_config.yaml --device cpu
```

---

## Expected File Structure After Running

After completing all steps, you should have:

```
data/
├── processed/
│   ├── synthetic.pt
│   └── synthetic_link_data.pt
└── checkpoints/
    └── graphsage/
        └── best_model.pt
```

---

## Running Jupyter Notebooks (Optional)

If you want to use the notebooks instead:

```cmd
REM Start Jupyter
jupyter notebook

REM Or start JupyterLab
jupyter lab
```

Then open:
- `notebooks/data-preprocessing.ipynb`
- `notebooks/training_graphsage_gat.ipynb`
- `notebooks/evaluation_and_ablation.ipynb`

---

## Time Estimates

- **Synthetic Dataset**: ~30 seconds
- **Training GraphSAGE**: ~1-2 minutes (CPU)
- **Training GAT**: ~1-2 minutes (CPU)
- **Evaluation**: ~10 seconds
- **Demo App**: Starts immediately

**Total Time**: ~5-10 minutes for complete pipeline

---

## Next Steps After Running

1. **Explore the Demo App**: 
   - Select different users
   - Try different models
   - View recommendations and explanations

2. **Train on Facebook Dataset**:
   - Download Facebook dataset
   - Train models for better results

3. **Run Notebooks**:
   - Detailed analysis
   - Baseline comparisons
   - Ablation studies

4. **Experiment with Hyperparameters**:
   - Edit config files in `configs/`
   - Retrain models
   - Compare results

---

## Summary

The complete workflow is:
1. Install dependencies
2. Create dataset
3. Train model
4. Evaluate model
5. Run demo app

All commands should be run in Command Prompt (cmd.exe) or PowerShell, in the project directory.

