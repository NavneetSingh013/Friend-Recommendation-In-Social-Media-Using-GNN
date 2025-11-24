# Accessing the Demo App in Browser

## ✅ No Need to Rerun Training!

All your models and data are already saved. You just need to start the demo app.

---

## 🚀 Quick Start

### Option 1: Streamlit Demo App (Recommended)

Run this command:
```cmd
python -m streamlit run demo/streamlit_app.py
```

**What you'll see:**
- The app will automatically open in your browser
- URL: `http://localhost:8501`
- If it doesn't open automatically, copy the URL from terminal and paste in browser

**Features:**
- Select model (GraphSAGE, GAT, SEAL)
- Choose user ID (0-99 for synthetic, 0-3962 for Facebook)
- Get top-K friend recommendations
- See explanations (mutual friends, shared groups, etc.)
- Visualize network graph

### Option 2: Jupyter Notebooks

If Jupyter is already running:
1. Check terminal for the URL (usually `http://localhost:8888`)
2. Copy the token from terminal
3. Open URL in browser
4. Navigate to `notebooks/` folder

If Jupyter is not running:
```cmd
jupyter notebook
```

---

## 📊 What Models Are Available

### Trained Models (Ready to Use):

1. **GraphSAGE - Synthetic Dataset**
   - Checkpoint: `data/checkpoints/graphsage/best_model.pt`
   - Dataset: Synthetic (100 nodes)
   - User IDs: 0-99

2. **GAT - Synthetic Dataset**
   - Checkpoint: `data/checkpoints/gat/best_model.pt`
   - Dataset: Synthetic (100 nodes)
   - User IDs: 0-99

3. **GraphSAGE - Facebook Dataset** ⭐ (Best Results!)
   - Checkpoint: `data/checkpoints/graphsage/best_model.pt` (overwritten with Facebook)
   - Dataset: Facebook (3,963 nodes)
   - User IDs: 0-3962
   - **AUC: 0.9874** (Excellent!)

---

## 🎯 Using the Demo App

### Step-by-Step:

1. **Start the app** (if not already running):
   ```cmd
   python -m streamlit run demo/streamlit_app.py
   ```

2. **In the browser:**
   - Select **Model**: Choose GraphSAGE, GAT, or SEAL
   - Select **Checkpoint Path**: Default path should work
   - Enter **User ID**: 
     - For Synthetic: 0-99
     - For Facebook: 0-3962
   - Set **Number of Recommendations (K)**: 5-50

3. **Click "Get Recommendations"**

4. **View Results:**
   - Top-K recommended friends
   - Confidence scores
   - Mutual friends count
   - Shared groups
   - Profile similarity
   - Path evidence
   - Network visualization

---

## 🔍 Using Facebook Dataset

The demo app will automatically detect if Facebook dataset is available. When you select a model checkpoint trained on Facebook:

- **More users**: 3,963 nodes (vs 100 for synthetic)
- **Better recommendations**: 98.74% AUC
- **More features**: Real social network data
- **Better explanations**: More mutual friends, richer groups

---

## 🐛 Troubleshooting

### If the app doesn't start:
```cmd
# Check if port 8501 is in use
netstat -ano | findstr :8501

# Use different port
python -m streamlit run demo/streamlit_app.py --server.port 8502
```

### If model checkpoint not found:
- Check that training completed successfully
- Verify checkpoint path in sidebar
- Default path: `data/checkpoints/graphsage/best_model.pt`

### If you want to use Facebook dataset:
1. Make sure Facebook dataset is processed (already done ✓)
2. The demo app should automatically load Facebook data if available
3. Select a checkpoint trained on Facebook dataset

### If you see "Using untrained model":
- This is okay - app will work but predictions will be random
- For better results, use a trained model checkpoint
- Make sure checkpoint path is correct

---

## 📝 Example Usage

### Example 1: Synthetic Dataset
```
Model: GraphSAGE
Checkpoint: data/checkpoints/graphsage/best_model.pt
User ID: 0
K: 10
```

### Example 2: Facebook Dataset (Better Results!)
```
Model: GraphSAGE
Checkpoint: data/checkpoints/graphsage/best_model.pt
User ID: 100
K: 10
```

The app will automatically detect which dataset to use based on available files.

---

## 💡 Tips

1. **Start with small K**: Try K=5 or 10 first
2. **Try different users**: Each user has different connections
3. **Compare models**: Switch between GraphSAGE and GAT
4. **Check explanations**: Click on recommendations to see details
5. **View visualization**: Scroll down to see network graph

---

## 🎉 Summary

**You don't need to rerun training!** Everything is ready:
- ✅ Models trained and saved
- ✅ Datasets processed
- ✅ Just start the demo app

**Command to start:**
```cmd
python -m streamlit run demo/streamlit_app.py
```

Then open `http://localhost:8501` in your browser!

