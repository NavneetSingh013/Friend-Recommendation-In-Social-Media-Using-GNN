# Files That Calculate and Evaluate Metrics for Different Models

This document shows exactly which files handle metric calculation and evaluation for different models.

---

## 📊 Core Metrics Computation

### **`src/evaluation/metrics.py`** ⭐ PRIMARY FILE

**Location:** `src/evaluation/metrics.py`

**Purpose:** Core functions that calculate all metrics

**Key Functions:**

1. **`compute_metrics(predictions, labels)`** - Classification Metrics
   - Computes: AUC, AP, Accuracy, Precision, Recall, F1
   - Uses: `sklearn.metrics.roc_auc_score`, `average_precision_score`
   - Returns: Dictionary with all classification metrics

2. **`compute_ranking_metrics(predictions, labels, k_values)`** - Ranking Metrics
   - Computes: Precision@K, Recall@K, NDCG@K, MAP@K
   - K values: [5, 10, 20, 50] (default)
   - Returns: Dictionary with all ranking metrics

3. **`get_top_k_recommendations(model, data, user_id, candidate_nodes, k)`**
   - Gets top-K recommendations for a user
   - Handles different model types (GraphSAGE, GAT, SEAL)
   - Returns: Top-K node IDs and scores

**This is the core file - all metric calculations happen here!**

---

## 🎯 Evaluation Scripts

### **1. `scripts/evaluate.py`** - Single Model Evaluation

**Location:** `scripts/evaluate.py`

**Purpose:** Evaluate a single model (GraphSAGE, GAT, or SEAL)

**What it does:**
1. Loads dataset and link prediction data
2. Creates model from config
3. Loads model checkpoint
4. Generates predictions on test set
5. **Calls `compute_metrics()` and `compute_ranking_metrics()`** from `metrics.py`
6. Prints results

**Usage:**
```bash
python scripts/evaluate.py --model graphsage --checkpoint data/checkpoints/graphsage/best_model.pt --config configs/graphsage_config.yaml
```

**Key Function:**
- `main()` - Main evaluation function
- Uses: `compute_metrics()`, `compute_ranking_metrics()` from `src.evaluation.metrics`

---

### **2. `scripts/evaluate_all_models.py`** ⭐ COMPREHENSIVE EVALUATION

**Location:** `scripts/evaluate_all_models.py`

**Purpose:** Evaluate ALL available models and compare them

**What it does:**
1. Tries to load all models (GraphSAGE, GAT, SEAL)
2. For each model:
   - Loads checkpoint
   - Generates predictions
   - **Calls `compute_metrics()` and `compute_ranking_metrics()`**
   - Collects all metrics
3. Displays comparison tables
4. Saves results to CSV: `model_evaluation_results.csv`

**Key Function:**
- `evaluate_model(model_name, model, data, link_data, device)` 
  - Calls: `compute_metrics()` and `compute_ranking_metrics()`
  - Returns: Combined dictionary with all metrics

**This is the script you ran earlier to see all model metrics!**

---

## 📓 Notebooks

### **3. `notebooks/baselines.ipynb`** - Baseline Methods

**Location:** `notebooks/baselines.ipynb`

**Purpose:** Evaluate baseline methods (heuristics, Node2Vec, Matrix Factorization)

**What it does:**
1. Loads data
2. Computes/evaluates heuristics (Common Neighbors, Jaccard, etc.)
3. Trains Node2Vec baseline
4. Trains Matrix Factorization baseline
5. **Calls `compute_metrics()` and `compute_ranking_metrics()`** for each baseline
6. Compares baseline results

**Uses:**
- `compute_metrics()` from `src.evaluation.metrics`
- `compute_ranking_metrics()` from `src.evaluation.metrics`

---

### **4. `notebooks/evaluation_and_ablation.ipynb`** - Comprehensive Evaluation

**Location:** `notebooks/evaluation_and_ablation.ipynb`

**Purpose:** Comprehensive evaluation and ablation study (currently incomplete)

**Note:** This notebook only loads data currently - evaluation code needs to be added.

**Would use:**
- `compute_metrics()` from `src.evaluation.metrics`
- `compute_ranking_metrics()` from `src.evaluation.metrics`

---

### **5. `notebooks/training_graphsage_gat.ipynb`** - Training & Evaluation

**Location:** `notebooks/training_graphsage_gat.ipynb`

**Purpose:** Train GraphSAGE/GAT and evaluate during/after training

**What it does:**
1. Trains models
2. Evaluates during training (validation metrics)
3. **Calls `compute_metrics()` and `compute_ranking_metrics()`** after training
4. Generates recommendations and evaluates them

**Uses:**
- `compute_metrics()` from `src.evaluation.metrics`
- `compute_ranking_metrics()` from `src.evaluation.metrics`
- `get_top_k_recommendations()` from `src.evaluation.metrics`

---

### **6. `notebooks/training_seal.ipynb`** - SEAL Training & Evaluation

**Location:** `notebooks/training_seal.ipynb`

**Purpose:** Train SEAL model and evaluate

**Uses:**
- `compute_metrics()` from `src.evaluation.metrics`
- `compute_ranking_metrics()` from `src.evaluation.metrics`

---

## 🔄 Training Scripts (Also Evaluate)

### **7. `src/training/trainer.py`** - Training Loop with Validation

**Location:** `src/training/trainer.py`

**Purpose:** Training loop that also evaluates during training

**Key Method:**
- `evaluate()` - Evaluates model on validation/test set
- Uses metrics to track validation performance
- Used during training for early stopping

---

## 📈 File Hierarchy

```
Metric Calculation Flow:

┌─────────────────────────────────────────┐
│  src/evaluation/metrics.py              │  ← CORE: All metric functions
│  - compute_metrics()                    │
│  - compute_ranking_metrics()            │
│  - get_top_k_recommendations()          │
└─────────────────────────────────────────┘
              ↑              ↑              ↑
              │              │              │
    ┌─────────┘              │              └─────────┐
    │                        │                        │
┌───┴───────────┐   ┌────────┴────────┐   ┌──────────┴─────────┐
│ evaluate.py   │   │evaluate_all_    │   │ notebooks/         │
│               │   │models.py        │   │ baselines.ipynb    │
│ (single model)│   │                 │   │ training_*.ipynb   │
│               │   │ (all models)    │   │ evaluation_*.ipynb │
└───────────────┘   └─────────────────┘   └────────────────────┘
```

---

## 📝 Summary Table

| File | Purpose | Models Evaluated | Key Functions |
|------|---------|------------------|---------------|
| **`src/evaluation/metrics.py`** | Core metric computation | All models | `compute_metrics()`, `compute_ranking_metrics()` |
| **`scripts/evaluate.py`** | Single model evaluation | GraphSAGE, GAT, or SEAL | `main()` |
| **`scripts/evaluate_all_models.py`** | All models comparison | GraphSAGE, GAT, SEAL | `evaluate_model()`, `main()` |
| `notebooks/baselines.ipynb` | Baseline evaluation | Heuristics, Node2Vec, MF | Uses metrics.py |
| `notebooks/training_*.ipynb` | Training + evaluation | GraphSAGE, GAT, SEAL | Uses metrics.py |
| `src/training/trainer.py` | Training with validation | All models | `evaluate()` |

---

## 🎯 Quick Reference

**To calculate metrics for any model:**
1. Import: `from src.evaluation.metrics import compute_metrics, compute_ranking_metrics`
2. Get predictions from model
3. Call: `metrics = compute_metrics(predictions, labels)`
4. Call: `ranking = compute_ranking_metrics(predictions, labels)`

**To evaluate all models:**
- Run: `python scripts/evaluate_all_models.py`
- This uses `src/evaluation/metrics.py` internally

**Core file (everything uses this):**
- `src/evaluation/metrics.py` - All metric calculations happen here!

---

## 🔍 Where Metrics Are Actually Computed

**All metrics are computed in:**
- **`src/evaluation/metrics.py`** - This is the single source of truth

**All evaluation scripts use:**
- `from src.evaluation.metrics import compute_metrics, compute_ranking_metrics`

**The metrics.py file contains:**
- ✅ Classification metrics: AUC, AP, Accuracy, Precision, Recall, F1
- ✅ Ranking metrics: Precision@K, Recall@K, NDCG@K, MAP@K
- ✅ Recommendation generation: `get_top_k_recommendations()`

---

## 📂 File Locations

```
Friend-Recommendation-In-Social-Media-Using-GNN/
│
├── src/evaluation/
│   └── metrics.py                    ← CORE: All metric calculations
│
├── scripts/
│   ├── evaluate.py                   ← Single model evaluation
│   └── evaluate_all_models.py        ← All models evaluation ⭐
│
└── notebooks/
    ├── baselines.ipynb               ← Baseline methods evaluation
    ├── training_graphsage_gat.ipynb  ← GraphSAGE/GAT training & eval
    ├── training_seal.ipynb           ← SEAL training & eval
    └── evaluation_and_ablation.ipynb ← Comprehensive evaluation
```

---

**Bottom Line:** `src/evaluation/metrics.py` is where ALL metric calculations happen. All other files import and use these functions to evaluate models.

