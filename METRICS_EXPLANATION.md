# Comprehensive Guide to Evaluation Metrics

This document explains all metrics used to evaluate the friend recommendation models.

---

## 📊 Classification Metrics

### 1. AUC (Area Under ROC Curve)
**Range:** 0.0 to 1.0 (higher is better)  
**Ideal Value:** 1.0 (perfect classifier)

**What it measures:**
- The probability that the model will rank a randomly chosen positive example (actual friend connection) higher than a randomly chosen negative example (non-connection).
- Measures the model's ability to distinguish between positive and negative examples.

**Interpretation:**
- **0.9-1.0**: Excellent - Model can reliably distinguish friends from non-friends
- **0.8-0.9**: Good - Model performs well at distinguishing
- **0.7-0.8**: Fair - Model is moderately good
- **0.5-0.7**: Poor - Model barely better than random guessing
- **0.0-0.5**: Very Poor - Model is worse than random (rare)

**Your Results:**
- GraphSAGE: **0.9877** (Excellent! 98.77% accuracy in ranking)
- GAT: **0.9867** (Excellent! 98.67% accuracy in ranking)

---

### 2. AP (Average Precision)
**Range:** 0.0 to 1.0 (higher is better)  
**Ideal Value:** 1.0

**What it measures:**
- Summarizes the Precision-Recall curve into a single value
- Better metric than AUC for imbalanced datasets (where positives are rare)
- Weighted average of precision scores at different recall levels

**Interpretation:**
- **0.9-1.0**: Excellent - Model maintains high precision across all recall levels
- **0.8-0.9**: Good - Model performs well across recall levels
- **0.7-0.8**: Fair - Moderate performance
- **< 0.7**: Poor - Low precision or recall

**Your Results:**
- GraphSAGE: **0.9840** (Excellent! 98.40% average precision)
- GAT: **0.9813** (Excellent! 98.13% average precision)

---

### 3. Accuracy
**Range:** 0.0 to 1.0 (higher is better)  
**Ideal Value:** 1.0

**What it measures:**
- The fraction of correct predictions (both true positives and true negatives) out of all predictions
- Formula: (TP + TN) / (TP + TN + FP + FN)

**Interpretation:**
- **> 0.95**: Excellent - 95%+ of predictions are correct
- **0.90-0.95**: Very Good - 90-95% correct
- **0.80-0.90**: Good - 80-90% correct
- **< 0.80**: Fair to Poor

**Your Results:**
- GraphSAGE: **0.9601** (Excellent! 96.01% of predictions are correct)
- GAT: **0.9563** (Excellent! 95.63% of predictions are correct)

**Note:** Accuracy can be misleading with imbalanced datasets. AUC and AP are often more reliable.

---

### 4. Precision
**Range:** 0.0 to 1.0 (higher is better)  
**Ideal Value:** 1.0

**What it measures:**
- The fraction of predicted positive examples that are actually positive
- Formula: TP / (TP + FP)
- Answers: "Of all friend recommendations made, how many were actually friends?"

**Interpretation:**
- **> 0.90**: Excellent - 90%+ of recommendations are correct
- **0.80-0.90**: Very Good
- **0.70-0.80**: Good
- **< 0.70**: Fair to Poor

**Your Results:**
- GraphSAGE: **0.9469** (Excellent! 94.69% of recommendations are correct)
- GAT: **0.9303** (Excellent! 93.03% of recommendations are correct)

---

### 5. Recall
**Range:** 0.0 to 1.0 (higher is better)  
**Ideal Value:** 1.0

**What it measures:**
- The fraction of actual positive examples that were correctly identified
- Formula: TP / (TP + FN)
- Answers: "Of all actual friends, how many did we find?"

**Interpretation:**
- **> 0.90**: Excellent - Model finds 90%+ of actual friends
- **0.80-0.90**: Very Good
- **0.70-0.80**: Good
- **< 0.70**: Fair to Poor

**Your Results:**
- GraphSAGE: **0.9748** (Excellent! Finds 97.48% of actual friends)
- GAT: **0.9865** (Excellent! Finds 98.65% of actual friends - slightly better than GraphSAGE)

---

### 6. F1 Score
**Range:** 0.0 to 1.0 (higher is better)  
**Ideal Value:** 1.0

**What it measures:**
- Harmonic mean of Precision and Recall
- Formula: 2 × (Precision × Recall) / (Precision + Recall)
- Balances both precision and recall in a single metric

**Interpretation:**
- **> 0.90**: Excellent balance between precision and recall
- **0.80-0.90**: Very Good balance
- **0.70-0.80**: Good balance
- **< 0.70**: Fair to Poor

**Your Results:**
- GraphSAGE: **0.9607** (Excellent! Great balance)
- GAT: **0.9576** (Excellent! Great balance)

---

## 📈 Ranking Metrics

These metrics evaluate how well the model ranks recommendations (most relevant first).

### 7. Precision@K
**Range:** 0.0 to 1.0 (higher is better)  
**Ideal Value:** 1.0

**What it measures:**
- Fraction of top-K recommendations that are actually positive (friends)
- Formula: (Number of correct recommendations in top-K) / K
- Answers: "Of the top K recommendations, how many are correct?"

**K values:** Usually 5, 10, 20, 50 (number of recommendations shown)

**Interpretation:**
- **1.0**: Perfect - All top-K recommendations are correct
- **0.8-1.0**: Excellent - 80%+ of top-K are correct
- **0.6-0.8**: Good
- **< 0.6**: Fair to Poor

**Your Results (Facebook Dataset):**
- GraphSAGE: **1.0 @ all K** (Perfect! All recommendations are correct)
- GAT: **1.0 @ all K** (Perfect! All recommendations are correct)

**What this means:** When you ask for top 10 friend recommendations, ALL 10 are actually friends. This is exceptional performance!

---

### 8. Recall@K
**Range:** 0.0 to 1.0 (higher is better)  
**Ideal Value:** 1.0

**What it measures:**
- Fraction of all actual friends found in the top-K recommendations
- Formula: (Number of friends found in top-K) / (Total number of actual friends)
- Answers: "How many of the actual friends appear in the top K recommendations?"

**Interpretation:**
- Higher is always better
- Depends on how many total friends exist
- Small value doesn't mean poor performance if there are many friends

**Your Results:**
- GraphSAGE: **0.000756 @ K=10** (Very small because there are many total friends)
- GAT: **0.000756 @ K=10** (Same)

**Why it's small:** With thousands of actual friends in the test set, finding 10 in top-10 recommendations gives low recall, but this is normal for large networks.

---

### 9. NDCG@K (Normalized Discounted Cumulative Gain)
**Range:** 0.0 to 1.0 (higher is better)  
**Ideal Value:** 1.0

**What it measures:**
- Position-weighted ranking quality
- Gives higher scores to relevant items (friends) appearing earlier in the ranking
- Accounts for the order of recommendations (position 1 matters more than position 10)

**How it works:**
1. DCG (Discounted Cumulative Gain): Scores items by relevance, with higher scores for items ranked higher
2. IDCG (Ideal DCG): Best possible DCG if items were perfectly ranked
3. NDCG = DCG / IDCG (normalized to 0-1)

**Interpretation:**
- **1.0**: Perfect ranking - All relevant items are at the top
- **0.8-1.0**: Excellent ranking quality
- **0.6-0.8**: Good ranking
- **< 0.6**: Fair to Poor

**Your Results:**
- GraphSAGE: **1.0 @ all K** (Perfect ranking!)
- GAT: **1.0 @ all K** (Perfect ranking!)

**What this means:** Not only are all recommendations correct, but they're also perfectly ordered by relevance.

---

### 10. MAP@K (Mean Average Precision at K)
**Range:** 0.0 to 1.0 (higher is better)  
**Ideal Value:** 1.0

**What it measures:**
- Average precision computed over the top-K results
- Combines both precision and ranking quality
- Higher scores when relevant items (friends) appear earlier in the ranking

**How it works:**
1. For each relevant item at position i, compute precision@i
2. Average these precision scores
3. This gives more weight to relevant items ranked higher

**Interpretation:**
- **1.0**: Perfect - All relevant items are at the top with perfect precision
- **0.8-1.0**: Excellent
- **0.6-0.8**: Good
- **< 0.6**: Fair to Poor

**Your Results:**
- GraphSAGE: **1.0 @ K=10** (Perfect!)
- GAT: **1.0 @ K=10** (Perfect!)

---

### 11. MAP (Mean Average Precision)
**Range:** 0.0 to 1.0 (higher is better)  
**Ideal Value:** 1.0

**What it measures:**
- Average precision across all positions where relevant items appear
- Similar to AP but computed for ranking scenarios
- Combines precision and recall in ranking context

**Your Results:**
- GraphSAGE: **0.9840** (Excellent!)
- GAT: **0.9813** (Excellent!)

---

## 🎯 Model Performance Summary

### GraphSAGE Performance (Facebook Dataset)
- **AUC:** 0.9877 (Excellent - 98.77% ranking accuracy)
- **AP:** 0.9840 (Excellent - 98.40% average precision)
- **Accuracy:** 0.9601 (Excellent - 96.01% correct predictions)
- **Precision:** 0.9469 (Excellent - 94.69% of recommendations are correct)
- **Recall:** 0.9748 (Excellent - Finds 97.48% of actual friends)
- **F1:** 0.9607 (Excellent balance)
- **Precision@10:** 1.0 (Perfect - All top 10 recommendations are correct)
- **NDCG@10:** 1.0 (Perfect ranking)

**Overall Assessment:** **EXCELLENT** - Model is highly accurate and reliable for friend recommendations.

---

### GAT Performance (Facebook Dataset)
- **AUC:** 0.9867 (Excellent - 98.67% ranking accuracy)
- **AP:** 0.9813 (Excellent - 98.13% average precision)
- **Accuracy:** 0.9563 (Excellent - 95.63% correct predictions)
- **Precision:** 0.9303 (Excellent - 93.03% of recommendations are correct)
- **Recall:** 0.9865 (Excellent - Finds 98.65% of actual friends)
- **F1:** 0.9576 (Excellent balance)
- **Precision@10:** 1.0 (Perfect - All top 10 recommendations are correct)
- **NDCG@10:** 1.0 (Perfect ranking)

**Overall Assessment:** **EXCELLENT** - Model is highly accurate and reliable for friend recommendations.

**Note:** GAT has slightly better recall (finds more actual friends) but slightly lower precision than GraphSAGE.

---

## 🔍 Understanding the Results

### What These Metrics Tell Us:

1. **Both models are highly accurate** (96%+ accuracy)
2. **Perfect precision at top-K** - Every recommendation in the top 10 is actually a friend
3. **Excellent ranking quality** - Recommendations are perfectly ordered by relevance
4. **Models are well-trained and reliable** for real-world use

### Trade-offs:

- **GraphSAGE:** Slightly better precision (fewer false positives), slightly lower recall
- **GAT:** Slightly better recall (finds more actual friends), slightly lower precision

Both models are production-ready and perform exceptionally well!

---

## 📝 Metric Selection Guide

**For Friend Recommendations, prioritize:**
1. **Precision@K** - Most important - Users want accurate recommendations
2. **AUC/AP** - Overall model quality
3. **NDCG@K** - Ranking order matters (most relevant first)
4. **Recall** - Less critical (there are many possible friends)

Your models excel in all these areas!

