# Dataset Explanation Guide
## For Project Presentation

This document provides a comprehensive explanation of the dataset used in the Friend Recommendation project. Use this to explain the dataset to your project guide.

---

## 📊 Dataset Overview

### **Dataset Name:** Facebook Social Circles Dataset (SNAP)

### **Source:** 
- Stanford Network Analysis Project (SNAP)
- Real-world social network data from Facebook
- Publicly available research dataset

### **Dataset Type:**
- **Social Network Graph**
- **Undirected Graph** (friendship is mutual)
- **Node Features** (user profile attributes)
- **Link Prediction Task** (predicting future friendships)

---

## 🎯 What is This Dataset?

This dataset represents a **real social network** where:
- **Nodes (3,963 users)** = People/users in the network
- **Edges (88,156 friendships)** = Existing friend connections
- **Features (576 dimensions)** = User profile attributes/embeddings

### **Real-World Application:**
This is the same type of data structure used by:
- Facebook's friend recommendation system
- LinkedIn's connection suggestions
- Twitter's "Who to follow"
- Instagram's friend suggestions

---

## 📈 Dataset Statistics

### **Scale:**
- **Total Users:** 3,963 people
- **Total Friendships:** 88,156 connections
- **Average Friends per User:** 44.49 friends
- **Network Density:** 1.12% (sparse network - typical for social networks)

### **Network Properties:**
- **Connected:** Yes - All users are reachable from each other
- **Network Diameter:** 8 (maximum distance between any two users)
- **Average Path Length:** 3.78 (average steps to connect any two users)
- **Clustering Coefficient:** 0.6172 (61.72% - friends of friends are likely to be friends)

### **Degree Distribution:**
- **Minimum Friends:** 2
- **Maximum Friends:** 1,034 (very popular user!)
- **Median Friends:** 26
- **Average Friends:** 44.49
- **Distribution:** Power-law (few users have many friends, most have few)

**Key Insight:** This follows the "small world" property - most people are connected through just a few steps.

---

## 🔍 Data Structure

### **1. Graph Structure (`facebook_combined.pt`)**

```python
Data(
    x: [3963, 576]          # Node features (user attributes)
    edge_index: [2, 176312] # Edge connections (undirected, so 2x edges)
    num_nodes: 3963         # Total number of users
)
```

**What each component means:**
- **`x` (Node Features):** 576-dimensional feature vector for each user
  - Represents user profile attributes
  - Normalized values (0 to 1)
  - Includes: profile information, interests, activities, etc.
  
- **`edge_index` (Edges):** List of all friend connections
  - Format: [source_node, target_node]
  - Undirected: If A is friends with B, then B is friends with A
  - Total: 88,156 unique friendships (stored as 176,312 directed edges)

### **2. Link Prediction Data (`facebook_link_data.pt`)**

This file contains the train/validation/test splits for link prediction:

```python
{
    'train_edges': [2, N]    # Training examples (edges to learn from)
    'train_labels': [N]      # 1 = friend, 0 = not friend
    'val_edges': [2, M]      # Validation examples (tuning)
    'val_labels': [M]        # Ground truth labels
    'test_edges': [2, K]     # Test examples (final evaluation)
    'test_labels': [K]       # Ground truth labels
}
```

**Data Split:**
- **Training:** ~60% of edges (used to train the model)
- **Validation:** ~20% of edges (used to tune hyperparameters)
- **Test:** ~20% of edges (used for final evaluation - never seen during training)

**Class Balance:**
- **Positive Examples (Friends):** ~50% (actual friend connections)
- **Negative Examples (Non-friends):** ~50% (random pairs that are not friends)

---

## 🎓 Machine Learning Task

### **Problem:** Link Prediction (Friend Recommendation)

**Goal:** Predict which users are likely to become friends in the future.

### **Input:**
- Graph structure (who is friends with whom)
- Node features (user profile attributes)
- Existing friendships

### **Output:**
- Probability score (0 to 1) for each potential friendship
- Higher score = more likely to be friends

### **Evaluation Metrics:**
1. **AUC (Area Under ROC Curve):** Overall ranking quality
2. **AP (Average Precision):** Precision across all recall levels
3. **Precision@K:** How many of top-K recommendations are correct
4. **NDCG@K:** Ranking quality (most relevant first)

---

## 🔬 Dataset Characteristics

### **1. Network Structure**

**Small World Property:**
- Average path length: 3.78 steps
- This means: Any two random users are connected through ~4 mutual friends
- Real-world example: "Six degrees of separation"

**High Clustering:**
- Clustering coefficient: 0.6172
- Meaning: If A is friends with B and C, there's a 61.72% chance B and C are also friends
- This is typical for social networks (friends of friends tend to be friends)

**Power-Law Degree Distribution:**
- Most users have few friends (median: 26)
- Few users have many friends (max: 1,034)
- This creates "influencers" or "hubs" in the network

### **2. Feature Representation**

**576-Dimensional Feature Vector:**
- Each user has 576 features
- Features represent:
  - Profile attributes (age, location, education, etc.)
  - Interests and preferences
  - Activity patterns
  - Embeddings from profile data

**Feature Properties:**
- Normalized to [0, 1] range
- Sparse (most values are 0)
- Mean: 0.0165 (very sparse - typical for categorical features)

### **3. Data Quality**

**Advantages:**
- ✅ Real-world data (not synthetic)
- ✅ Large enough for meaningful results (3,963 users)
- ✅ Well-structured and clean
- ✅ Balanced train/val/test splits
- ✅ Includes both graph structure and node features

**Challenges:**
- ⚠️ Sparse network (only 1.12% of possible connections exist)
- ⚠️ Imbalanced degree distribution (some users have many more connections)
- ⚠️ High-dimensional features (576 dimensions)

---

## 📊 How the Dataset Was Created

### **Source:**
1. **Original Data:** Facebook ego networks from SNAP
2. **Processing:** Combined multiple ego networks into one large graph
3. **Feature Extraction:** Extracted 576-dimensional feature vectors from user profiles
4. **Link Prediction Split:** Randomly split edges into train/val/test sets

### **Preprocessing Steps:**
1. Downloaded raw Facebook ego network data
2. Combined multiple ego networks (merged overlapping users)
3. Extracted and normalized node features
4. Created train/validation/test splits (60/20/20)
5. Generated negative samples (non-friend pairs) for training

---

## 🎯 Why This Dataset?

### **1. Real-World Relevance**
- Represents actual social network structure
- Same data type used by major social media platforms
- Practical application: Friend recommendation systems

### **2. Appropriate Scale**
- Large enough: 3,963 users, 88,156 edges
- Small enough: Can train models efficiently
- Good balance for research and demonstration

### **3. Rich Features**
- 576-dimensional feature vectors
- Includes both structure (graph) and attributes (features)
- Enables learning from multiple information sources

### **4. Standard Benchmark**
- Widely used in research
- Allows comparison with other methods
- Well-documented and understood

---

## 💡 Key Insights for Presentation

### **What to Emphasize:**

1. **Real-World Data:**
   - "This is real Facebook social network data, not synthetic"
   - "Same type of data structure used by actual social media platforms"

2. **Appropriate Scale:**
   - "3,963 users with 88,156 friendships - large enough to be meaningful, small enough to train efficiently"
   - "Average of 44 friends per user - realistic social network"

3. **Rich Information:**
   - "Each user has 576 features representing their profile attributes"
   - "We use both network structure AND user features for recommendations"

4. **Well-Structured:**
   - "Properly split into train/validation/test sets"
   - "Balanced positive and negative examples"
   - "Follows standard machine learning practices"

5. **Network Properties:**
   - "Small world network - average 3.78 steps between any two users"
   - "High clustering - friends of friends are likely to be friends"
   - "Power-law distribution - typical of real social networks"

---

## 📝 Presentation Script

### **Opening:**
"Let me explain the dataset we're using for this friend recommendation project."

### **Dataset Overview:**
"We're using the Facebook Social Circles dataset from Stanford's SNAP project. This is real-world social network data with 3,963 users and 88,156 friendships."

### **Structure:**
"The dataset consists of:
- A graph where nodes are users and edges are friendships
- 576-dimensional feature vectors for each user representing their profile attributes
- Train/validation/test splits for link prediction"

### **Key Statistics:**
"Key characteristics:
- Average of 44 friends per user
- Network is connected - all users are reachable
- Average path length of 3.78 - typical 'small world' property
- High clustering coefficient of 0.62 - friends of friends are likely to be friends"

### **Task:**
"Our goal is to predict which users are likely to become friends - this is a link prediction problem. We use both the network structure and user features to make these predictions."

### **Why This Dataset:**
"This dataset is ideal because:
1. It's real-world data, not synthetic
2. It's the same structure used by actual social media platforms
3. It has both graph structure and rich node features
4. It's a standard benchmark in the research community"

---

## 🔍 Technical Details (If Asked)

### **Data Format:**
- **PyTorch Geometric (PyG) Data object**
- Standard format for graph neural networks
- Efficient for batch processing and GPU computation

### **Feature Engineering:**
- Features extracted from user profiles
- Normalized to [0, 1] range
- Sparse representation (most values are 0)

### **Negative Sampling:**
- For each positive edge (friend), we sample negative edges (non-friends)
- Maintains balanced dataset for training
- Random sampling from non-existing edges

### **Data Splits:**
- **Temporal Split:** Not used (we use random split)
- **Random Split:** 60% train, 20% validation, 20% test
- Ensures no data leakage between splits

---

## ❓ Common Questions & Answers

### **Q: Why not use a larger dataset?**
**A:** This dataset is large enough to demonstrate the approach effectively while being manageable for training and evaluation. Larger datasets would require more computational resources.

### **Q: Are the features real or synthetic?**
**A:** The features are extracted from real user profile data in the original Facebook dataset. They represent actual user attributes.

### **Q: How do you know if a recommendation is correct?**
**A:** We use the test set which contains actual friend connections. If our model predicts a high score for a pair that are actually friends, that's a correct recommendation.

### **Q: Why is the network sparse?**
**A:** Real social networks are naturally sparse - people don't connect to everyone. Only 1.12% of possible connections exist, which is typical for social networks.

### **Q: What if users have very different numbers of friends?**
**A:** This is handled by the graph neural network models, which can learn from nodes with varying degrees. The models are designed to work with this power-law distribution.

---

## 📚 Additional Resources

- **SNAP Dataset Page:** https://snap.stanford.edu/data/ego-Facebook.html
- **PyTorch Geometric:** https://pytorch-geometric.readthedocs.io/
- **Link Prediction:** Standard task in graph machine learning

---

## ✅ Summary Checklist for Presentation

- [ ] Mention it's real-world Facebook data
- [ ] State the scale (3,963 users, 88,156 edges)
- [ ] Explain graph structure (nodes = users, edges = friendships)
- [ ] Mention 576-dimensional features
- [ ] Highlight network properties (small world, clustering)
- [ ] Explain the link prediction task
- [ ] Mention train/val/test splits
- [ ] Emphasize practical application (friend recommendation)
- [ ] Be ready to answer questions about data quality and preprocessing

---

**Good luck with your presentation!** 🎉

