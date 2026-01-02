# Why Recommendations Are Asymmetric

## The Observation

**User 1043 → User 1254:** Recommended at rank 4 ✅  
**User 1254 → User 1043:** NOT recommended ❌

**Question:** Why is the recommendation not bidirectional?

---

## Possible Reasons

### **Reason 1: They're Already Friends (Most Likely)**

If users 1043 and 1254 are **already friends**, then:
- When recommending for user 1254, user 1043 is **excluded from candidates**
- The system only recommends **non-friends**
- Result: 1043 won't appear in 1254's recommendations

**How to check:** Verify if there's an edge between 1043 and 1254 in the graph.

---

### **Reason 2: Different Candidate Sets**

Even if they're not friends, they might have:
- **Different numbers of non-friends** (different candidate pool sizes)
- **Different other strong candidates** competing for top-K positions
- User 1254 might have many other better candidates, pushing 1043 out of top-K

**Example:**
- User 1043: 1000 non-friends → 1254 ranks #4
- User 1254: 2000 non-friends → 1043 ranks #150 (outside top-10)

---

### **Reason 3: Asymmetric Prediction Scores**

Link prediction models can produce **different scores** depending on direction:
- Score(1043 → 1254) might be 0.9817 (high, rank #4)
- Score(1254 → 1043) might be 0.6500 (lower, doesn't make top-K)

**Why asymmetric?**
- Different local graph neighborhoods
- Different mutual friend patterns when viewed from each user
- Feature aggregation differs depending on starting node
- GraphSAGE/GAT aggregation is directional in how it processes neighborhoods

---

### **Reason 4: Different Graph Neighborhood Context**

From user 1043's perspective:
- 1254 might have strong signals (mutual friends, paths, etc.)
- Clear recommendation signal

From user 1254's perspective:
- 1043 might be less prominent in 1254's local neighborhood
- Other candidates have stronger signals
- 1043 gets ranked lower

---

## Most Likely Explanation

**They're probably already friends!**

If 1043 → 1254 is recommended, but 1254 → 1043 is not:
- **Most common reason:** They're already connected (existing edge)
- When recommending for 1254, the system excludes existing friends
- 1043 is filtered out before ranking

---

## How to Verify

Check if there's an edge between users 1043 and 1254:
```python
# Check if edge exists
edge_exists = (edge_index[0] == 1043) & (edge_index[1] == 1254) | 
              (edge_index[0] == 1254) & (edge_index[1] == 1043)
```

If edge exists → They're friends → 1043 won't be in 1254's recommendations  
If no edge → Asymmetric scores or different candidate pools

---

## Why This Makes Sense

**Real-world analogy:**
- You might think: "I should recommend my friend X to Y"
- But if X and Y are already friends, there's no need to recommend!
- The system correctly excludes existing connections from recommendations

---

## Summary

**Most likely:** Users 1043 and 1254 are already friends, so 1043 is excluded from 1254's candidate set.

**Alternative:** If they're not friends, asymmetric scores or different candidate pools cause 1043 to rank lower when recommending for 1254.

**This is normal behavior** - recommendations are not required to be symmetric!

