# Why Path Evidence is Always 2-Hop

## The Observation

**Path Evidence:** Almost always shows "2-hop" for recommended users  
**Question:** Why is it always 2-hop, never 1-hop or 3+ hop?

---

## Explanation

### **1. 1-Hop Paths Are Excluded**

**1-hop = Direct Friends:**
- If two users are directly connected (1-hop), they're already friends
- The recommendation system **excludes existing friends** from candidates
- Code: `candidate_nodes = [i for i in range(num_nodes) if i != user_id and i not in existing_friends]`
- **Result:** Recommendations can NEVER be 1-hop

---

### **2. 2-Hop Paths Are Most Common**

**2-hop = Friends of Friends:**
- Most recommended users are exactly 2 steps away
- This is the **most natural recommendation pattern**

**Why 2-hop is dominant:**

#### **A. Mutual Friends Pattern**
- 2-hop users share mutual friends with you
- Example: You → Friend A → Recommended User
- Strong signal for friendship (triadic closure principle)

#### **B. Graph Neural Network Learning**
- GNN models learn that 2-hop connections are strong predictors
- The model gives high scores to users 2 steps away
- Most top-K recommendations end up being 2-hop

#### **C. Social Network Property**
- Real social networks exhibit "triadic closure"
- If A is friends with B, and B is friends with C, then A and C are likely to become friends
- This creates many 2-hop recommendations

---

### **3. 3+ Hop Paths Are Rare in Recommendations**

**Why longer paths don't appear:**

#### **A. Weaker Signals**
- 3-hop connections are weaker
- Longer paths = less likely to be friends
- Model gives lower confidence scores to distant users

#### **B. Lower Ranking**
- Even if 3-hop users are predicted, they rank lower
- Top-K recommendations (K=10, 20, etc.) mostly contain 2-hop users
- 3+ hop users are pushed out of top-K

#### **C. Network Structure**
- Most reachable users in social networks are 2-3 hops away
- But recommendations prioritize the strongest signals (2-hop)

---

## Mathematical Explanation

### **Path Length Distribution in Social Networks:**

- **1-hop:** Direct friends (excluded from recommendations)
- **2-hop:** Friends of friends (most common in recommendations) ⭐
- **3-hop:** Friends of friends of friends (rare in top-K)
- **4+ hop:** Very distant (almost never recommended)

### **Why 2-Hop Dominates:**

```
P(friendship | 2-hop) > P(friendship | 3-hop) > P(friendship | 4-hop)
```

The probability of friendship decreases with path length, so 2-hop users get the highest recommendation scores.

---

## Code Evidence

From `compute_shortest_path()`:
- Computes shortest path between two users
- If path = 1 → They're friends → Excluded from candidates
- If path = 2 → Most common in recommendations
- If path = 3+ → Usually rank too low for top-K

---

## Real-World Analogy

**1-hop:** Your direct friends (already connected) ❌  
**2-hop:** Friends of your friends (most likely recommendations) ✅  
**3-hop:** Friends of friends of friends (less likely) ⚠️  
**4+ hop:** Distant connections (very unlikely) ❌

**Example:**
- You know Alice (1-hop)
- Alice knows Bob (2-hop from you) ← Most likely recommendation
- Bob knows Charlie (3-hop from you) ← Less likely
- Charlie knows Dave (4-hop from you) ← Very unlikely

---

## Is This Normal?

**YES!** This is completely normal and expected behavior:

1. ✅ **1-hop excluded** - Can't recommend existing friends
2. ✅ **2-hop dominant** - Friends of friends are the best recommendations
3. ✅ **3+ hop rare** - Longer paths have weaker signals

---

## What If You See 3-Hop or More?

If you occasionally see 3-hop or longer paths in recommendations:

- **It's still valid** - Just means the model found a strong signal despite distance
- **Rare occurrence** - Most recommendations will still be 2-hop
- **Special case** - Might indicate strong profile similarity or other factors

---

## Summary

**Why path evidence is always 2-hop:**

1. ✅ **1-hop excluded** - Direct friends aren't in candidate set
2. ✅ **2-hop dominant** - Friends of friends are the strongest recommendations
3. ✅ **3+ hop rare** - Longer paths rank lower, rarely appear in top-K
4. ✅ **Expected behavior** - This is normal for friend recommendation systems

**This is a feature, not a bug!** The system is correctly identifying the most likely recommendations (friends of friends).

