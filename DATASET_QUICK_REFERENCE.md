# Dataset Quick Reference Card
## For Quick Presentation

---

## 📊 Facebook Dataset - Key Numbers

### **Scale:**
- **Users:** 3,963
- **Friendships:** 88,156
- **Average Friends per User:** 44.49
- **Features per User:** 576 dimensions

### **Network Properties:**
- **Connected:** Yes (all users reachable)
- **Network Diameter:** 8 steps (max distance)
- **Average Path Length:** 3.78 steps (typical "6 degrees of separation")
- **Clustering Coefficient:** 0.6172 (61.72% - friends of friends are likely friends)

### **Degree Distribution:**
- **Minimum:** 2 friends
- **Maximum:** 1,034 friends (very popular user!)
- **Median:** 26 friends
- **Average:** 44.49 friends

### **Data Splits:**
- **Training:** 123,418 examples (70%)
- **Validation:** 26,446 examples (15%)
- **Test:** 26,448 examples (15%)
- **Class Balance:** 50% positive (friends), 50% negative (non-friends)

---

## 🎯 One-Sentence Summary

**"We use the Facebook Social Circles dataset with 3,963 users and 88,156 friendships, where each user has 576 profile features, to predict which users are likely to become friends using graph neural networks."**

---

## 💡 Key Points to Mention

1. **Real-world data** from Stanford SNAP project
2. **3,963 users** with **88,156 friendships**
3. **576 features** per user (profile attributes)
4. **Small world network** - average 3.78 steps between users
5. **High clustering** - 61.72% (friends of friends are friends)
6. **Balanced dataset** - 50% positive, 50% negative examples
7. **Proper splits** - 70% train, 15% validation, 15% test

---

## 📝 Presentation Template

### **Opening:**
"We're using the Facebook Social Circles dataset from Stanford's SNAP project."

### **Scale:**
"The dataset contains 3,963 users with 88,156 existing friendships, and each user has 576-dimensional feature vectors representing their profile attributes."

### **Network Properties:**
"This is a real social network with interesting properties:
- It's a connected network where all users are reachable
- Average path length of 3.78 - typical 'small world' property
- High clustering of 0.62 - friends of friends are likely to be friends
- Power-law degree distribution - most users have few friends, few have many"

### **Task:**
"Our goal is link prediction - predicting which users are likely to become friends. We use both the network structure and user features to make these predictions."

### **Data Quality:**
"The dataset is properly split into 70% training, 15% validation, and 15% test sets, with balanced positive and negative examples."

---

## 🔢 Important Numbers (Memorize These)

- **3,963** users
- **88,156** friendships
- **44.49** average friends per user
- **576** features per user
- **3.78** average path length
- **0.6172** clustering coefficient
- **70/15/15** train/val/test split

---

## ❓ Quick Answers

**Q: How big is the dataset?**  
A: 3,963 users, 88,156 friendships - large enough to be meaningful, manageable for training.

**Q: What do the features represent?**  
A: 576-dimensional vectors encoding user profile attributes, interests, and activities.

**Q: Is the network realistic?**  
A: Yes, it shows typical social network properties like small world and high clustering.

**Q: How is the data split?**  
A: 70% training, 15% validation, 15% test - standard machine learning practice.

**Q: Is the dataset balanced?**  
A: Yes, 50% positive examples (friends) and 50% negative examples (non-friends).

---

**Print this page for quick reference during your presentation!**

