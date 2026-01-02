"""
Check the distribution of path lengths in recommendations.
"""

import torch
import sys
import os
sys.path.append('.')

from src.evaluation.explainability import compute_shortest_path

def check_path_distribution(user_id, top_k_recommendations):
    """Check path lengths for a user's recommendations."""
    data = torch.load('data/processed/facebook_combined.pt', weights_only=False)
    
    print("=" * 60)
    print(f"PATH LENGTH DISTRIBUTION FOR USER {user_id}")
    print("=" * 60)
    
    path_lengths = []
    for friend_id in top_k_recommendations:
        path_len = compute_shortest_path(
            data.edge_index, user_id, friend_id, data.num_nodes
        )
        path_lengths.append(path_len)
    
    from collections import Counter
    path_dist = Counter(path_lengths)
    
    print(f"\nPath length distribution:")
    for path_len in sorted(path_dist.keys()):
        count = path_dist[path_len]
        percentage = 100 * count / len(path_lengths)
        print(f"  {path_len}-hop: {count} recommendations ({percentage:.1f}%)")
    
    print(f"\nTotal recommendations checked: {len(path_lengths)}")
    
    # Check if all are 2-hop
    if all(p == 2 for p in path_lengths):
        print("\n[NOTE] All recommendations are 2-hop!")
        print("This is expected because:")
        print("  1. They can't be 1-hop (direct friends - excluded from candidates)")
        print("  2. 2-hop means 'friends of friends' (most common recommendation pattern)")
        print("  3. Longer paths (>2-hop) are less likely to be recommended")
    
    return path_lengths

def check_why_2hop():
    """Explain why path evidence is usually 2-hop."""
    data = torch.load('data/processed/facebook_combined.pt', weights_only=False)
    
    print("\n" + "=" * 60)
    print("WHY PATH EVIDENCE IS USUALLY 2-HOP")
    print("=" * 60)
    
    # Get a sample user
    user_id = 1043
    
    # Get existing friends (1-hop neighbors)
    existing_friends = set()
    for i in range(data.edge_index.size(1)):
        src = data.edge_index[0, i].item()
        dst = data.edge_index[1, i].item()
        if src == user_id:
            existing_friends.add(dst)
        if dst == user_id:
            existing_friends.add(src)
    
    print(f"\nUser {user_id}:")
    print(f"  Direct friends (1-hop): {len(existing_friends)}")
    
    # Get 2-hop neighbors (friends of friends, excluding direct friends)
    two_hop_neighbors = set()
    for friend in list(existing_friends)[:10]:  # Sample
        for i in range(data.edge_index.size(1)):
            src = data.edge_index[0, i].item()
            dst = data.edge_index[1, i].item()
            if src == friend and dst != user_id and dst not in existing_friends:
                two_hop_neighbors.add(dst)
            if dst == friend and src != user_id and src not in existing_friends:
                two_hop_neighbors.add(src)
    
    print(f"  2-hop neighbors (friends of friends, sample): {len(two_hop_neighbors)}")
    
    print("\n" + "-" * 60)
    print("EXPLANATION:")
    print("-" * 60)
    print("""
Path evidence is usually 2-hop because:

1. **1-hop paths are excluded:**
   - 1-hop = direct friends
   - Direct friends are excluded from candidate set
   - So recommendations can't be 1-hop

2. **2-hop paths are most common:**
   - 2-hop = friends of friends
   - These are the most likely recommendations
   - Most mutual friends come from 2-hop connections
   - Model learns this pattern strongly

3. **3+ hop paths are rare in recommendations:**
   - Longer paths = weaker connections
   - Model typically gives lower scores to distant users
   - Most top-K recommendations are 2-hop away

4. **Social network property:**
   - Real social networks show "triadic closure"
   - Friends of friends are likely to become friends
   - This is why 2-hop recommendations are common
    """)

if __name__ == '__main__':
    check_why_2hop()

