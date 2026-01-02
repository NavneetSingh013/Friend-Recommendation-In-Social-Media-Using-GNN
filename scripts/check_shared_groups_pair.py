"""
Quick script to check if two specific users share groups.
"""

import torch
import pickle
import os
import sys

def check_shared_groups(user1_id, user2_id):
    """Check if two users share any groups."""
    # Load dataset
    data_path = 'data/processed/facebook_combined.pt'
    metadata_path = 'data/processed/facebook_metadata.pkl'
    
    data = torch.load(data_path, weights_only=False)
    
    # Get circles
    circles = {}
    if hasattr(data, 'circles') and data.circles:
        circles = data.circles
    
    if not circles and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                circles = metadata.get('circles', {})
        except:
            pass
    
    if not circles:
        print(f"No circles/groups information found in the dataset.")
        return
    
    # Get node mapping
    if hasattr(data, 'idx_to_node'):
        idx_to_node = data.idx_to_node
    else:
        idx_to_node = list(range(data.num_nodes))
    
    # Map to original IDs
    if isinstance(idx_to_node, list):
        user1_orig = idx_to_node[user1_id] if user1_id < len(idx_to_node) else user1_id
        user2_orig = idx_to_node[user2_id] if user2_id < len(idx_to_node) else user2_id
    elif isinstance(idx_to_node, dict):
        user1_orig = idx_to_node.get(user1_id, user1_id)
        user2_orig = idx_to_node.get(user2_id, user2_id)
    else:
        user1_orig = user1_id
        user2_orig = user2_id
    
    # Find groups for each user
    user1_groups = []
    user2_groups = []
    
    for circle_name, members in circles.items():
        if user1_orig in members:
            user1_groups.append(circle_name)
        if user2_orig in members:
            user2_groups.append(circle_name)
    
    # Find shared groups
    shared_groups = set(user1_groups) & set(user2_groups)
    
    # Print results
    print("=" * 60)
    print(f"CHECKING SHARED GROUPS: Users {user1_id} and {user2_id}")
    print("=" * 60)
    print(f"\nUser {user1_id} (original ID: {user1_orig}):")
    if user1_groups:
        print(f"  Groups: {', '.join(user1_groups)}")
        print(f"  Total groups: {len(user1_groups)}")
    else:
        print(f"  Groups: NONE (not in any group)")
    
    print(f"\nUser {user2_id} (original ID: {user2_orig}):")
    if user2_groups:
        print(f"  Groups: {', '.join(user2_groups)}")
        print(f"  Total groups: {len(user2_groups)}")
    else:
        print(f"  Groups: NONE (not in any group)")
    
    print("\n" + "-" * 60)
    if shared_groups:
        print(f"[YES] They share {len(shared_groups)} group(s):")
        for group in sorted(shared_groups):
            print(f"  - {group}")
    else:
        print("[NO] They do NOT share any groups")
    print("=" * 60)
    
    return shared_groups

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        user1 = int(sys.argv[1])
        user2 = int(sys.argv[2])
    else:
        # Default: check users 828 and 798
        user1 = 828
        user2 = 798
    
    check_shared_groups(user1, user2)

