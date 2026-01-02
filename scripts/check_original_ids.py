"""
Check if two original node IDs (not mapped indices) share groups.
"""

import torch
import pickle
import os
import sys

def check_original_ids(user1_orig_id, user2_orig_id):
    """Check if two original node IDs share groups."""
    # Load dataset
    data_path = 'data/processed/facebook_combined.pt'
    data = torch.load(data_path, weights_only=False)
    
    # Get circles
    circles = {}
    if hasattr(data, 'circles') and data.circles:
        circles = data.circles
    
    if not circles:
        print(f"No circles/groups information found in the dataset.")
        return
    
    # Find groups for each user (by original ID)
    user1_groups = []
    user2_groups = []
    
    for circle_name, members in circles.items():
        if user1_orig_id in members:
            user1_groups.append(circle_name)
        if user2_orig_id in members:
            user2_groups.append(circle_name)
    
    # Find shared groups
    shared_groups = set(user1_groups) & set(user2_groups)
    
    # Print results
    print("=" * 60)
    print(f"CHECKING SHARED GROUPS: Original Node IDs {user1_orig_id} and {user2_orig_id}")
    print("=" * 60)
    
    print(f"\nOriginal Node ID {user1_orig_id}:")
    if user1_groups:
        print(f"  Groups: {', '.join(user1_groups)}")
        print(f"  Total groups: {len(user1_groups)}")
    else:
        print(f"  Groups: NONE (not in any group)")
    
    print(f"\nOriginal Node ID {user2_orig_id}:")
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
        # Default: check original IDs 828 and 798
        user1 = 828
        user2 = 798
    
    check_original_ids(user1, user2)

