"""
Debug script to understand why shared groups might show 0.
"""

import torch
import pickle
import os

def debug_shared_groups(user1_id, user2_id):
    """Debug why shared groups might not be detected."""
    print("=" * 80)
    print(f"DEBUGGING SHARED GROUPS: Users {user1_id} and {user2_id}")
    print("=" * 80)
    
    # Load processed data
    data_path = 'data/processed/facebook_combined.pt'
    data = torch.load(data_path, weights_only=False)
    
    print(f"\n1. Checking processed data structure:")
    print(f"   Has circles attribute: {hasattr(data, 'circles')}")
    if hasattr(data, 'circles'):
        print(f"   Circles is not None: {data.circles is not None}")
        print(f"   Number of circles: {len(data.circles) if data.circles else 0}")
        if data.circles:
            print(f"   Sample circle names: {list(data.circles.keys())[:5]}")
    
    # Check node mappings
    print(f"\n2. Checking node ID mappings:")
    if hasattr(data, 'node_to_idx'):
        print(f"   Has node_to_idx: {type(data.node_to_idx)}")
        if isinstance(data.node_to_idx, dict):
            print(f"   User {user1_id} mapped to: {data.node_to_idx.get(user1_id, 'NOT FOUND')}")
            print(f"   User {user2_id} mapped to: {data.node_to_idx.get(user2_id, 'NOT FOUND')}")
    if hasattr(data, 'idx_to_node'):
        print(f"   Has idx_to_node: {type(data.idx_to_node)}")
        if isinstance(data.idx_to_node, list):
            print(f"   idx_to_node length: {len(data.idx_to_node)}")
            if user1_id < len(data.idx_to_node):
                print(f"   Index {user1_id} maps to original ID: {data.idx_to_node[user1_id]}")
            if user2_id < len(data.idx_to_node):
                print(f"   Index {user2_id} maps to original ID: {data.idx_to_node[user2_id]}")
    
    # Check circles with original IDs
    print(f"\n3. Checking circles with ORIGINAL node IDs ({user1_id}, {user2_id}):")
    if hasattr(data, 'circles') and data.circles:
        user1_groups_orig = []
        user2_groups_orig = []
        for circle_name, members in data.circles.items():
            if user1_id in members:
                user1_groups_orig.append(circle_name)
            if user2_id in members:
                user2_groups_orig.append(circle_name)
        
        print(f"   User {user1_id} (original ID) in circles: {user1_groups_orig}")
        print(f"   User {user2_id} (original ID) in circles: {user2_groups_orig}")
        shared_orig = set(user1_groups_orig) & set(user2_groups_orig)
        print(f"   Shared circles (original IDs): {list(shared_orig)}")
    
    # Check circles with mapped indices
    print(f"\n4. Checking circles with MAPPED indices:")
    if hasattr(data, 'circles') and data.circles and hasattr(data, 'node_to_idx'):
        if isinstance(data.node_to_idx, dict):
            user1_mapped = data.node_to_idx.get(user1_id, user1_id)
            user2_mapped = data.node_to_idx.get(user2_id, user2_id)
            print(f"   User {user1_id} mapped to index: {user1_mapped}")
            print(f"   User {user2_id} mapped to index: {user2_mapped}")
            
            user1_groups_mapped = []
            user2_groups_mapped = []
            for circle_name, members in data.circles.items():
                if user1_mapped in members:
                    user1_groups_mapped.append(circle_name)
                if user2_mapped in members:
                    user2_groups_mapped.append(circle_name)
            
            print(f"   User {user1_mapped} (mapped index) in circles: {user1_groups_mapped}")
            print(f"   User {user2_mapped} (mapped index) in circles: {user2_groups_mapped}")
            shared_mapped = set(user1_groups_mapped) & set(user2_groups_mapped)
            print(f"   Shared circles (mapped indices): {list(shared_mapped)}")
    
    # Check raw file
    print(f"\n5. Checking raw 107.circles file:")
    raw_file = 'data/raw/facebook/facebook/107.circles'
    if os.path.exists(raw_file):
        print(f"   File exists: {raw_file}")
        with open(raw_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > 0 and parts[0] == 'circle0':
                    members = [int(x) for x in parts[1:] if x.isdigit()]
                    print(f"   circle0 in 107.circles has {len(members)} members")
                    print(f"   First 10 members: {members[:10]}")
                    print(f"   User {user1_id} in circle0: {user1_id in members}")
                    print(f"   User {user2_id} in circle0: {user2_id in members}")
                    if user1_id in members and user2_id in members:
                        print(f"   [YES] Both users are in circle0 in raw file!")
                    break
    else:
        print(f"   File not found: {raw_file}")
    
    # Simulate what explainability function does
    print(f"\n6. Simulating explainability.compute_shared_groups():")
    if hasattr(data, 'circles') and data.circles:
        # This is what the function does
        user_groups = []
        friend_groups = []
        
        # Check if idx_to_node mapping exists
        if hasattr(data, 'idx_to_node') and data.idx_to_node:
            if isinstance(data.idx_to_node, list):
                if user1_id < len(data.idx_to_node):
                    user1_orig_id = data.idx_to_node[user1_id]
                else:
                    user1_orig_id = user1_id
                if user2_id < len(data.idx_to_node):
                    user2_orig_id = data.idx_to_node[user2_id]
                else:
                    user2_orig_id = user2_id
            else:
                user1_orig_id = user1_id
                user2_orig_id = user2_id
        else:
            user1_orig_id = user1_id
            user2_orig_id = user2_id
        
        print(f"   Using original IDs: {user1_orig_id}, {user2_orig_id}")
        
        for circle_name, members in data.circles.items():
            if user1_orig_id in members:
                user_groups.append(circle_name)
            if user2_orig_id in members:
                friend_groups.append(circle_name)
        
        print(f"   User {user1_id} (orig {user1_orig_id}) groups: {user_groups}")
        print(f"   User {user2_id} (orig {user2_orig_id}) groups: {friend_groups}")
        shared = set(user_groups) & set(friend_groups)
        print(f"   Shared groups: {list(shared)}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    debug_shared_groups(1043, 1254)

