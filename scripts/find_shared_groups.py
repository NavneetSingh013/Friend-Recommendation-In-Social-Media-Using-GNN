"""
Script to find users who share social groups/circles.
"""

import os
import sys
import torch
import pickle
from collections import defaultdict
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def load_dataset_with_circles():
    """Load the Facebook dataset and extract circle information."""
    # Load processed data
    data_path = 'data/processed/facebook_combined.pt'
    metadata_path = 'data/processed/facebook_metadata.pkl'
    
    data = torch.load(data_path, weights_only=False)
    
    # Try to load circles from metadata
    circles = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                circles = metadata.get('circles', {})
        except:
            pass
    
    # If no circles in metadata, check if data object has circles
    if not circles and hasattr(data, 'circles'):
        circles = data.circles
    
    # Get node mappings
    if hasattr(data, 'node_to_idx') and hasattr(data, 'idx_to_node'):
        node_to_idx = data.node_to_idx
        idx_to_node = data.idx_to_node
    elif hasattr(data, 'node_to_idx'):
        node_to_idx = data.node_to_idx
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    else:
        # Create default mapping
        node_to_idx = {i: i for i in range(data.num_nodes)}
        idx_to_node = {i: i for i in range(data.num_nodes)}
    
    return data, circles, node_to_idx, idx_to_node


def find_shared_groups(circles, node_to_idx, idx_to_node):
    """Find all pairs of users who share groups."""
    # Map nodes to their groups
    node_to_groups = defaultdict(set)
    
    # Process circles
    for circle_name, members in circles.items():
        for member in members:
            # Map original node ID to index
            if member in node_to_idx:
                node_idx = node_to_idx[member]
                node_to_groups[node_idx].add(circle_name)
    
    # Find users with shared groups
    shared_groups_dict = {}
    node_list = list(node_to_groups.keys())
    
    for i in range(len(node_list)):
        node1 = node_list[i]
        groups1 = node_to_groups[node1]
        
        for j in range(i + 1, len(node_list)):
            node2 = node_list[j]
            groups2 = node_to_groups[node2]
            
            # Find shared groups
            shared = groups1.intersection(groups2)
            if shared:
                pair_key = tuple(sorted([node1, node2]))
                shared_groups_dict[pair_key] = {
                    'user1': node1,
                    'user2': node2,
                    'shared_groups': list(shared),
                    'num_shared': len(shared)
                }
    
    return shared_groups_dict, node_to_groups


def display_shared_groups(shared_groups_dict, node_to_groups, circles, idx_to_node, top_n=20):
    """Display users with shared groups."""
    print("=" * 80)
    print("USERS WITH SHARED GROUPS")
    print("=" * 80)
    
    if not shared_groups_dict:
        print("\n[INFO] No shared groups found in the processed dataset.")
        print("       The circles information may not be included in the processed data.")
        print("\n       You can check the raw data in: data/raw/facebook/facebook/")
        print("       Each ego network has a .circles file with group information.")
        return
    
    # Sort by number of shared groups
    sorted_pairs = sorted(shared_groups_dict.items(), 
                         key=lambda x: x[1]['num_shared'], 
                         reverse=True)
    
    print(f"\nTotal pairs of users with shared groups: {len(sorted_pairs)}")
    print(f"Showing top {min(top_n, len(sorted_pairs))} pairs:\n")
    
    print("-" * 80)
    for idx, (pair_key, info) in enumerate(sorted_pairs[:top_n], 1):
        user1_idx = info['user1']
        user2_idx = info['user2']
        shared = info['shared_groups']
        num_shared = info['num_shared']
        
        # Map back to original node IDs if available
        if isinstance(idx_to_node, dict):
            user1_id = idx_to_node.get(user1_idx, user1_idx)
            user2_id = idx_to_node.get(user2_idx, user2_idx)
        elif isinstance(idx_to_node, list) and user1_idx < len(idx_to_node):
            user1_id = idx_to_node[user1_idx]
            user2_id = idx_to_node[user2_idx] if user2_idx < len(idx_to_node) else user2_idx
        else:
            user1_id = user1_idx
            user2_id = user2_idx
        
        print(f"\n[{idx}] Users {user1_id} and {user2_id}")
        print(f"    Shared Groups: {num_shared}")
        print(f"    Group Names: {', '.join(shared[:5])}")
        if len(shared) > 5:
            print(f"    ... and {len(shared) - 5} more")
    
    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    num_shared_counts = [info['num_shared'] for info in shared_groups_dict.values()]
    if num_shared_counts:
        print(f"\nDistribution of shared groups per pair:")
        print(f"  Average: {sum(num_shared_counts) / len(num_shared_counts):.2f}")
        print(f"  Maximum: {max(num_shared_counts)}")
        print(f"  Minimum: {min(num_shared_counts)}")
    
    # Users with most groups
    print(f"\nUsers with most groups:")
    user_group_counts = [(node, len(groups)) for node, groups in node_to_groups.items()]
    user_group_counts.sort(key=lambda x: x[1], reverse=True)
    
    for node, count in user_group_counts[:10]:
        if isinstance(idx_to_node, dict):
            node_id = idx_to_node.get(node, node)
        elif isinstance(idx_to_node, list) and node < len(idx_to_node):
            node_id = idx_to_node[node]
        else:
            node_id = node
        print(f"  User {node_id}: {count} groups")
    
    # Total groups
    print(f"\nTotal unique groups/circles: {len(circles)}")
    print(f"Users belonging to at least one group: {len(node_to_groups)}")


def display_group_members(circles, node_to_idx, idx_to_node, top_groups=10):
    """Display group membership information."""
    print("\n" + "=" * 80)
    print("GROUP MEMBERSHIP INFORMATION")
    print("=" * 80)
    
    if not circles:
        print("\n[INFO] No circles/groups information available.")
        return
    
    # Sort groups by size
    sorted_groups = sorted(circles.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\nTotal groups/circles: {len(circles)}")
    print(f"Showing top {min(top_groups, len(sorted_groups))} largest groups:\n")
    
    for idx, (group_name, members) in enumerate(sorted_groups[:top_groups], 1):
        # Map original IDs to indices
        member_indices = []
        for member in members:
            if member in node_to_idx:
                member_indices.append(node_to_idx[member])
        
        print(f"\n[{idx}] Group: {group_name}")
        print(f"    Size: {len(member_indices)} members")
        print(f"    Members (first 10): {member_indices[:10]}")
        if len(member_indices) > 10:
            print(f"    ... and {len(member_indices) - 10} more")


def main():
    """Main function."""
    print("Loading dataset...")
    data, circles, node_to_idx, idx_to_node = load_dataset_with_circles()
    
    print(f"Dataset loaded: {data.num_nodes} nodes, {data.edge_index.size(1) // 2} edges")
    print(f"Circles found: {len(circles)}")
    
    if circles:
        print("\nFinding users with shared groups...")
        shared_groups_dict, node_to_groups = find_shared_groups(
            circles, node_to_idx, idx_to_node
        )
        
        display_shared_groups(shared_groups_dict, node_to_groups, 
                            circles, idx_to_node, top_n=30)
        display_group_members(circles, node_to_idx, idx_to_node, top_groups=15)
    else:
        print("\n[WARNING] No circles/groups information found in processed data.")
        print("\nThe circles information exists in the raw data files.")
        print("Location: data/raw/facebook/facebook/*.circles")
        print("\nEach .circles file contains group information for an ego network.")
        print("The processed dataset may not include this information.")
        
        # Try to read from raw files
        raw_dir = 'data/raw/facebook/facebook'
        if os.path.exists(raw_dir):
            print(f"\nChecking raw files in: {raw_dir}")
            circle_files = [f for f in os.listdir(raw_dir) if f.endswith('.circles')]
            if circle_files:
                print(f"Found {len(circle_files)} circle files:")
                for cf in circle_files[:5]:
                    print(f"  - {cf}")
                print(f"  ... and {len(circle_files) - 5} more")
                
                # Read one example
                if circle_files:
                    example_file = os.path.join(raw_dir, circle_files[0])
                    print(f"\nExample from {circle_files[0]}:")
                    try:
                        with open(example_file, 'r') as f:
                            lines = f.readlines()[:5]
                            for line in lines:
                                parts = line.strip().split('\t')
                                if len(parts) > 1:
                                    group_name = parts[0]
                                    num_members = len(parts) - 1
                                    print(f"  Group: {group_name}, Members: {num_members}")
                    except Exception as e:
                        print(f"  Error reading file: {e}")


if __name__ == '__main__':
    main()

