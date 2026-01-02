"""
Check which ego networks were processed and why some circles might be missing.
"""

import os
import torch

def check_ego_processing():
    """Check which ego networks were processed."""
    print("=" * 80)
    print("CHECKING EGO NETWORK PROCESSING")
    print("=" * 80)
    
    # Check raw data
    raw_dir = 'data/raw/facebook/facebook'
    if not os.path.exists(raw_dir):
        print(f"Raw data directory not found: {raw_dir}")
        return
    
    # Get all ego networks
    edge_files = [f for f in os.listdir(raw_dir) if f.endswith('.edges')]
    ego_ids = sorted([int(f.replace('.edges', '')) for f in edge_files])
    
    print(f"\n1. Available ego networks in raw data: {len(ego_ids)}")
    print(f"   Ego IDs: {ego_ids}")
    
    # Check which have circles
    circles_files = [f for f in os.listdir(raw_dir) if f.endswith('.circles')]
    ego_ids_with_circles = sorted([int(f.replace('.circles', '')) for f in circles_files])
    
    print(f"\n2. Ego networks with circle files: {len(ego_ids_with_circles)}")
    print(f"   Ego IDs with circles: {ego_ids_with_circles}")
    print(f"   Ego 107 has circles: {107 in ego_ids_with_circles}")
    
    # Check processed data
    data_path = 'data/processed/facebook_combined.pt'
    if os.path.exists(data_path):
        data = torch.load(data_path, weights_only=False)
        
        print(f"\n3. Processed dataset:")
        print(f"   Total nodes: {data.num_nodes}")
        print(f"   Total edges: {data.edge_index.size(1) // 2}")
        
        # Check if ego nodes are in graph
        if hasattr(data, 'node_to_idx') and isinstance(data.node_to_idx, dict):
            all_original_ids = set(data.node_to_idx.keys())
            ego_nodes_in_graph = [eid for eid in ego_ids if eid in all_original_ids]
            print(f"\n4. Ego nodes present in processed graph: {len(ego_nodes_in_graph)}/{len(ego_ids)}")
            print(f"   Present ego IDs: {sorted(ego_nodes_in_graph)}")
            print(f"   Ego 107 in graph: {107 in all_original_ids}")
        
        # Check circles
        if hasattr(data, 'circles') and data.circles:
            print(f"\n5. Circles in processed data:")
            print(f"   Total circles: {len(data.circles)}")
            circle_names = sorted(list(data.circles.keys()))
            print(f"   Circle names: {circle_names}")
            
            # Try to identify which ego networks these circles came from
            # by checking if ego nodes are members
            print(f"\n6. Identifying which ego networks contributed circles:")
            ego_to_circles = {}
            for ego_id in ego_ids_with_circles:
                ego_circles = []
                for circle_name, members in data.circles.items():
                    if ego_id in members:
                        ego_circles.append(circle_name)
                if ego_circles:
                    ego_to_circles[ego_id] = ego_circles
                    print(f"   Ego {ego_id}: {len(ego_circles)} circles ({', '.join(ego_circles[:3])}...)")
            
            print(f"\n   Ego networks with circles in processed data: {len(ego_to_circles)}")
            print(f"   Ego IDs: {sorted(ego_to_circles.keys())}")
            print(f"   Ego 107 circles in processed data: {'YES' if 107 in ego_to_circles else 'NO'}")
    
    # Check what the default max_egos is
    print(f"\n7. Default processing settings:")
    print(f"   The dataset is typically processed with max_egos=10")
    print(f"   This means only the first 10 ego networks are processed")
    print(f"   Ego networks are processed in sorted order: {sorted(ego_ids)[:10]}")
    print(f"   Ego 107 would be processed: {'YES' if 107 in sorted(ego_ids)[:10] else 'NO (only first 10 are processed)'}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    check_ego_processing()

