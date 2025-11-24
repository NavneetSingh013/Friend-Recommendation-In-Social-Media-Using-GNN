"""
Script to download and prepare datasets.
"""

import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.facebook_loader import FacebookDatasetLoader, create_synthetic_dataset
from src.data.ogb_loader import OGBDatasetLoader
from src.data.preprocessing import prepare_link_prediction_data
from src.data.heuristics import compute_heuristics
import torch


def main():
    parser = argparse.ArgumentParser(description='Download and prepare datasets')
    parser.add_argument('--dataset', type=str, choices=['facebook', 'ogbl-collab', 'synthetic'],
                       default='facebook', help='Dataset to download')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory (for processed data)')
    parser.add_argument('--download', action='store_true', help='Download dataset')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess dataset')
    parser.add_argument('--max_egos', type=int, default=10,
                       help='Maximum number of ego networks (for Facebook)')
    parser.add_argument('--synthetic_nodes', type=int, default=100,
                       help='Number of nodes for synthetic dataset')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.dataset == 'facebook':
        loader = FacebookDatasetLoader()
        
        if args.download:
            loader.download()
        elif args.preprocess:
            # Try to process without downloading (assumes data exists)
            print("Attempting to process existing data...")
        
        if args.preprocess:
            try:
                data = loader.process_and_save(max_egos=args.max_egos)
                
                # Prepare link prediction data
                print("Preparing link prediction data...")
                link_data = prepare_link_prediction_data(data, seed=42)
                
                # Save link prediction data
                link_data_path = os.path.join(args.output, 'facebook_link_data.pt')
                torch.save(link_data, link_data_path)
                print(f"Saved link prediction data to {link_data_path}")
                
                # Compute heuristics (optional, can be slow)
                print("Computing heuristics...")
                try:
                    heuristics = compute_heuristics(data)
                    heuristics_path = os.path.join(args.output, 'facebook_heuristics.pt')
                    torch.save(heuristics, heuristics_path)
                    print(f"Saved heuristics to {heuristics_path}")
                except Exception as e:
                    print(f"Warning: Could not compute heuristics: {e}")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Please run with --download first to download the dataset.")
                return
    
    elif args.dataset == 'ogbl-collab':
        loader = OGBDatasetLoader('ogbl-collab')
        
        if args.download or args.preprocess:
            data, splits = loader.process_and_save(output_dir=args.output)
            print(f"Dataset processed: {data.num_nodes} nodes, {data.edge_index.size(1) // 2} edges")
    
    elif args.dataset == 'synthetic':
        print(f"Creating synthetic dataset with {args.synthetic_nodes} nodes...")
        data = create_synthetic_dataset(num_nodes=args.synthetic_nodes, num_edges=args.synthetic_nodes * 2, feature_dim=16)
        
        # Save
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, 'synthetic.pt')
        torch.save(data, output_path)
        print(f"Saved synthetic dataset to {output_path}")
        
        # Prepare link prediction data
        print("Preparing link prediction data...")
        link_data = prepare_link_prediction_data(data, seed=42)
        link_data_path = os.path.join(args.output, 'synthetic_link_data.pt')
        torch.save(link_data, link_data_path)
        print(f"Saved link prediction data to {link_data_path}")
        
        print("Synthetic dataset created successfully!")
        print(f"  - Nodes: {data.num_nodes}")
        print(f"  - Edges: {data.edge_index.size(1) // 2}")
        print(f"  - Features: {data.x.size(1)}")


if __name__ == '__main__':
    main()

