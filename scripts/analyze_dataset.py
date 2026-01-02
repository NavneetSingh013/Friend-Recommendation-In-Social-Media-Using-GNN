"""
Comprehensive dataset analysis script.
Analyzes the dataset structure, statistics, and characteristics.
"""

import os
import sys
import torch
import numpy as np
import networkx as nx
import pandas as pd
from collections import Counter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def analyze_graph_structure(data, dataset_name):
    """Analyze the graph structure."""
    print(f"\n{'='*80}")
    print(f"GRAPH STRUCTURE ANALYSIS - {dataset_name.upper()}")
    print(f"{'='*80}\n")
    
    # Basic statistics
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1) // 2  # Undirected graph
    
    print("[Basic Statistics]")
    print(f"  • Total Nodes (Users): {num_nodes:,}")
    print(f"  • Total Edges (Friendships): {num_edges:,}")
    print(f"  • Average Degree: {2 * num_edges / num_nodes:.2f}")
    print(f"  • Graph Density: {2 * num_edges / (num_nodes * (num_nodes - 1)):.6f}")
    print(f"  • Node Features: {data.x.size(1)}")
    
    # Build NetworkX graph for analysis
    print("\n[Building graph for analysis...]")
    G = nx.Graph()
    edge_list = data.edge_index.t().cpu().numpy()
    G.add_edges_from(edge_list)
    
    # Connectivity
    print("\n[Connectivity Analysis]")
    if nx.is_connected(G):
        print("  • Graph is CONNECTED (all nodes reachable)")
        print(f"  • Diameter: {nx.diameter(G)}")
        print(f"  • Average Path Length: {nx.average_shortest_path_length(G):.2f}")
    else:
        components = list(nx.connected_components(G))
        print(f"  • Graph has {len(components)} connected components")
        largest = max(components, key=len)
        print(f"  • Largest component: {len(largest):,} nodes ({100*len(largest)/num_nodes:.1f}%)")
        if len(components) > 1:
            print(f"  • Second largest: {len(max(components - {largest}, key=len)):,} nodes")
    
    # Degree distribution
    print("\n[Degree Distribution]")
    degrees = [G.degree(n) for n in G.nodes()]
    degree_counter = Counter(degrees)
    print(f"  • Min Degree: {min(degrees)}")
    print(f"  • Max Degree: {max(degrees)}")
    print(f"  • Average Degree: {np.mean(degrees):.2f}")
    print(f"  • Median Degree: {np.median(degrees):.2f}")
    print(f"  • Std Deviation: {np.std(degrees):.2f}")
    
    # Degree percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\n  Degree Percentiles:")
    for p in percentiles:
        val = np.percentile(degrees, p)
        print(f"    {p}th percentile: {val:.1f}")
    
    # Most connected nodes
    top_degrees = sorted(degrees, reverse=True)[:10]
    print(f"\n  Top 10 Degrees: {top_degrees}")
    
    # Clustering
    print("\n[Clustering Analysis]")
    clustering = nx.clustering(G)
    avg_clustering = np.mean(list(clustering.values()))
    print(f"  • Average Clustering Coefficient: {avg_clustering:.4f}")
    print(f"  • This measures how likely friends of a user are also friends with each other")
    
    # Feature analysis
    print("\n[Feature Analysis]")
    features = data.x.cpu().numpy()
    print(f"  • Feature Shape: {features.shape}")
    print(f"  • Feature Type: {type(features[0,0])}")
    print(f"  • Min Value: {features.min():.4f}")
    print(f"  • Max Value: {features.max():.4f}")
    print(f"  • Mean Value: {features.mean():.4f}")
    print(f"  • Std Deviation: {features.std():.4f}")
    
    # Check for node attributes
    if hasattr(data, 'y') and data.y is not None:
        print(f"\n  • Has Node Labels: Yes")
        print(f"  • Label Shape: {data.y.shape}")
    else:
        print(f"\n  • Has Node Labels: No")
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': 2 * num_edges / num_nodes,
        'density': 2 * num_edges / (num_nodes * (num_nodes - 1)),
        'is_connected': nx.is_connected(G),
        'num_components': len(list(nx.connected_components(G))),
        'avg_clustering': avg_clustering,
        'degrees': degrees,
        'graph': G
    }


def analyze_link_data(link_data, dataset_name):
    """Analyze the link prediction data splits."""
    print(f"\n{'='*80}")
    print(f"LINK PREDICTION DATA ANALYSIS - {dataset_name.upper()}")
    print(f"{'='*80}\n")
    
    train_edges = link_data['train_edges']
    val_edges = link_data['val_edges']
    test_edges = link_data['test_edges']
    train_labels = link_data['train_labels']
    val_labels = link_data['val_labels']
    test_labels = link_data['test_labels']
    
    print("[Data Split Statistics]")
    print(f"  • Training Edges: {train_edges.size(1):,}")
    print(f"    - Positive: {train_labels.sum().item():,} ({100*train_labels.sum().item()/train_labels.size(0):.1f}%)")
    print(f"    - Negative: {(train_labels.size(0) - train_labels.sum().item()):,} ({100*(1-train_labels.sum().item()/train_labels.size(0)):.1f}%)")
    
    print(f"  • Validation Edges: {val_edges.size(1):,}")
    print(f"    - Positive: {val_labels.sum().item():,} ({100*val_labels.sum().item()/val_labels.size(0):.1f}%)")
    print(f"    - Negative: {(val_labels.size(0) - val_labels.sum().item()):,} ({100*(1-val_labels.sum().item()/val_labels.size(0)):.1f}%)")
    
    print(f"  • Test Edges: {test_edges.size(1):,}")
    print(f"    - Positive: {test_labels.sum().item():,} ({100*test_labels.sum().item()/test_labels.size(0):.1f}%)")
    print(f"    - Negative: {(test_labels.size(0) - test_labels.sum().item()):,} ({100*(1-test_labels.sum().item()/test_labels.size(0)):.1f}%)")
    
    total = train_edges.size(1) + val_edges.size(1) + test_edges.size(1)
    print(f"\n  • Total Edges: {total:,}")
    print(f"  • Train/Val/Test Split: {100*train_edges.size(1)/total:.1f}% / {100*val_edges.size(1)/total:.1f}% / {100*test_edges.size(1)/total:.1f}%")
    
    return {
        'train_size': train_edges.size(1),
        'val_size': val_edges.size(1),
        'test_size': test_edges.size(1),
        'train_pos_ratio': train_labels.sum().item() / train_labels.size(0),
        'val_pos_ratio': val_labels.sum().item() / val_labels.size(0),
        'test_pos_ratio': test_labels.sum().item() / test_labels.size(0)
    }


def create_summary_report(dataset_name, graph_stats, link_stats):
    """Create a summary report."""
    print(f"\n{'='*80}")
    print(f"DATASET SUMMARY - {dataset_name.upper()}")
    print(f"{'='*80}\n")
    
    print("[Key Characteristics]")
    print(f"  • Dataset Type: Social Network (Friend Recommendation)")
    print(f"  • Task: Link Prediction (Predicting future friendships)")
    print(f"  • Graph Type: Undirected (Friendship is mutual)")
    print(f"  • Scale: {'Large' if graph_stats['num_nodes'] > 1000 else 'Medium' if graph_stats['num_nodes'] > 100 else 'Small'}")
    
    print(f"\n[Graph Properties]")
    print(f"  • Nodes represent: Users/People in the social network")
    print(f"  • Edges represent: Existing friendships/connections")
    print(f"  • Features represent: User profile attributes/embeddings")
    print(f"  • Average connections per user: {graph_stats['avg_degree']:.1f}")
    print(f"  • Network sparsity: {100*(1-graph_stats['density']):.2f}% sparse")
    
    print(f"\n[Machine Learning Task]")
    print(f"  • Goal: Predict which users are likely to become friends")
    print(f"  • Input: Graph structure + node features")
    print(f"  • Output: Probability score for each potential friendship")
    print(f"  • Evaluation: Classification (AUC, AP) + Ranking (Precision@K, NDCG@K)")
    
    print(f"\n[Data Quality]")
    print(f"  • Training data: {link_stats['train_size']:,} examples")
    print(f"  • Class balance: {100*link_stats['train_pos_ratio']:.1f}% positive, {100*(1-link_stats['train_pos_ratio']):.1f}% negative")
    print(f"  • This is a {'balanced' if 0.4 < link_stats['train_pos_ratio'] < 0.6 else 'imbalanced'} dataset")


def main():
    """Main analysis function."""
    datasets = []
    
    # Check Facebook dataset
    if os.path.exists('data/processed/facebook_combined.pt'):
        datasets.append('facebook')
    
    # Check Synthetic dataset
    if os.path.exists('data/processed/synthetic.pt'):
        datasets.append('synthetic')
    
    if not datasets:
        print("No datasets found! Please run data preparation first.")
        return
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'#'*80}")
        print(f"# ANALYZING {dataset_name.upper()} DATASET")
        print(f"{'#'*80}")
        
        # Load data
        if dataset_name == 'facebook':
            data_path = 'data/processed/facebook_combined.pt'
            link_data_path = 'data/processed/facebook_link_data.pt'
        else:
            data_path = 'data/processed/synthetic.pt'
            link_data_path = 'data/processed/synthetic_link_data.pt'
        
        data = torch.load(data_path, weights_only=False)
        link_data = torch.load(link_data_path, weights_only=False)
        
        # Analyze
        graph_stats = analyze_graph_structure(data, dataset_name)
        link_stats = analyze_link_data(link_data, dataset_name)
        create_summary_report(dataset_name, graph_stats, link_stats)
        
        all_results[dataset_name] = {
            'graph': graph_stats,
            'links': link_stats
        }
    
    # Save summary
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print("\nUse this information to explain the dataset to your project guide!")
    print("Key points to mention:")
    print("  1. Dataset size and scale")
    print("  2. Graph structure (nodes, edges, features)")
    print("  3. Task (link prediction for friend recommendation)")
    print("  4. Data splits (train/val/test)")
    print("  5. Network properties (connectivity, clustering, degree distribution)")


if __name__ == '__main__':
    main()

