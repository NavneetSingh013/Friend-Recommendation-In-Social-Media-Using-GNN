"""
SNAP Facebook Social Circles Dataset Loader
Downloads and processes Facebook ego-networks from SNAP dataset.
"""

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Optional
import pickle


class FacebookDatasetLoader:
    """Loader for SNAP Facebook Social Circles dataset."""
    
    BASE_URL = "https://snap.stanford.edu/data/facebook.tar.gz"
    
    def __init__(self, data_dir: str = "data/raw/facebook"):
        self.data_dir = data_dir
        self.raw_dir = data_dir
        # Processed dir should be in data/processed (sibling to raw)
        # Get the parent directory of raw
        raw_parent = os.path.dirname(data_dir) if os.path.dirname(data_dir) else "data"
        if "raw" in os.path.basename(raw_parent):
            # If we're in data/raw/facebook, go up to data
            self.processed_dir = os.path.join(os.path.dirname(raw_parent), "processed")
        else:
            # Otherwise assume data/processed
            self.processed_dir = os.path.join("data", "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def download(self):
        """Download Facebook dataset from SNAP."""
        print("Downloading Facebook dataset...")
        tar_path = os.path.join(self.raw_dir, "facebook.tar.gz")
        
        if not os.path.exists(tar_path):
            urllib.request.urlretrieve(self.BASE_URL, tar_path)
            print(f"Downloaded to {tar_path}")
            
            # Extract
            import tarfile
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(self.raw_dir)
            print("Extraction complete")
        else:
            print("Dataset already downloaded")
    
    def load_ego_network(self, ego_id: int) -> Tuple[nx.Graph, dict]:
        """
        Load a single ego network.
        
        Args:
            ego_id: Ego node ID
            
        Returns:
            graph: NetworkX graph
            features: Node features dictionary
        """
        # Try facebook subdirectory first, then raw_dir
        base_path = os.path.join(self.raw_dir, "facebook")
        if not os.path.exists(base_path):
            base_path = self.raw_dir
        
        edges_file = os.path.join(base_path, f"{ego_id}.edges")
        feat_file = os.path.join(base_path, f"{ego_id}.feat")
        featnames_file = os.path.join(base_path, f"{ego_id}.featnames")
        circles_file = os.path.join(base_path, f"{ego_id}.circles")
        
        # Load edges
        G = nx.Graph()
        if os.path.exists(edges_file):
            edges = pd.read_csv(edges_file, sep=' ', header=None, names=['src', 'dst'])
            G.add_edges_from(edges.values)
        
        # Load features
        features = {}
        if os.path.exists(feat_file) and os.path.exists(featnames_file):
            feat_df = pd.read_csv(feat_file, sep=' ', header=None)
            featnames = pd.read_csv(featnames_file, sep=' ', header=None, names=['feature_id', 'feature_name'])
            
            for idx, row in feat_df.iterrows():
                node_id = int(row[0])
                feature_vector = row[1:].values.astype(float)
                features[node_id] = feature_vector
        
        # Load circles (optional)
        circles = {}
        if os.path.exists(circles_file):
            with open(circles_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) > 1:
                        circle_name = parts[0]
                        circle_members = [int(x) for x in parts[1:]]
                        circles[circle_name] = circle_members
        
        return G, features, circles
    
    def combine_ego_networks(self, ego_ids: Optional[List[int]] = None, max_egos: int = 10) -> nx.Graph:
        """
        Combine multiple ego networks into a single graph.
        
        Args:
            ego_ids: List of ego IDs to combine. If None, uses first max_egos.
            max_egos: Maximum number of ego networks to combine
            
        Returns:
            Combined graph
        """
        # Find available ego networks
        # Check both facebook subdirectory and raw_dir directly
        facebook_dir = os.path.join(self.raw_dir, "facebook")
        if not os.path.exists(facebook_dir):
            facebook_dir = self.raw_dir
        
        if not os.path.exists(facebook_dir):
            raise FileNotFoundError(f"Facebook data not found in {self.raw_dir}. Run download() first.")
        
        available_egos = [int(f.replace('.edges', '')) 
                         for f in os.listdir(facebook_dir) 
                         if f.endswith('.edges')]
        
        if ego_ids is None:
            ego_ids = available_egos[:max_egos]
        
        # Combine graphs
        combined_G = nx.Graph()
        all_features = {}
        all_circles = {}
        
        for ego_id in ego_ids:
            if ego_id in available_egos:
                G, features, circles = self.load_ego_network(ego_id)
                combined_G = nx.compose(combined_G, G)
                all_features.update(features)
                all_circles.update(circles)
                # Add ego node and connect to all its neighbors
                for neighbor in G.nodes():
                    combined_G.add_edge(ego_id, neighbor)
        
        return combined_G, all_features, all_circles
    
    def to_pyg_data(self, graph: nx.Graph, features: dict, circles: dict = None) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data object.
        
        Args:
            graph: NetworkX graph
            features: Node features dictionary
            circles: Circles dictionary (for node attributes)
            
        Returns:
            PyG Data object
        """
        # Create node mapping
        nodes = sorted(list(graph.nodes()))
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Create edge index
        edge_list = []
        for src, dst in graph.edges():
            edge_list.append([node_to_idx[src], node_to_idx[dst]])
            edge_list.append([node_to_idx[dst], node_to_idx[src]])  # Undirected
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create node features
        if features:
            # Find maximum feature dimension
            feature_dims = [len(f) for f in features.values() if len(f) > 0]
            if feature_dims:
                feature_dim = max(feature_dims)
                # If inconsistent dimensions, pad or truncate
                node_features = torch.zeros(len(nodes), feature_dim)
                for node, idx in node_to_idx.items():
                    if node in features:
                        feat = torch.tensor(features[node], dtype=torch.float)
                        if feat.size(0) < feature_dim:
                            # Pad with zeros
                            padded = torch.zeros(feature_dim)
                            padded[:feat.size(0)] = feat
                            node_features[idx] = padded
                        elif feat.size(0) > feature_dim:
                            # Truncate
                            node_features[idx] = feat[:feature_dim]
                        else:
                            node_features[idx] = feat
            else:
                # No valid features, create synthetic
                features = {}
                feature_dim = 2
        else:
            # No features provided, create synthetic
            feature_dim = 2
        
        # Check if we need to create synthetic features
        if not features or (feature_dims and not any(feature_dims)):
            # Create synthetic features (degree, clustering coefficient, etc.)
            degrees = [graph.degree(node) for node in nodes]
            clustering = list(nx.clustering(graph, nodes).values())
            node_features = torch.tensor([[d, c] for d, c in zip(degrees, clustering)], dtype=torch.float)
            # Normalize
            node_features = (node_features - node_features.mean(dim=0)) / (node_features.std(dim=0) + 1e-8)
        
        # Create data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            num_nodes=len(nodes)
        )
        
        # Store node mapping and circles
        data.node_to_idx = node_to_idx
        data.idx_to_node = nodes
        data.circles = circles if circles else {}
        
        return data
    
    def process_and_save(self, ego_ids: Optional[List[int]] = None, max_egos: int = 10):
        """
        Process and save Facebook dataset.
        
        Args:
            ego_ids: List of ego IDs to process
            max_egos: Maximum number of ego networks
        """
        print("Processing Facebook dataset...")
        
        # Combine ego networks
        graph, features, circles = self.combine_ego_networks(ego_ids, max_egos)
        print(f"Combined graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Convert to PyG
        data = self.to_pyg_data(graph, features, circles)
        
        # Save (with weights_only=False compatibility)
        output_path = os.path.join(self.processed_dir, "facebook_combined.pt")
        torch.save(data, output_path, _use_new_zipfile_serialization=False)
        print(f"Saved processed data to {output_path}")
        
        # Save metadata
        metadata = {
            'num_nodes': data.num_nodes,
            'num_edges': data.edge_index.size(1) // 2,
            'feature_dim': data.x.size(1),
            'node_to_idx': data.node_to_idx,
            'circles': data.circles
        }
        metadata_path = os.path.join(self.processed_dir, "facebook_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        return data


def create_synthetic_dataset(num_nodes: int = 100, num_edges: int = 200, feature_dim: int = 16):
    """
    Create a synthetic dataset for testing.
    
    Args:
        num_nodes: Number of nodes
        num_edges: Number of edges
        feature_dim: Feature dimension
        
    Returns:
        PyG Data object
    """
    # Create random graph
    G = nx.gnm_random_graph(num_nodes, num_edges)
    
    # Create random features
    node_features = torch.randn(num_nodes, feature_dim)
    
    # Create edge index
    edge_list = []
    for src, dst in G.edges():
        edge_list.append([src, dst])
        edge_list.append([dst, src])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    data = Data(
        x=node_features,
        edge_index=edge_index,
        num_nodes=num_nodes
    )
    
    # Create node mapping
    nodes = list(range(num_nodes))
    data.node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    data.idx_to_node = nodes
    data.circles = {}
    
    return data

