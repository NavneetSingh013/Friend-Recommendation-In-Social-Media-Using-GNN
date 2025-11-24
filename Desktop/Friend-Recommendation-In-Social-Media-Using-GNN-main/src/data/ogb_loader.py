"""
OGB (Open Graph Benchmark) Dataset Loader
Supports ogbl-collab and other OGB link prediction datasets.
"""

import os
import torch
from torch_geometric.data import Data
from typing import Optional, Tuple
import numpy as np

try:
    from ogb.linkproppred import PygLinkPropPredDataset
    OGB_AVAILABLE = True
except ImportError:
    OGB_AVAILABLE = False
    print("Warning: OGB not installed. OGBDatasetLoader will not be available.")


class OGBDatasetLoader:
    """Loader for OGB link prediction datasets."""
    
    def __init__(self, dataset_name: str = "ogbl-collab", root: str = "data/raw/ogb"):
        self.dataset_name = dataset_name
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.dataset = None
        self.split_edge = None
    
    def load(self):
        """Load OGB dataset."""
        if not OGB_AVAILABLE:
            raise ImportError("OGB is not installed. Please install it with: pip install ogb")
        print(f"Loading OGB dataset: {self.dataset_name}")
        self.dataset = PygLinkPropPredDataset(name=self.dataset_name, root=self.root)
        self.split_edge = self.dataset.get_edge_split()
        print(f"Dataset loaded: {self.dataset[0]}")
        return self.dataset[0]
    
    def get_data(self) -> Data:
        """Get the graph data."""
        if self.dataset is None:
            self.load()
        return self.dataset[0]
    
    def get_splits(self):
        """Get train/val/test splits."""
        if self.split_edge is None:
            self.load()
        return self.split_edge
    
    def process_and_save(self, output_dir: str = "data/processed/ogb"):
        """Process and save OGB dataset."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        data = self.get_data()
        splits = self.get_splits()
        
        # Save data
        output_path = os.path.join(output_dir, f"{self.dataset_name.replace('ogbl-', '')}.pt")
        torch.save(data, output_path)
        print(f"Saved data to {output_path}")
        
        # Save splits
        splits_path = os.path.join(output_dir, f"{self.dataset_name.replace('ogbl-', '')}_splits.pt")
        torch.save(splits, splits_path)
        print(f"Saved splits to {splits_path}")
        
        return data, splits


def load_ogb_dataset(dataset_name: str = "ogbl-collab"):
    """Convenience function to load OGB dataset."""
    loader = OGBDatasetLoader(dataset_name)
    return loader.get_data(), loader.get_splits()

