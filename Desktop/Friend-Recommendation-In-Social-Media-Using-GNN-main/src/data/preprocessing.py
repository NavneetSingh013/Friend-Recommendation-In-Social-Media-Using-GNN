"""
Graph preprocessing utilities.
Includes temporal splitting, negative sampling, and feature engineering.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import random


class GraphPreprocessor:
    """Graph preprocessing utilities."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
    
    def temporal_split(self, edge_index: torch.Tensor, edge_times: Optional[torch.Tensor] = None,
                      train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[torch.Tensor, ...]:
        """
        Split edges temporally.
        
        Args:
            edge_index: Edge index tensor [2, num_edges]
            edge_times: Edge timestamps (optional)
            train_ratio: Training ratio
            val_ratio: Validation ratio
            
        Returns:
            train_edge_index, val_edge_index, test_edge_index
        """
        num_edges = edge_index.size(1)
        
        if edge_times is not None:
            # Sort by time
            sorted_indices = torch.argsort(edge_times)
            edge_index = edge_index[:, sorted_indices]
            edge_times = edge_times[sorted_indices]
        else:
            # Random shuffle
            indices = torch.randperm(num_edges)
            edge_index = edge_index[:, indices]
        
        # Split
        train_size = int(num_edges * train_ratio)
        val_size = int(num_edges * val_ratio)
        
        train_edge_index = edge_index[:, :train_size]
        val_edge_index = edge_index[:, train_size:train_size + val_size]
        test_edge_index = edge_index[:, train_size + val_size:]
        
        return train_edge_index, val_edge_index, test_edge_index
    
    def random_split(self, edge_index: torch.Tensor, train_ratio: float = 0.7,
                    val_ratio: float = 0.15, seed: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """
        Random split of edges.
        
        Args:
            edge_index: Edge index tensor [2, num_edges]
            train_ratio: Training ratio
            val_ratio: Validation ratio
            seed: Random seed
            
        Returns:
            train_edge_index, val_edge_index, test_edge_index
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        num_edges = edge_index.size(1)
        indices = torch.randperm(num_edges)
        edge_index = edge_index[:, indices]
        
        train_size = int(num_edges * train_ratio)
        val_size = int(num_edges * val_ratio)
        
        train_edge_index = edge_index[:, :train_size]
        val_edge_index = edge_index[:, train_size:train_size + val_size]
        test_edge_index = edge_index[:, train_size + val_size:]
        
        return train_edge_index, val_edge_index, test_edge_index
    
    def negative_sampling(self, edge_index: torch.Tensor, num_nodes: int,
                         num_neg_samples: Optional[int] = None,
                         ratio: float = 1.0) -> torch.Tensor:
        """
        Generate negative samples (non-edges).
        
        Args:
            edge_index: Positive edge index [2, num_edges]
            num_nodes: Number of nodes
            num_neg_samples: Number of negative samples (if None, uses ratio * num_edges)
            ratio: Ratio of negative to positive samples
            
        Returns:
            Negative edge index [2, num_neg_samples]
        """
        if num_neg_samples is None:
            num_neg_samples = int(edge_index.size(1) * ratio)
        
        # Create set of existing edges (both directions)
        edge_set = set()
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_set.add((src, dst))
            edge_set.add((dst, src))  # Undirected
        
        # Generate negative samples
        neg_edges = []
        attempts = 0
        max_attempts = num_neg_samples * 10
        
        while len(neg_edges) < num_neg_samples and attempts < max_attempts:
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            
            if src != dst and (src, dst) not in edge_set:
                neg_edges.append([src, dst])
                edge_set.add((src, dst))  # Avoid duplicates
                edge_set.add((dst, src))
            
            attempts += 1
        
        if len(neg_edges) < num_neg_samples:
            print(f"Warning: Only generated {len(neg_edges)} negative samples out of {num_neg_samples} requested")
        
        return torch.tensor(neg_edges, dtype=torch.long).t().contiguous()
    
    def create_link_labels(self, pos_edges: torch.Tensor, neg_edges: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create edge labels for link prediction.
        
        Args:
            pos_edges: Positive edges [2, num_pos]
            neg_edges: Negative edges [2, num_neg]
            
        Returns:
            edges: Concatenated edges [2, num_pos + num_neg]
            labels: Labels [num_pos + num_neg] (1 for positive, 0 for negative)
        """
        edges = torch.cat([pos_edges, neg_edges], dim=1)
        labels = torch.cat([
            torch.ones(pos_edges.size(1)),
            torch.zeros(neg_edges.size(1))
        ])
        return edges, labels


def prepare_link_prediction_data(data: Data, train_ratio: float = 0.7, val_ratio: float = 0.15,
                                 neg_ratio: float = 1.0, temporal: bool = False,
                                 seed: int = 42) -> dict:
    """
    Prepare data for link prediction task.
    
    Args:
        data: PyG Data object
        train_ratio: Training ratio
        val_ratio: Validation ratio
        neg_ratio: Negative sampling ratio
        temporal: Whether to use temporal splitting
        seed: Random seed
        
    Returns:
        Dictionary with train/val/test splits and labels
    """
    preprocessor = GraphPreprocessor(seed=seed)
    
    # Remove self-loops and duplicate edges
    edge_index = data.edge_index
    # Keep only upper triangle for undirected graphs
    mask = edge_index[0] < edge_index[1]
    edge_index = edge_index[:, mask]
    
    # Split edges
    if temporal and hasattr(data, 'edge_times'):
        train_edges, val_edges, test_edges = preprocessor.temporal_split(
            edge_index, data.edge_times, train_ratio, val_ratio
        )
    else:
        train_edges, val_edges, test_edges = preprocessor.random_split(
            edge_index, train_ratio, val_ratio, seed
        )
    
    # Generate negative samples
    train_neg = preprocessor.negative_sampling(train_edges, data.num_nodes, ratio=neg_ratio)
    val_neg = preprocessor.negative_sampling(
        torch.cat([train_edges, val_edges], dim=1), data.num_nodes, 
        num_neg_samples=val_edges.size(1)
    )
    test_neg = preprocessor.negative_sampling(
        torch.cat([train_edges, val_edges, test_edges], dim=1), data.num_nodes,
        num_neg_samples=test_edges.size(1)
    )
    
    # Create labels
    train_edges_labeled, train_labels = preprocessor.create_link_labels(train_edges, train_neg)
    val_edges_labeled, val_labels = preprocessor.create_link_labels(val_edges, val_neg)
    test_edges_labeled, test_labels = preprocessor.create_link_labels(test_edges, test_neg)
    
    return {
        'train_edges': train_edges_labeled,
        'train_labels': train_labels,
        'val_edges': val_edges_labeled,
        'val_labels': val_labels,
        'test_edges': test_edges_labeled,
        'test_labels': test_labels,
        'train_pos': train_edges,
        'val_pos': val_edges,
        'test_pos': test_edges,
        'train_neg': train_neg,
        'val_neg': val_neg,
        'test_neg': test_neg
    }

