"""
Unit tests for preprocessing modules.
"""

import torch
import pytest
from src.data.preprocessing import GraphPreprocessor, prepare_link_prediction_data
from src.data.facebook_loader import create_synthetic_dataset


def test_graph_preprocessor():
    """Test GraphPreprocessor."""
    preprocessor = GraphPreprocessor(seed=42)
    
    # Create test data
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    num_nodes = 3
    
    # Test random split
    train, val, test = preprocessor.random_split(edge_index, train_ratio=0.6, val_ratio=0.2)
    assert train.size(1) + val.size(1) + test.size(1) == edge_index.size(1)
    
    # Test negative sampling
    neg_edges = preprocessor.negative_sampling(edge_index, num_nodes, num_neg_samples=5)
    assert neg_edges.size(0) == 2
    assert neg_edges.size(1) <= 5


def test_prepare_link_prediction_data():
    """Test prepare_link_prediction_data."""
    # Create synthetic data
    data = create_synthetic_dataset(num_nodes=50, num_edges=100, feature_dim=16)
    
    # Prepare link prediction data
    link_data = prepare_link_prediction_data(data, train_ratio=0.7, val_ratio=0.15, seed=42)
    
    # Check keys
    assert 'train_edges' in link_data
    assert 'train_labels' in link_data
    assert 'val_edges' in link_data
    assert 'val_labels' in link_data
    assert 'test_edges' in link_data
    assert 'test_labels' in link_data
    
    # Check shapes
    assert link_data['train_edges'].size(0) == 2
    assert link_data['train_labels'].size(0) == link_data['train_edges'].size(1)
    
    # Check labels
    assert link_data['train_labels'].min() >= 0
    assert link_data['train_labels'].max() <= 1


if __name__ == '__main__':
    pytest.main([__file__])

