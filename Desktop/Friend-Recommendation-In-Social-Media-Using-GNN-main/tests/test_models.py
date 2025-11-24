"""
Unit tests for model architectures.
"""

import torch
import pytest
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT
from src.models.link_predictor import LinkPredictor, GraphSAGELinkPredictor
from src.data.facebook_loader import create_synthetic_dataset


def test_graphsage():
    """Test GraphSAGE model."""
    data = create_synthetic_dataset(num_nodes=50, num_edges=100, feature_dim=16)
    
    model = GraphSAGE(
        input_dim=data.x.size(1),
        hidden_dim=32,
        output_dim=16,
        num_layers=2,
        dropout=0.5,
        aggregator='mean'
    )
    
    # Forward pass
    embeddings = model(data.x, data.edge_index)
    assert embeddings.size(0) == data.num_nodes
    assert embeddings.size(1) == 16


def test_gat():
    """Test GAT model."""
    data = create_synthetic_dataset(num_nodes=50, num_edges=100, feature_dim=16)
    
    model = GAT(
        input_dim=data.x.size(1),
        hidden_dim=32,
        output_dim=16,
        num_layers=2,
        num_heads=4,
        dropout=0.5
    )
    
    # Forward pass
    embeddings = model(data.x, data.edge_index)
    assert embeddings.size(0) == data.num_nodes
    assert embeddings.size(1) == 16


def test_link_predictor():
    """Test LinkPredictor."""
    data = create_synthetic_dataset(num_nodes=50, num_edges=100, feature_dim=16)
    
    # Create embeddings
    model = GraphSAGE(
        input_dim=data.x.size(1),
        hidden_dim=32,
        output_dim=16,
        num_layers=2
    )
    embeddings = model(data.x, data.edge_index)
    
    # Test link predictor
    predictor = LinkPredictor(embedding_dim=16, hidden_dim=32, method='mlp')
    pred_edges = data.edge_index[:, :10]
    scores = predictor(embeddings, pred_edges)
    assert scores.size(0) == pred_edges.size(1)


def test_graphsage_link_predictor():
    """Test GraphSAGELinkPredictor."""
    data = create_synthetic_dataset(num_nodes=50, num_edges=100, feature_dim=16)
    
    model = GraphSAGELinkPredictor(
        input_dim=data.x.size(1),
        hidden_dim=32,
        embedding_dim=16,
        num_layers=2,
        dropout=0.5
    )
    
    # Forward pass
    pred_edges = data.edge_index[:, :10]
    scores = model(data.x, data.edge_index, pred_edges)
    assert scores.size(0) == pred_edges.size(1)


if __name__ == '__main__':
    pytest.main([__file__])

