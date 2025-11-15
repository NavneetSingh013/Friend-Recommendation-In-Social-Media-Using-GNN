"""
Evaluation script for GNN models.
"""

import argparse
import os
import sys
import torch
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import GraphSAGE, GAT, SEAL
from src.models.link_predictor import GraphSAGELinkPredictor, GATLinkPredictor
from src.evaluation import compute_metrics, compute_ranking_metrics
from src.training import set_seed
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(dataset: str, data_dir: str = 'data/processed'):
    """Load dataset."""
    if dataset == 'facebook':
        data_path = os.path.join(data_dir, 'facebook_combined.pt')
        link_data_path = os.path.join(data_dir, 'facebook_link_data.pt')
    elif dataset == 'synthetic':
        data_path = os.path.join(data_dir, 'synthetic.pt')
        link_data_path = os.path.join(data_dir, 'synthetic_link_data.pt')
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Load with weights_only=False for PyTorch 2.6+ compatibility
    data = torch.load(data_path, weights_only=False)
    link_data = torch.load(link_data_path, weights_only=False)
    
    return data, link_data


def create_model(model_name: str, input_dim: int, config: dict, device: torch.device):
    """Create model."""
    hidden_dim = config.get('hidden_dim', 64)
    embedding_dim = config.get('embedding_dim', 64)
    num_layers = config.get('num_layers', 2)
    dropout = config.get('dropout', 0.5)
    
    if model_name == 'graphsage':
        aggregator = config.get('aggregator', 'mean')
        predictor_method = config.get('predictor_method', 'mlp')
        model = GraphSAGELinkPredictor(
            input_dim, hidden_dim, embedding_dim, num_layers, dropout, aggregator, predictor_method
        )
    elif model_name == 'gat':
        num_heads = config.get('num_heads', 4)
        predictor_method = config.get('predictor_method', 'mlp')
        model = GATLinkPredictor(
            input_dim, hidden_dim, embedding_dim, num_layers, num_heads, dropout, predictor_method
        )
    elif model_name == 'seal':
        num_hops = config.get('num_hops', 2)
        pool = config.get('pool', 'mean')
        model = SEAL(
            input_dim, hidden_dim, num_layers, num_hops, dropout, pool
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description='Evaluate GNN model')
    parser.add_argument('--model', type=str, choices=['graphsage', 'gat', 'seal'],
                       required=True, help='Model to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Checkpoint path')
    parser.add_argument('--dataset', type=str, default='facebook',
                       help='Dataset to use')
    parser.add_argument('--config', type=str, required=True,
                       help='Config file path')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Data directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Load data
    data, link_data = load_data(args.dataset, args.data_dir)
    
    # Create model
    input_dim = data.x.size(1)
    model = create_model(args.model, input_dim, config, device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Evaluate on test set
    model.eval()
    data = data.to(device)
    test_edges = link_data['test_edges'].to(device)
    test_labels = link_data['test_labels'].to(device)
    
    with torch.no_grad():
        if hasattr(model, 'encoder'):
            scores = model(data.x, data.edge_index, test_edges)
        else:
            scores = model(data.edge_index, data.x, test_edges)
        
        scores = torch.sigmoid(scores).cpu().numpy()
        labels = test_labels.cpu().numpy()
    
    # Compute metrics
    print("\n=== Classification Metrics ===")
    metrics = compute_metrics(scores, labels)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\n=== Ranking Metrics ===")
    ranking_metrics = compute_ranking_metrics(scores, labels, k_values=[5, 10, 20, 50])
    for key, value in ranking_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == '__main__':
    main()

