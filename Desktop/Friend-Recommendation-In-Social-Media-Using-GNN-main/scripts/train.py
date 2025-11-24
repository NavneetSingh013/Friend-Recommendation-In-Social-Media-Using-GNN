"""
Training script for GNN models.
"""

import argparse
import os
import sys
import yaml
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import GraphSAGE, GAT, SEAL
from src.models.link_predictor import GraphSAGELinkPredictor, GATLinkPredictor
from src.training import Trainer, set_seed, EarlyStopping
from src.data.preprocessing import prepare_link_prediction_data
import torch


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
    parser = argparse.ArgumentParser(description='Train GNN model')
    parser.add_argument('--model', type=str, choices=['graphsage', 'gat', 'seal'],
                       required=True, help='Model to train')
    parser.add_argument('--dataset', type=str, default='facebook',
                       help='Dataset to use')
    parser.add_argument('--config', type=str, required=True,
                       help='Config file path')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='data/checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Load data
    print(f"Loading dataset: {args.dataset}")
    data, link_data = load_data(args.dataset, args.data_dir)
    print(f"Data: {data.num_nodes} nodes, {data.edge_index.size(1) // 2} edges")
    
    # Create model
    input_dim = data.x.size(1)
    model = create_model(args.model, input_dim, config, device)
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create trainer
    lr = config.get('lr', 0.01)
    weight_decay = config.get('weight_decay', 5e-4)
    trainer = Trainer(model, device, lr=lr, weight_decay=weight_decay)
    
    # Prepare data
    train_data = {
        'edges': link_data['train_edges'],
        'labels': link_data['train_labels']
    }
    val_data = {
        'edges': link_data['val_edges'],
        'labels': link_data['val_labels']
    }
    
    # Early stopping
    patience = config.get('patience', 10)
    early_stopping = EarlyStopping(patience=patience)
    
    # Train
    num_epochs = config.get('num_epochs', 100)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.model)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save with dataset-specific name
    dataset_checkpoint_path = os.path.join(checkpoint_dir, f'{args.dataset}_best_model.pt')
    
    history = trainer.train(
        data, train_data, val_data, num_epochs, early_stopping, checkpoint_dir
    )
    
    # Rename the saved checkpoint to be dataset-specific
    default_checkpoint = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(default_checkpoint):
        if os.path.exists(dataset_checkpoint_path):
            os.remove(dataset_checkpoint_path)  # Remove old if exists
        os.rename(default_checkpoint, dataset_checkpoint_path)
        print(f"Saved checkpoint to {dataset_checkpoint_path}")
    
    print("Training completed!")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")


if __name__ == '__main__':
    main()

