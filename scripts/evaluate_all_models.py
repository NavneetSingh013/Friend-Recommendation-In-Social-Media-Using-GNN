"""
Comprehensive evaluation script for all trained GNN models.
Evaluates all available models and displays comprehensive metrics.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.link_predictor import GraphSAGELinkPredictor, GATLinkPredictor
from src.models.seal import SEAL
from src.evaluation.metrics import compute_metrics, compute_ranking_metrics
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(dataset: str):
    """Load dataset."""
    if dataset == 'facebook':
        data_path = 'data/processed/facebook_combined.pt'
        link_data_path = 'data/processed/facebook_link_data.pt'
    elif dataset == 'synthetic':
        data_path = 'data/processed/synthetic.pt'
        link_data_path = 'data/processed/synthetic_link_data.pt'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    data = torch.load(data_path, weights_only=False)
    link_data = torch.load(link_data_path, weights_only=False)
    return data, link_data


def evaluate_model(model_name: str, model, data, link_data, device):
    """Evaluate a single model and return all metrics."""
    model.eval()
    data = data.to(device)
    test_edges = link_data['test_edges'].to(device)
    test_labels = link_data['test_labels'].to(device)
    
    print(f"\n[{model_name}] Generating predictions...")
    with torch.no_grad():
        # Handle different model types
        if hasattr(model, 'encoder'):
            # GraphSAGE or GAT with encoder
            scores = model(data.x, data.edge_index, test_edges)
        else:
            # SEAL model
            batch_size = 100  # SEAL needs smaller batches
            scores_list = []
            for i in range(0, test_edges.size(1), batch_size):
                batch_edges = test_edges[:, i:i+batch_size]
                try:
                    batch_scores = model(data.edge_index, data.x, batch_edges, batch_size=len(batch_edges[0]))
                except:
                    batch_scores = model(data.edge_index, data.x, batch_edges)
                scores_list.append(batch_scores)
            scores = torch.cat(scores_list, dim=0)
        
        scores = torch.sigmoid(scores).cpu().numpy().flatten()
        labels = test_labels.cpu().numpy().flatten()
    
    # Compute all metrics
    print(f"[{model_name}] Computing metrics...")
    classification_metrics = compute_metrics(scores, labels)
    ranking_metrics = compute_ranking_metrics(scores, labels, k_values=[5, 10, 20, 50])
    
    return {
        'model': model_name,
        **classification_metrics,
        **ranking_metrics
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    
    # Try both datasets
    datasets_to_try = ['facebook', 'synthetic']
    all_results = {}
    
    for dataset in datasets_to_try:
        try:
            print(f"\n{'='*80}")
            print(f"Evaluating on {dataset.upper()} dataset")
            print(f"{'='*80}")
            
            # Load data
            data, link_data = load_data(dataset)
            input_dim = data.x.size(1)
            print(f"Dataset loaded: {data.num_nodes} nodes, {data.edge_index.size(1)//2} edges")
            print(f"Test set: {link_data['test_edges'].size(1)} edges")
            
            results = {}
            
            # Evaluate GraphSAGE
            checkpoint_path = f"data/checkpoints/graphsage/best_model.pt"
            config_path = "configs/graphsage_config.yaml"
            if os.path.exists(checkpoint_path):
                try:
                    config = load_config(config_path)
                    hidden_dim = config.get('hidden_dim', 64)
                    embedding_dim = config.get('embedding_dim', 64)
                    model = GraphSAGELinkPredictor(
                        input_dim, hidden_dim, embedding_dim, 2, 0.5,
                        aggregator='mean', predictor_method='mlp'
                    ).to(device)
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    metrics = evaluate_model(f"GraphSAGE ({dataset})", model, data, link_data, device)
                    results['GraphSAGE'] = metrics
                except Exception as e:
                    print(f"Error evaluating GraphSAGE: {e}")
            else:
                print(f"GraphSAGE checkpoint not found: {checkpoint_path}")
            
            # Evaluate GAT
            checkpoint_path = f"data/checkpoints/gat/best_model.pt"
            config_path = "configs/gat_config.yaml"
            if os.path.exists(checkpoint_path):
                try:
                    config = load_config(config_path)
                    hidden_dim = config.get('hidden_dim', 64)
                    embedding_dim = config.get('embedding_dim', 64)
                    model = GATLinkPredictor(
                        input_dim, hidden_dim, embedding_dim, 2,
                        num_heads=4, dropout=0.5, predictor_method='mlp'
                    ).to(device)
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    metrics = evaluate_model(f"GAT ({dataset})", model, data, link_data, device)
                    results['GAT'] = metrics
                except Exception as e:
                    print(f"[WARNING] Error evaluating GAT: {e}")
            else:
                print(f"[INFO] GAT checkpoint not found: {checkpoint_path}")
            
            # Evaluate SEAL (skip if dimension mismatch, needs retraining)
            checkpoint_path = f"data/checkpoints/seal/best_model.pt"
            config_path = "configs/seal_config.yaml"
            if os.path.exists(checkpoint_path):
                try:
                    config = load_config(config_path)
                    hidden_dim = config.get('hidden_dim', 64)
                    num_hops = config.get('num_hops', 1)
                    model = SEAL(
                        input_dim, hidden_dim, 2, num_hops, 0.5, pool='mean'
                    ).to(device)
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    metrics = evaluate_model(f"SEAL ({dataset})", model, data, link_data, device)
                    results['SEAL'] = metrics
                except Exception as e:
                    print(f"[WARNING] Error evaluating SEAL (may need retraining with correct config): {e}")
            else:
                print(f"[INFO] SEAL checkpoint not found: {checkpoint_path}")
            
            if results:
                all_results[dataset] = results
                # Display results
                print(f"\n{'='*80}")
                print(f"RESULTS SUMMARY - {dataset.upper()}")
                print(f"{'='*80}")
                
                # Create DataFrame
                df = pd.DataFrame(results).T
                
                # Display classification metrics
                print("\n[CLASSIFICATION METRICS]")
                print("-" * 80)
                class_cols = ['auc', 'ap', 'accuracy', 'precision', 'recall', 'f1']
                print(df[class_cols].to_string())
                
                # Display ranking metrics
                print("\n[RANKING METRICS]")
                print("-" * 80)
                rank_cols = ['precision@5', 'precision@10', 'precision@20', 'precision@50',
                            'recall@5', 'recall@10', 'ndcg@5', 'ndcg@10', 'map', 'map@10']
                available_rank_cols = [c for c in rank_cols if c in df.columns]
                if available_rank_cols:
                    print(df[available_rank_cols].to_string())
                
        except FileNotFoundError:
            print(f"\n[WARNING] Dataset '{dataset}' not found. Skipping...")
            continue
        except Exception as e:
            print(f"\n[ERROR] Error evaluating on {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if all_results:
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}")
        
        # Save to CSV
        all_rows = []
        for dataset, results in all_results.items():
            for model_name, metrics in results.items():
                metrics['dataset'] = dataset
                all_rows.append(metrics)
        
        df_all = pd.DataFrame(all_rows)
        output_path = 'model_evaluation_results.csv'
        df_all.to_csv(output_path, index=False)
        print(f"\n[OK] Results saved to: {output_path}")
    else:
        print("\n[WARNING] No models were successfully evaluated.")


if __name__ == '__main__':
    main()

