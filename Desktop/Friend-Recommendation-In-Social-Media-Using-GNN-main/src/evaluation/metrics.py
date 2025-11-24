"""
Evaluation metrics for link prediction.
Includes AUC, AP, Precision@K, Recall@K, NDCG@K, MAP@K.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Tuple
import torch.nn.functional as F


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Prediction scores
        labels: True labels
        
    Returns:
        Dictionary of metrics
    """
    # Convert to binary predictions
    binary_preds = (predictions > 0.5).astype(int)
    
    # Compute AUC
    try:
        auc = roc_auc_score(labels, predictions)
    except ValueError:
        auc = 0.0
    
    # Compute AP
    try:
        ap = average_precision_score(labels, predictions)
    except ValueError:
        ap = 0.0
    
    # Compute accuracy
    accuracy = (binary_preds == labels).mean()
    
    # Compute precision, recall, F1
    tp = ((binary_preds == 1) & (labels == 1)).sum()
    fp = ((binary_preds == 1) & (labels == 0)).sum()
    fn = ((binary_preds == 0) & (labels == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'auc': auc,
        'ap': ap,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_ranking_metrics(predictions: np.ndarray, labels: np.ndarray,
                           k_values: list = [5, 10, 20, 50]) -> Dict[str, float]:
    """
    Compute ranking metrics (Precision@K, Recall@K, NDCG@K, MAP@K).
    
    Args:
        predictions: Prediction scores
        labels: True labels
        k_values: List of K values
        
    Returns:
        Dictionary of metrics
    """
    # Sort by prediction score
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_labels = labels[sorted_indices]
    
    results = {}
    
    for k in k_values:
        # Top-K predictions
        top_k_labels = sorted_labels[:k]
        
        # Precision@K
        precision_k = top_k_labels.sum() / k
        results[f'precision@{k}'] = precision_k
        
        # Recall@K
        total_positives = labels.sum()
        recall_k = top_k_labels.sum() / total_positives if total_positives > 0 else 0.0
        results[f'recall@{k}'] = recall_k
        
        # NDCG@K
        dcg_k = 0.0
        for i, label in enumerate(top_k_labels):
            dcg_k += label / np.log2(i + 2)
        
        # Ideal DCG
        ideal_labels = np.sort(labels)[::-1][:k]
        idcg_k = 0.0
        for i, label in enumerate(ideal_labels):
            idcg_k += label / np.log2(i + 2)
        
        ndcg_k = dcg_k / idcg_k if idcg_k > 0 else 0.0
        results[f'ndcg@{k}'] = ndcg_k
    
    # MAP (Mean Average Precision)
    # Average Precision for each query
    ap_scores = []
    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            # Precision at this position
            precision_at_i = sorted_labels[:i+1].sum() / (i + 1)
            ap_scores.append(precision_at_i)
    
    map_score = np.mean(ap_scores) if ap_scores else 0.0
    results['map'] = map_score
    
    # MAP@K
    for k in k_values:
        ap_scores_k = []
        top_k_labels = sorted_labels[:k]
        for i in range(min(k, len(sorted_labels))):
            if top_k_labels[i] == 1:
                precision_at_i = top_k_labels[:i+1].sum() / (i + 1)
                ap_scores_k.append(precision_at_i)
        map_k = np.mean(ap_scores_k) if ap_scores_k else 0.0
        results[f'map@{k}'] = map_k
    
    return results


def get_top_k_recommendations(model, data, user_id: int, candidate_nodes: torch.Tensor,
                             k: int = 10, device: torch.device = None, batch_size: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get top-K friend recommendations for a user.
    
    Args:
        model: Trained model
        data: Graph data
        user_id: User node ID
        candidate_nodes: Candidate nodes to recommend
        k: Number of recommendations
        device: Device
        
    Returns:
        Top-K node IDs and scores
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    data = data.to(device)
    
    # Process in batches to avoid memory issues
    all_scores = []
    num_candidates = candidate_nodes.size(0)
    
    with torch.no_grad():
        for i in range(0, num_candidates, batch_size):
            batch_candidates = candidate_nodes[i:i+batch_size].to(device)
            batch_size_actual = batch_candidates.size(0)
            
            # Create candidate edges
            user_tensor = torch.full((batch_size_actual,), user_id, dtype=torch.long, device=device)
            candidate_edges = torch.stack([user_tensor, batch_candidates], dim=0)
            
            # Predict
            if hasattr(model, 'encoder'):
                scores = model(data.x, data.edge_index, candidate_edges)
            elif hasattr(model, 'forward'):
                # Try SEAL-style forward
                try:
                    scores = model(data.edge_index, data.x, candidate_edges, batch_size=batch_size_actual)
                except:
                    scores = model(data.edge_index, data.x, candidate_edges)
            else:
                raise ValueError("Model does not have expected interface")
            
            scores = torch.sigmoid(scores).cpu()
            all_scores.append(scores)
    
    # Concatenate all scores
    all_scores = torch.cat(all_scores, dim=0)
    
    # Get top-K
    top_k_indices = torch.topk(all_scores, k=min(k, all_scores.size(0))).indices
    top_k_nodes = candidate_nodes[top_k_indices]
    top_k_scores = all_scores[top_k_indices]
    
    return top_k_nodes, top_k_scores

