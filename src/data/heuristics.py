"""
Graph heuristics for link prediction.
Includes common neighbors, Jaccard, Adamic-Adar, and preferential attachment.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple
import networkx as nx


def common_neighbors(edge_index: torch.Tensor, num_nodes: int) -> Dict[Tuple[int, int], int]:
    """
    Compute common neighbors for all node pairs.
    
    Args:
        edge_index: Edge index [2, num_edges]
        num_nodes: Number of nodes
        
    Returns:
        Dictionary mapping (src, dst) -> number of common neighbors
    """
    # Build adjacency list
    adj_list = {i: set() for i in range(num_nodes)}
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[src].add(dst)
        adj_list[dst].add(src)
    
    # Compute common neighbors for all pairs
    scores = {}
    nodes = list(range(num_nodes))
    
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            common = len(adj_list[node1] & adj_list[node2])
            scores[(node1, node2)] = common
            scores[(node2, node1)] = common
    
    return scores


def jaccard_coefficient(edge_index: torch.Tensor, num_nodes: int) -> Dict[Tuple[int, int], float]:
    """
    Compute Jaccard coefficient for all node pairs.
    
    Args:
        edge_index: Edge index [2, num_edges]
        num_nodes: Number of nodes
        
    Returns:
        Dictionary mapping (src, dst) -> Jaccard coefficient
    """
    # Build adjacency list
    adj_list = {i: set() for i in range(num_nodes)}
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[src].add(dst)
        adj_list[dst].add(src)
    
    # Compute Jaccard coefficient
    scores = {}
    nodes = list(range(num_nodes))
    
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            neighbors1 = adj_list[node1]
            neighbors2 = adj_list[node2]
            intersection = len(neighbors1 & neighbors2)
            union = len(neighbors1 | neighbors2)
            jaccard = intersection / union if union > 0 else 0.0
            scores[(node1, node2)] = jaccard
            scores[(node2, node1)] = jaccard
    
    return scores


def adamic_adar(edge_index: torch.Tensor, num_nodes: int) -> Dict[Tuple[int, int], float]:
    """
    Compute Adamic-Adar score for all node pairs.
    
    Args:
        edge_index: Edge index [2, num_edges]
        num_nodes: Number of nodes
        
    Returns:
        Dictionary mapping (src, dst) -> Adamic-Adar score
    """
    # Build adjacency list and degrees
    adj_list = {i: set() for i in range(num_nodes)}
    degrees = {i: 0 for i in range(num_nodes)}
    
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[src].add(dst)
        adj_list[dst].add(src)
        degrees[src] += 1
        degrees[dst] += 1
    
    # Compute Adamic-Adar
    scores = {}
    nodes = list(range(num_nodes))
    
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            common_neighbors = adj_list[node1] & adj_list[node2]
            score = sum(1.0 / np.log(max(degrees[cn], 1)) for cn in common_neighbors if degrees[cn] > 1)
            scores[(node1, node2)] = score
            scores[(node2, node1)] = score
    
    return scores


def preferential_attachment(edge_index: torch.Tensor, num_nodes: int) -> Dict[Tuple[int, int], int]:
    """
    Compute preferential attachment score for all node pairs.
    
    Args:
        edge_index: Edge index [2, num_edges]
        num_nodes: Number of nodes
        
    Returns:
        Dictionary mapping (src, dst) -> preferential attachment score
    """
    # Compute degrees
    degrees = {i: 0 for i in range(num_nodes)}
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        degrees[src] += 1
        degrees[dst] += 1
    
    # Compute preferential attachment
    scores = {}
    nodes = list(range(num_nodes))
    
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            score = degrees[node1] * degrees[node2]
            scores[(node1, node2)] = score
            scores[(node2, node1)] = score
    
    return scores


def compute_heuristics(data: Data, methods: list = None) -> Dict[str, Dict[Tuple[int, int], float]]:
    """
    Compute all heuristics for a graph.
    
    Args:
        data: PyG Data object
        methods: List of methods to compute ['common_neighbors', 'jaccard', 'adamic_adar', 'preferential_attachment']
        
    Returns:
        Dictionary mapping method name -> scores dictionary
    """
    if methods is None:
        methods = ['common_neighbors', 'jaccard', 'adamic_adar', 'preferential_attachment']
    
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    results = {}
    
    if 'common_neighbors' in methods:
        print("Computing common neighbors...")
        results['common_neighbors'] = common_neighbors(edge_index, num_nodes)
    
    if 'jaccard' in methods:
        print("Computing Jaccard coefficient...")
        results['jaccard'] = jaccard_coefficient(edge_index, num_nodes)
    
    if 'adamic_adar' in methods:
        print("Computing Adamic-Adar...")
        results['adamic_adar'] = adamic_adar(edge_index, num_nodes)
    
    if 'preferential_attachment' in methods:
        print("Computing preferential attachment...")
        results['preferential_attachment'] = preferential_attachment(edge_index, num_nodes)
    
    return results


def predict_links_heuristics(scores: Dict[Tuple[int, int], float], 
                             candidates: torch.Tensor,
                             top_k: int = 10) -> torch.Tensor:
    """
    Predict top-K links using heuristic scores.
    
    Args:
        scores: Dictionary mapping (src, dst) -> score
        candidates: Candidate edges [2, num_candidates]
        top_k: Number of top predictions
        
    Returns:
        Top-K edge indices [2, top_k]
    """
    candidate_scores = []
    candidate_edges = []
    
    for i in range(candidates.size(1)):
        src, dst = candidates[0, i].item(), candidates[1, i].item()
        edge = (src, dst)
        if edge in scores:
            candidate_scores.append(scores[edge])
            candidate_edges.append([src, dst])
    
    if not candidate_scores:
        return torch.tensor([], dtype=torch.long).reshape(2, 0)
    
    # Sort by score
    sorted_indices = np.argsort(candidate_scores)[::-1]
    top_k = min(top_k, len(sorted_indices))
    
    top_edges = torch.tensor([candidate_edges[i] for i in sorted_indices[:top_k]], dtype=torch.long).t()
    
    return top_edges

