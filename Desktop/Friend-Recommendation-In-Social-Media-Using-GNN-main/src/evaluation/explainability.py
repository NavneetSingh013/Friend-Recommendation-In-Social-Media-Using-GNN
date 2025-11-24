"""
Explainability utilities for friend recommendations.
Computes mutual friends, shared groups, similarity scores, etc.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import Dict, List, Tuple
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


def compute_mutual_friends(edge_index: torch.Tensor, user_id: int, friend_id: int,
                          num_nodes: int) -> int:
    """
    Compute number of mutual friends between two users.
    
    Args:
        edge_index: Edge index
        user_id: User node ID
        friend_id: Friend node ID
        num_nodes: Number of nodes
        
    Returns:
        Number of mutual friends
    """
    # Build adjacency sets
    user_neighbors = set()
    friend_neighbors = set()
    
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src == user_id:
            user_neighbors.add(dst)
        if dst == user_id:
            user_neighbors.add(src)
        if src == friend_id:
            friend_neighbors.add(dst)
        if dst == friend_id:
            friend_neighbors.add(src)
    
    mutual = user_neighbors & friend_neighbors
    return len(mutual)


def compute_shared_groups(data: Data, user_id: int, friend_id: int) -> List[str]:
    """
    Compute shared groups between two users.
    
    Args:
        data: Graph data (with circles attribute)
        user_id: User node ID (mapped index)
        friend_id: Friend node ID (mapped index)
        
    Returns:
        List of shared group names
    """
    if not hasattr(data, 'circles') or not data.circles:
        return []
    
    # Handle node ID mapping: circles contain original node IDs
    # We need to check if the original node IDs (from idx_to_node) are in the circles
    user_groups = []
    friend_groups = []
    
    # Get original node IDs if mapping exists
    if hasattr(data, 'idx_to_node') and data.idx_to_node:
        # idx_to_node maps: index -> original_node_id
        if user_id < len(data.idx_to_node):
            user_original_id = data.idx_to_node[user_id]
        else:
            user_original_id = user_id
            
        if friend_id < len(data.idx_to_node):
            friend_original_id = data.idx_to_node[friend_id]
        else:
            friend_original_id = friend_id
    else:
        # No mapping, assume IDs are already correct
        user_original_id = user_id
        friend_original_id = friend_id
    
    # Check circles: circle members are original node IDs
    for circle_name, members in data.circles.items():
        # Check if the original node IDs are in the circle members
        if user_original_id in members:
            user_groups.append(circle_name)
        if friend_original_id in members:
            friend_groups.append(circle_name)
    
    shared = set(user_groups) & set(friend_groups)
    return list(shared)


def compute_profile_similarity(data: Data, user_id: int, friend_id: int) -> float:
    """
    Compute profile similarity (cosine similarity of node features).
    
    Args:
        data: Graph data
        user_id: User node ID
        friend_id: Friend node ID
        
    Returns:
        Similarity score [0, 1]
    """
    # Handle node ID mapping
    if hasattr(data, 'node_to_idx') and data.node_to_idx:
        user_idx = data.node_to_idx.get(user_id, user_id)
        friend_idx = data.node_to_idx.get(friend_id, friend_id)
    else:
        # Assume sequential node IDs, but check bounds
        user_idx = user_id if user_id < data.num_nodes else None
        friend_idx = friend_id if friend_id < data.num_nodes else None
    
    if user_idx is None or friend_idx is None:
        return 0.0
    
    if user_idx >= data.x.size(0) or friend_idx >= data.x.size(0):
        return 0.0
    
    user_features = data.x[user_idx].unsqueeze(0)
    friend_features = data.x[friend_idx].unsqueeze(0)
    
    # Handle CPU/GPU
    if user_features.is_cuda:
        user_features = user_features.cpu()
    if friend_features.is_cuda:
        friend_features = friend_features.cpu()
    
    similarity = cosine_similarity(user_features.numpy(), friend_features.numpy())[0, 0]
    return float(similarity)


def compute_shortest_path(edge_index: torch.Tensor, user_id: int, friend_id: int,
                         num_nodes: int) -> int:
    """
    Compute shortest path length between two users.
    
    Args:
        edge_index: Edge index
        user_id: User node ID
        friend_id: Friend node ID
        num_nodes: Number of nodes
        
    Returns:
        Shortest path length (-1 if no path exists)
    """
    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        G.add_edge(src, dst)
    
    try:
        path_length = nx.shortest_path_length(G, user_id, friend_id)
        return path_length
    except nx.NetworkXNoPath:
        return -1


def explain_recommendation(data: Data, user_id: int, friend_id: int,
                          model_score: float) -> Dict:
    """
    Generate explanation for a friend recommendation.
    
    Args:
        data: Graph data
        user_id: User node ID
        friend_id: Recommended friend node ID
        model_score: Model prediction score
        
    Returns:
        Explanation dictionary
    """
    # Compute mutual friends
    mutual_friends = compute_mutual_friends(data.edge_index, user_id, friend_id, data.num_nodes)
    
    # Compute shared groups
    shared_groups = compute_shared_groups(data, user_id, friend_id)
    
    # Compute profile similarity
    similarity = compute_profile_similarity(data, user_id, friend_id)
    
    # Compute shortest path
    path_length = compute_shortest_path(data.edge_index, user_id, friend_id, data.num_nodes)
    
    explanation = {
        'user_id': int(user_id),
        'friend_id': int(friend_id),
        'confidence_score': float(model_score),
        'mutual_friends': int(mutual_friends),
        'shared_groups': shared_groups,
        'num_shared_groups': len(shared_groups),
        'profile_similarity': float(similarity),
        'shortest_path_length': int(path_length) if path_length >= 0 else None,
        'path_evidence': f"{path_length}-hop" if path_length >= 0 else "No path"
    }
    
    return explanation

