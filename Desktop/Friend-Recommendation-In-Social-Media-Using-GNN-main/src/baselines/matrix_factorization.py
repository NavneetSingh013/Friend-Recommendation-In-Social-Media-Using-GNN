"""
Matrix Factorization baseline for link prediction.
"""

import torch
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple


class MatrixFactorizationBaseline:
    """Matrix Factorization baseline model."""
    
    def __init__(self, n_components: int = 64, alpha: float = 0.1):
        """
        Initialize Matrix Factorization.
        
        Args:
            n_components: Number of components
            alpha: Regularization parameter
        """
        self.n_components = n_components
        self.alpha = alpha
        self.model = None
        self.W = None
        self.H = None
    
    def train(self, edge_index: torch.Tensor, num_nodes: int):
        """
        Train matrix factorization model.
        
        Args:
            edge_index: Edge index [2, num_edges]
            num_nodes: Number of nodes
        """
        # Build adjacency matrix
        adj = np.zeros((num_nodes, num_nodes))
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj[src, dst] = 1.0
            adj[dst, src] = 1.0  # Undirected
        
        # Non-negative Matrix Factorization
        print("Training Matrix Factorization...")
        self.model = NMF(n_components=self.n_components, alpha=self.alpha, max_iter=200)
        self.W = self.model.fit_transform(adj)
        self.H = self.model.components_
    
    def predict(self, edges: torch.Tensor) -> np.ndarray:
        """
        Predict link existence.
        
        Args:
            edges: Edge index [2, num_edges]
            
        Returns:
            Prediction scores
        """
        if self.W is None or self.H is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Reconstruct adjacency matrix
        adj_reconstructed = np.dot(self.W, self.H)
        
        scores = []
        for i in range(edges.size(1)):
            src, dst = edges[0, i].item(), edges[1, i].item()
            score = adj_reconstructed[src, dst]
            scores.append(score)
        
        return np.array(scores)

