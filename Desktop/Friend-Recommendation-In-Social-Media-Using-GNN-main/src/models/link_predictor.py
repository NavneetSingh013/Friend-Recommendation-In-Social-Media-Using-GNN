"""
Link predictor models.
Combines node embeddings to predict link existence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class LinkPredictor(nn.Module):
    """Link predictor using node embeddings."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 64,
                 method: Literal['dot', 'cosine', 'mlp'] = 'mlp'):
        """
        Initialize link predictor.
        
        Args:
            embedding_dim: Node embedding dimension
            hidden_dim: Hidden dimension for MLP
            method: Prediction method ('dot', 'cosine', 'mlp')
        """
        super(LinkPredictor, self).__init__()
        
        self.method = method
        
        if method == 'mlp':
            self.predictor = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )
        elif method == 'dot':
            self.predictor = None
        elif method == 'cosine':
            self.predictor = None
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def forward(self, embeddings, edge_index):
        """
        Predict link existence.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            edge_index: Edge index [2, num_edges]
            
        Returns:
            Link scores [num_edges]
        """
        src_embeddings = embeddings[edge_index[0]]
        dst_embeddings = embeddings[edge_index[1]]
        
        if self.method == 'dot':
            scores = (src_embeddings * dst_embeddings).sum(dim=1)
        elif self.method == 'cosine':
            scores = F.cosine_similarity(src_embeddings, dst_embeddings, dim=1)
        else:  # mlp
            combined = torch.cat([src_embeddings, dst_embeddings], dim=1)
            scores = self.predictor(combined).squeeze()
        
        return scores
    
    def predict_edges(self, embeddings, edge_index):
        """Predict link scores for edges."""
        return self.forward(embeddings, edge_index)


class GraphSAGELinkPredictor(nn.Module):
    """Combined GraphSAGE + Link Predictor."""
    
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int,
                 num_layers: int = 2, dropout: float = 0.5,
                 aggregator: Literal['mean', 'max', 'lstm'] = 'mean',
                 predictor_method: Literal['dot', 'cosine', 'mlp'] = 'mlp'):
        super(GraphSAGELinkPredictor, self).__init__()
        
        from .graphsage import GraphSAGE
        
        self.encoder = GraphSAGE(
            input_dim, hidden_dim, embedding_dim, num_layers, dropout, aggregator
        )
        self.predictor = LinkPredictor(embedding_dim, hidden_dim, predictor_method)
    
    def forward(self, x, edge_index, pred_edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Graph edges for message passing
            pred_edge_index: Edges to predict
            
        Returns:
            Link scores
        """
        embeddings = self.encoder(x, edge_index)
        scores = self.predictor(embeddings, pred_edge_index)
        return scores
    
    def encode(self, x, edge_index):
        """Encode nodes to embeddings."""
        return self.encoder(x, edge_index)


class GATLinkPredictor(nn.Module):
    """Combined GAT + Link Predictor."""
    
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.5,
                 predictor_method: Literal['dot', 'cosine', 'mlp'] = 'mlp'):
        super(GATLinkPredictor, self).__init__()
        
        from .gat import GAT
        
        self.encoder = GAT(
            input_dim, hidden_dim, embedding_dim, num_layers, num_heads, dropout
        )
        self.predictor = LinkPredictor(embedding_dim, hidden_dim, predictor_method)
    
    def forward(self, x, edge_index, pred_edge_index):
        """Forward pass."""
        embeddings = self.encoder(x, edge_index)
        scores = self.predictor(embeddings, pred_edge_index)
        return scores
    
    def encode(self, x, edge_index):
        """Encode nodes to embeddings."""
        return self.encoder(x, edge_index)

