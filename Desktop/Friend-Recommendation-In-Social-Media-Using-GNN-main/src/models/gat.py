"""
Graph Attention Network (GAT) implementation.
Multi-head attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional


class GAT(nn.Module):
    """Graph Attention Network for node embedding."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.5,
                 concat: bool = True):
        """
        Initialize GAT.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            num_layers: Number of layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            concat: Whether to concatenate or average attention heads
        """
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=concat)
        )
        
        # Hidden layers
        head_dim = hidden_dim * num_heads if concat else hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(head_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=concat)
            )
        
        # Output layer (single head)
        if num_layers > 1:
            head_dim = hidden_dim * num_heads if concat else hidden_dim
            self.convs.append(
                GATConv(head_dim, output_dim, heads=1, dropout=dropout, concat=False)
            )
        else:
            self.convs.append(
                GATConv(input_dim, output_dim, heads=1, dropout=dropout, concat=False)
            )
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge index [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def reset_parameters(self):
        """Reset model parameters."""
        for conv in self.convs:
            conv.reset_parameters()

