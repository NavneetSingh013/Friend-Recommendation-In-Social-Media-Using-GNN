"""
GraphSAGE model implementation.
Supports mean and pooling aggregators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from typing import Literal


class GraphSAGE(nn.Module):
    """GraphSAGE model for node embedding."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.5,
                 aggregator: Literal['mean', 'max', 'lstm'] = 'mean'):
        """
        Initialize GraphSAGE.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            num_layers: Number of layers
            dropout: Dropout rate
            aggregator: Aggregation method ('mean', 'max', 'lstm')
        """
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggregator))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim, aggr=aggregator))
        else:
            self.convs.append(SAGEConv(input_dim, output_dim, aggr=aggregator))
    
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

