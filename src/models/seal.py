"""
SEAL (Subgraph Embedding And Link prediction) implementation.
Extracts subgraphs around target links and uses GNN for prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from torch_geometric.utils import k_hop_subgraph, to_networkx
from typing import Tuple, Optional
import numpy as np
from collections import defaultdict


def double_radius_node_labeling(edge_index: torch.Tensor, src: int, dst: int,
                                num_hops: int = 2) -> torch.Tensor:
    """
    Double-radius node labeling for SEAL.
    
    Args:
        edge_index: Edge index
        src: Source node
        dst: Destination node
        num_hops: Number of hops
        
    Returns:
        Node labels
    """
    # Extract subgraph
    subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
        [src, dst], num_hops, edge_index, relabel_nodes=True
    )
    
    src_mapped = mapping[0].item()
    dst_mapped = mapping[1].item()
    
    # Compute distances
    num_nodes = subset.size(0)
    labels = torch.zeros(num_nodes, dtype=torch.long)
    
    # Create adjacency matrix for distance computation
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    for i in range(edge_index_sub.size(1)):
        u, v = edge_index_sub[0, i].item(), edge_index_sub[1, i].item()
        adj[u, v] = True
        adj[v, u] = True
    
    # BFS to compute distances
    def bfs_distances(start, max_dist):
        distances = torch.full((num_nodes,), -1, dtype=torch.long)
        queue = [start]
        distances[start] = 0
        
        while queue:
            node = queue.pop(0)
            if distances[node] >= max_dist:
                continue
            
            for neighbor in range(num_nodes):
                if adj[node, neighbor] and distances[neighbor] == -1:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        
        return distances
    
    dist_src = bfs_distances(src_mapped, num_hops)
    dist_dst = bfs_distances(dst_mapped, num_hops)
    
    # Compute labels: combination of distances
    for i in range(num_nodes):
        d1 = dist_src[i].item()
        d2 = dist_dst[i].item()
        if d1 == -1:
            d1 = num_hops + 1
        if d2 == -1:
            d2 = num_hops + 1
        labels[i] = 1 + min(d1, d2) + (num_hops + 1) * min(max(d1, d2), num_hops + 1)
    
    return labels


class SEALGCN(nn.Module):
    """GCN for SEAL subgraph classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3,
                 dropout: float = 0.5, pool: str = 'mean'):
        """
        Initialize SEAL GCN.
        
        Args:
            input_dim: Input feature dimension (including node labels)
            hidden_dim: Hidden dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
            pool: Pooling method ('mean', 'max', 'sum')
        """
        super(SEALGCN, self).__init__()
        
        self.num_layers = num_layers
        self.pool = pool
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge index [2, num_edges]
            batch: Batch vector [num_nodes]
            
        Returns:
            Link scores [batch_size]
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        
        # Pooling
        if self.pool == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pool == 'max':
            x = global_max_pool(x, batch)
        else:  # sum
            from torch_geometric.nn import global_add_pool
            x = global_add_pool(x, batch)
        
        # Classification
        scores = self.classifier(x).squeeze()
        return scores


class SEAL(nn.Module):
    """SEAL model for link prediction."""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 64, num_layers: int = 3,
                 num_hops: int = 2, dropout: float = 0.5, pool: str = 'mean'):
        """
        Initialize SEAL.
        
        Args:
            feature_dim: Original node feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GCN layers
            num_hops: Number of hops for subgraph extraction
            dropout: Dropout rate
            pool: Pooling method
        """
        super(SEAL, self).__init__()
        
        self.num_hops = num_hops
        self.feature_dim = feature_dim
        
        # Maximum label value (based on num_hops)
        max_label = 1 + 2 * num_hops + (num_hops + 1) * (num_hops + 1)
        
        # Label embedding
        self.label_encoder = nn.Embedding(max_label + 1, hidden_dim // 4)
        
        # Input dimension: original features + label embedding
        input_dim = feature_dim + hidden_dim // 4
        
        self.gcn = SEALGCN(input_dim, hidden_dim, num_layers, dropout, pool)
    
    def extract_subgraph(self, edge_index: torch.Tensor, src: int, dst: int,
                        node_features: torch.Tensor) -> Data:
        """
        Extract subgraph around target link.
        
        Args:
            edge_index: Full graph edge index
            src: Source node
            dst: Destination node
            node_features: Node features
            
        Returns:
            Subgraph data
        """
        # Extract k-hop subgraph
        subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
            [src, dst], self.num_hops, edge_index, relabel_nodes=True
        )
        
        # Get node features for subgraph
        x_sub = node_features[subset]
        
        # Compute node labels
        labels = double_radius_node_labeling(edge_index, src, dst, self.num_hops)
        
        # Encode labels
        label_emb = self.label_encoder(labels)
        
        # Combine features
        x_combined = torch.cat([x_sub, label_emb], dim=1)
        
        # Create data object
        data = Data(x=x_combined, edge_index=edge_index_sub, num_nodes=subset.size(0))
        data.src_mapped = mapping[0].item()
        data.dst_mapped = mapping[1].item()
        
        return data
    
    def forward(self, edge_index: torch.Tensor, node_features: torch.Tensor,
                pred_edges: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            edge_index: Full graph edge index
            node_features: Node features
            pred_edges: Edges to predict [2, num_edges]
            batch_size: Batch size for subgraph processing
            
        Returns:
            Link scores [num_edges]
        """
        if batch_size is None:
            batch_size = pred_edges.size(1)
        
        all_scores = []
        
        # Process in batches
        for i in range(0, pred_edges.size(1), batch_size):
            batch_edges = pred_edges[:, i:i+batch_size]
            batch_subgraphs = []
            
            for j in range(batch_edges.size(1)):
                src = batch_edges[0, j].item()
                dst = batch_edges[1, j].item()
                subgraph = self.extract_subgraph(edge_index, src, dst, node_features)
                batch_subgraphs.append(subgraph)
            
            # Batch subgraphs
            batch = Batch.from_data_list(batch_subgraphs)
            
            # Forward through GCN
            scores = self.gcn(batch.x, batch.edge_index, batch.batch)
            all_scores.append(scores)
        
        return torch.cat(all_scores, dim=0)
    
    def predict_links(self, edge_index: torch.Tensor, node_features: torch.Tensor,
                     candidate_edges: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        """Predict link scores for candidate edges."""
        self.eval()
        with torch.no_grad():
            scores = self.forward(edge_index, node_features, candidate_edges, batch_size)
        return scores

