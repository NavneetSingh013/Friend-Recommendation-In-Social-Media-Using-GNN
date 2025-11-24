"""
Node2Vec baseline for link prediction.
"""

import torch
import numpy as np
from node2vec import Node2Vec
import networkx as nx
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


class Node2VecBaseline:
    """Node2Vec baseline model."""
    
    def __init__(self, dimensions: int = 64, walk_length: int = 30,
                 num_walks: int = 200, p: float = 1.0, q: float = 1.0):
        """
        Initialize Node2Vec.
        
        Args:
            dimensions: Embedding dimensions
            walk_length: Length of random walks
            num_walks: Number of walks per node
            p: Return parameter
            q: In-out parameter
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.model = None
        self.embeddings = None
        self.classifier = None
    
    def train(self, edge_index: torch.Tensor, num_nodes: int):
        """
        Train Node2Vec model.
        
        Args:
            edge_index: Edge index [2, num_edges]
            num_nodes: Number of nodes
        """
        # Convert to NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            G.add_edge(src, dst)
        
        # Train Node2Vec
        print("Training Node2Vec...")
        node2vec = Node2Vec(G, dimensions=self.dimensions, walk_length=self.walk_length,
                           num_walks=self.num_walks, p=self.p, q=self.q, workers=1)
        self.model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        # Get embeddings
        self.embeddings = np.zeros((num_nodes, self.dimensions))
        for node in range(num_nodes):
            if str(node) in self.model.wv:
                self.embeddings[node] = self.model.wv[str(node)]
    
    def predict(self, edges: torch.Tensor) -> np.ndarray:
        """
        Predict link existence.
        
        Args:
            edges: Edge index [2, num_edges]
            
        Returns:
            Prediction scores
        """
        if self.embeddings is None:
            raise ValueError("Model not trained. Call train() first.")
        
        scores = []
        for i in range(edges.size(1)):
            src, dst = edges[0, i].item(), edges[1, i].item()
            # Cosine similarity
            src_emb = self.embeddings[src]
            dst_emb = self.embeddings[dst]
            score = np.dot(src_emb, dst_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(dst_emb) + 1e-8)
            scores.append(score)
        
        return np.array(scores)
    
    def train_classifier(self, train_edges: torch.Tensor, train_labels: torch.Tensor):
        """
        Train a classifier on top of embeddings.
        
        Args:
            train_edges: Training edges [2, num_edges]
            train_labels: Training labels
        """
        if self.embeddings is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create features
        features = []
        for i in range(train_edges.size(1)):
            src, dst = train_edges[0, i].item(), train_edges[1, i].item()
            # Concatenate embeddings
            feature = np.concatenate([self.embeddings[src], self.embeddings[dst]])
            features.append(feature)
        
        features = np.array(features)
        labels = train_labels.numpy()
        
        # Train classifier
        self.classifier = LogisticRegression(max_iter=1000)
        self.classifier.fit(features, labels)
    
    def predict_with_classifier(self, edges: torch.Tensor) -> np.ndarray:
        """
        Predict using trained classifier.
        
        Args:
            edges: Edge index [2, num_edges]
            
        Returns:
            Prediction scores
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train_classifier() first.")
        
        # Create features
        features = []
        for i in range(edges.size(1)):
            src, dst = edges[0, i].item(), edges[1, i].item()
            feature = np.concatenate([self.embeddings[src], self.embeddings[dst]])
            features.append(feature)
        
        features = np.array(features)
        
        # Predict
        scores = self.classifier.predict_proba(features)[:, 1]
        return scores

