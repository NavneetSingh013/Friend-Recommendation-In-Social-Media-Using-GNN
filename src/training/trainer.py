"""
Training utilities for GNN models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Callable
import os
from .utils import EarlyStopping


class Trainer:
    """Trainer for GNN link prediction models."""
    
    def __init__(self, model: nn.Module, device: torch.device,
                 optimizer: Optional[optim.Optimizer] = None,
                 criterion: Optional[nn.Module] = None,
                 lr: float = 0.01, weight_decay: float = 5e-4):
        """
        Initialize trainer.
        
        Args:
            model: GNN model
            device: Device (cpu/cuda)
            optimizer: Optimizer (if None, uses Adam)
            criterion: Loss function (if None, uses BCEWithLogitsLoss)
            lr: Learning rate
            weight_decay: Weight decay
        """
        self.model = model.to(device)
        self.device = device
        
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer
        
        if criterion is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = criterion
    
    def train_epoch(self, data, train_edges: torch.Tensor, train_labels: torch.Tensor, batch_size: int = None):
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        if hasattr(self.model, 'encoder'):  # GraphSAGE/GAT with encoder
            scores = self.model(data.x, data.edge_index, train_edges)
        else:  # SEAL - may need batching
            if batch_size is None:
                batch_size = 32  # Default batch size for SEAL
            try:
                scores = self.model(data.edge_index, data.x, train_edges, batch_size=batch_size)
            except TypeError:
                # Fallback if batch_size not supported
                scores = self.model(data.edge_index, data.x, train_edges)
        
        # Compute loss
        loss = self.criterion(scores, train_labels.to(self.device))
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data, edges: torch.Tensor, labels: torch.Tensor, batch_size: int = None):
        """Evaluate model."""
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'encoder'):
                scores = self.model(data.x, data.edge_index, edges)
            else:
                # SEAL - may need batching
                if batch_size is None:
                    batch_size = 32
                try:
                    scores = self.model(data.edge_index, data.x, edges, batch_size=batch_size)
                except TypeError:
                    scores = self.model(data.edge_index, data.x, edges)
            
            loss = self.criterion(scores, labels.to(self.device))
            predictions = torch.sigmoid(scores).cpu().numpy()
            labels_np = labels.cpu().numpy()
        
        return loss.item(), predictions, labels_np
    
    def train(self, data, train_data: Dict, val_data: Dict, num_epochs: int = 100,
             early_stopping: Optional[EarlyStopping] = None,
             checkpoint_dir: Optional[str] = None, verbose: bool = True):
        """
        Train model.
        
        Args:
            data: Graph data
            train_data: Training data dict with 'edges' and 'labels'
            val_data: Validation data dict with 'edges' and 'labels'
            num_epochs: Number of epochs
            early_stopping: Early stopping callback
            checkpoint_dir: Directory to save checkpoints
            verbose: Whether to print progress
        """
        # Move data to device
        data = data.to(self.device)
        train_edges = train_data['edges'].to(self.device)
        train_labels = train_data['labels'].to(self.device)
        val_edges = val_data['edges'].to(self.device)
        val_labels = val_data['labels'].to(self.device)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(data, train_edges, train_labels)
            train_losses.append(train_loss)
            
            # Validate
            val_loss, val_pred, val_labels_np = self.evaluate(data, val_edges, val_labels)
            val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if early_stopping:
                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    if early_stopping.best_model_state is not None:
                        self.model.load_state_dict(early_stopping.best_model_state)
                    break
            
            # Save checkpoint
            if checkpoint_dir and val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint

