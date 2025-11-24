"""
Training utilities.
"""

import torch
import numpy as np
import random
import os


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
            self.best_score = float('inf')
        else:
            self.is_better = lambda current, best: current > best + min_delta
            self.best_score = float('-inf')
    
    def __call__(self, score: float, model: torch.nn.Module):
        """Check if training should stop."""
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

