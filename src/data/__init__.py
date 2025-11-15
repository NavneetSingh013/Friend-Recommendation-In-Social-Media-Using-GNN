from .facebook_loader import FacebookDatasetLoader
from .preprocessing import GraphPreprocessor
from .heuristics import compute_heuristics

# Optional OGB import
try:
    from .ogb_loader import OGBDatasetLoader
    __all__ = [
        'FacebookDatasetLoader',
        'OGBDatasetLoader',
        'GraphPreprocessor',
        'compute_heuristics'
    ]
except ImportError:
    __all__ = [
        'FacebookDatasetLoader',
        'GraphPreprocessor',
        'compute_heuristics'
    ]

