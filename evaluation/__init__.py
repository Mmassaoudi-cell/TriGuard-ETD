# evaluation/__init__.py
from .evaluate      import evaluate, predict, compute_metrics
from .results_table import generate_results

__all__ = ["evaluate", "predict", "compute_metrics", "generate_results"]
