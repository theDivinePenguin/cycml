"""Reproducibility utilities for deterministic experiments."""
import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set random seeds across all libraries for deterministic execution.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Worker initialization function for PyTorch DataLoader workers.

    Args:
        worker_id: The unique worker identifier.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
