"""
Uncertainty metrics for RL policies.

This module is intentionally model-agnostic and read-only:
- It does NOT affect decisions
- It only measures uncertainty

Primary metric: policy entropy
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List


# ---------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------

def policy_entropy_from_logits(logits: torch.Tensor) -> float:
    """
    Compute entropy of a categorical policy from logits.

    Args:
        logits: Tensor of shape (n_actions,)

    Returns:
        entropy (float)
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum()
    return entropy.item()


# ---------------------------------------------------------------------
# Batch-level metrics
# ---------------------------------------------------------------------

def batch_policy_entropy(logits_batch: torch.Tensor) -> np.ndarray:
    """
    Compute entropy for a batch of logits.

    Args:
        logits_batch: Tensor of shape (batch_size, n_actions)

    Returns:
        np.ndarray of shape (batch_size,)
    """
    probs = torch.softmax(logits_batch, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.detach().cpu().numpy()


# ---------------------------------------------------------------------
# Aggregation utilities
# ---------------------------------------------------------------------

def entropy_statistics(entropies: List[float]) -> Dict[str, float]:
    """
    Aggregate entropy statistics over a rollout or experiment.

    Args:
        entropies: list of entropy values

    Returns:
        dict with mean, std, min, max
    """
    entropies = np.asarray(entropies)

    return {
        "entropy_mean": float(entropies.mean()),
        "entropy_std": float(entropies.std()),
        "entropy_min": float(entropies.min()),
        "entropy_max": float(entropies.max()),
    }


# if __name__ == "__main__":
#     logits = torch.tensor([2.0, 0.5, -1.0, 0.1])
#     entropy = policy_entropy_from_logits(logits)
#     print("Entropy:", entropy)
