"""
Entropy-based uncertainty estimation for RL policies and SLMs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy


# -------------------------------------------------
# RL policy entropy (PPO)
# -------------------------------------------------

def policy_entropy(model, obs: np.ndarray) -> float:
    """
    Compute the action entropy of a PPO policy for a single observation.

    Args:
        model: trained SB3 PPO model
        obs: observation array

    Returns:
        Scalar entropy value
    """
    if not isinstance(model.policy, ActorCriticPolicy):
        raise TypeError("Entropy computation expects an ActorCriticPolicy.")

    obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
    obs_tensor = obs_tensor.to(model.device)

    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        entropy = distribution.entropy()

    return float(entropy.mean().cpu().item())


# -------------------------------------------------
# SLM token entropy
# -------------------------------------------------

def entropy_from_logits(logits: torch.Tensor) -> float:
    """
    Compute categorical entropy from SLM logits.

    Args:
        logits: Tensor of shape (V,) or (1, V)

    Returns:
        Scalar entropy value
    """
    if logits is None:
        return float("nan")

    if logits.dim() == 2:
        logits = logits.squeeze(0)

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    entropy = -(probs * log_probs).sum()

    return float(entropy.item())


# -------------------------------------------------
# Sanity check
# -------------------------------------------------

if __name__ == "__main__":
    print("entropy.py loaded correctly")
