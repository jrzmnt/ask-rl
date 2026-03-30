from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: Optional[int] = None) -> None:
    """
    Set random seed for reproducibility.

    This function seeds:
    - Python random
    - NumPy
    - Environment-level hash seed

    Torch is intentionally NOT seeded here to avoid
    unintended side effects on model-level randomness.
    """
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
