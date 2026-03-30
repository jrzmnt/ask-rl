"""
Decision controllers based on uncertainty signals.

This module decides *when* to intervene, not *how*.
It is agnostic to the policy (RL or SLM) and environment.
"""

from __future__ import annotations


class EntropyGate:
    """
    Simple uncertainty gate based on policy entropy.

    If the entropy exceeds a fixed threshold, an intervention
    (e.g., querying an SLM) should be triggered.
    """

    def __init__(self, threshold: float):
        """
        Args:
            threshold: entropy value above which intervention is triggered
        """
        self.threshold = float(threshold)

    def should_query(self, entropy: float) -> bool:
        """
        Decide whether to intervene based on entropy.

        Args:
            entropy: policy entropy at current step

        Returns:
            True if entropy exceeds threshold, else False
        """
        return entropy > self.threshold


# Optional quick sanity check
# if __name__ == "__main__":
#     gate = EntropyGate(threshold=0.5)

#     test_values = [0.05, 0.2, 0.49, 0.5, 0.7, 1.2]

#     for e in test_values:
#         print(f"Entropy={e:.3f} -> query={gate.should_query(e)}")
