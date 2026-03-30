from __future__ import annotations

import time


class Timer:
    """
    Simple wall-clock timer.

    Usage:
        with Timer() as t:
            ...
        elapsed = t.elapsed
    """

    def __enter__(self):
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self._start
        return False  # do not suppress exceptions
