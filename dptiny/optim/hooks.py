"""Optimizer hooks run on the parameter list before each update."""

from typing import List

from dptiny.backend import xp
from dptiny.core import Variable


class WeightDecay:
    """Add ``rate * param.data`` to every gradient (L2 regularization)."""

    def __init__(self, rate: float):
        self.rate = rate

    def __call__(self, params: List[Variable]):
        for p in params:
            if p.grad is None:
                continue
            xp.add(p.grad, self.rate * p.data, out=p.grad)


class ClipGrad:
    """Clip gradients so their global L2 norm does not exceed ``max_norm``."""

    def __init__(self, max_norm: float):
        self.max_norm = max_norm

    def __call__(self, params: List[Variable]):
        total = 0.0
        for p in params:
            if p.grad is None:
                continue
            total += float((p.grad**2).sum())
        norm = total**0.5
        if norm > self.max_norm and norm > 0:
            rate = self.max_norm / norm
            for p in params:
                if p.grad is None:
                    continue
                xp.multiply(p.grad, rate, out=p.grad)
