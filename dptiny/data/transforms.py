"""Sample transforms usable with ``Dataset``."""

from typing import Callable, Iterable

from dptiny.backend import xp


class Compose:
    """Apply a sequence of callables in order."""

    def __init__(self, transforms: Iterable[Callable]):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class Normalize:
    """Normalize an array to zero mean / unit variance per given stats."""

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std


class Flatten:
    """Flatten each sample to a 1-D vector."""

    def __call__(self, x):
        return x.reshape(-1)


class ToFloat:
    """Cast samples to ``float32``."""

    def __call__(self, x):
        return x.astype(xp.float32)
