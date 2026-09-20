"""Learning-rate schedulers."""

import math

from dptiny.optim.optimizer import Optimizer


class _Scheduler:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.t = 0

    def get_lr(self) -> float:
        raise NotImplementedError()

    def step(self) -> None:
        self.t += 1
        self.optimizer.set_lr(self.get_lr())


class StepLR(_Scheduler):
    """Multiply the learning rate by ``gamma`` every ``step_size`` steps."""

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> float:
        return self.base_lr * self.gamma ** (self.t // self.step_size)


class ExponentialLR(_Scheduler):
    """Multiply the learning rate by ``gamma`` every step."""

    def __init__(self, optimizer: Optimizer, gamma: float):
        super().__init__(optimizer)
        self.gamma = gamma

    def get_lr(self) -> float:
        return self.base_lr * self.gamma**self.t


class CosineAnnealingLR(_Scheduler):
    """Cosine annealing from ``base_lr`` down to ``eta_min`` over ``T_max``."""

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self) -> float:
        return self.eta_min + (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self.t / self.T_max)
        ) / 2
