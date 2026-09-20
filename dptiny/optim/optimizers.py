"""Concrete optimizer implementations."""

from typing import Any, Dict

from dptiny.backend import xp
from dptiny.core import Variable
from dptiny.optim.optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum."""

    def __init__(self, params: Any = None, lr: float = 0.01,
                 momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.vs: Dict[int, Any] = {}

    def update_one(self, param: Variable):
        key = id(param)
        if key not in self.vs:
            self.vs[key] = xp.zeros_like(param.data)

        grad = self._prepare_grad(param)
        v = self.vs[key]

        xp.multiply(v, self.momentum, out=v)
        xp.add(v, -self.lr * grad, out=v)
        xp.add(param.data, v, out=param.data)


# ``MomentumSGD`` is ``SGD`` with a momentum argument, kept as an alias.
MomentumSGD = SGD


class Adam(Optimizer):
    """Adam optimizer with bias-corrected learning rate."""

    def __init__(
        self,
        params: Any = None,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms: Dict[int, Any] = {}
        self.vs: Dict[int, Any] = {}

    def update_one(self, param: Variable):
        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        lr_t = (
            self.lr
            * xp.sqrt(1 - self.beta2**self.t)
            / (1 - self.beta1**self.t)
        )

        grad = self._prepare_grad(param)
        m = self.ms[key]
        v = self.vs[key]

        xp.multiply(m, self.beta1, out=m)
        xp.add(m, (1 - self.beta1) * grad, out=m)

        xp.multiply(v, self.beta2, out=v)
        xp.add(v, (1 - self.beta2) * grad**2, out=v)

        xp.add(
            param.data,
            -lr_t * m / (xp.sqrt(v) + self.eps),
            out=param.data,
        )


class AdamW(Adam):
    """Adam with decoupled weight decay."""

    def __init__(self, params: Any = None, lr: float = 0.001,
                 weight_decay: float = 0.01, **kwargs):
        super().__init__(params, lr, **kwargs)
        self.weight_decay = weight_decay

    def update_one(self, param: Variable):
        xp.multiply(
            param.data,
            1 - self.lr * self.weight_decay,
            out=param.data,
        )
        super().update_one(param)


class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(
        self,
        params: Any = None,
        lr: float = 0.001,
        alpha: float = 0.99,
        eps: float = 1e-8,
    ):
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps
        self.vs: Dict[int, Any] = {}

    def update_one(self, param: Variable):
        key = id(param)
        if key not in self.vs:
            self.vs[key] = xp.zeros_like(param.data)

        grad = self._prepare_grad(param)
        v = self.vs[key]

        xp.multiply(v, self.alpha, out=v)
        xp.add(v, (1 - self.alpha) * grad**2, out=v)
        xp.add(
            param.data,
            -self.lr * grad / (xp.sqrt(v) + self.eps),
            out=param.data,
        )
