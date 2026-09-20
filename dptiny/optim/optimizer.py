"""Base ``Optimizer`` class for DPTiny."""

from typing import Any, Callable, Dict, Iterable, List, Optional

from dptiny.core import Variable


class Optimizer:
    """Base class for parameter-update algorithms.

    ``setup``/``update`` accept a ``Module``, a ``{name: Parameter}`` dict, or
    an iterable of ``Parameter``s.  Optimizer state is keyed by ``id(param)``
    and updates are performed in place so they work for NumPy and CuPy alike.
    """

    def __init__(self, params: Any = None, lr: Optional[float] = None):
        self.lr = lr
        self.t = 0
        self.hooks: List[Callable[[List[Variable]], None]] = []
        self.target: Any = None
        if params is not None:
            self.setup(params)

    def setup(self, target: Any) -> "Optimizer":
        """Attach the parameters this optimizer should update."""
        self.target = target
        return self

    def set_lr(self, lr: float) -> None:
        self.lr = lr

    def add_hook(self, hook: Callable[[List[Variable]], None]) -> None:
        """Add a callable run on the parameter list before every update."""
        self.hooks.append(hook)

    @staticmethod
    def _resolve(target: Any) -> List[Variable]:
        if target is None:
            return []
        if hasattr(target, "parameters"):  # Module
            return [p for p in target.parameters() if p is not None]
        if isinstance(target, dict):
            return [p for p in target.values() if p is not None]
        if isinstance(target, Iterable):
            return [p for p in target if p is not None]
        raise TypeError(f"cannot resolve parameters from {type(target)}")

    def update(self, params: Any = None) -> None:
        """Perform one update step.

        ``params`` may be a ``Module``, dict, or iterable; when omitted the
        target given to ``__init__``/``setup`` is used.
        """
        plist = self._resolve(params if params is not None else self.target)
        for hook in self.hooks:
            hook(plist)
        self.t += 1
        for param in plist:
            if getattr(param, "grad", None) is None:
                continue
            self.update_one(param)

    # ``step`` is an alias for ``update`` (PyTorch-style naming).
    step = update

    def zero_grad(self) -> None:
        """Reset gradients of all attached parameters."""
        for param in self._resolve(self.target):
            param.cleargrad()

    def update_one(self, param: Variable) -> None:
        """Update a single parameter in place."""
        raise NotImplementedError()

    @staticmethod
    def _prepare_grad(param: Variable):
        """Return a gradient whose shape matches ``param.data``."""
        grad = param.grad
        if grad.shape != param.data.shape:
            grad = grad.reshape(param.data.shape)
        return grad
