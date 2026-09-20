"""Base ``Module`` class for all neural-network components."""

from typing import Any, Dict, Iterator, Tuple

import numpy as np

from dptiny import backend
from dptiny.backend import xp
from dptiny.core import Config, Parameter


class Module:
    """Base class for all neural network modules.

    Any ``Parameter`` or ``Module`` assigned as an attribute is registered
    automatically in ``self._params`` so that ``parameters()`` can find it.
    Plain arrays can be tracked with ``register_buffer`` (e.g. BatchNorm's
    running statistics) and are included in ``state_dict``.
    """

    def __init__(self):
        # Dicts used as insertion-ordered sets so parameter iteration order
        # is deterministic across runs.
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_buffers", {})

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, (Parameter, Module)):
            self._params.setdefault(name, None)
        object.__setattr__(self, name, value)

    def register_buffer(self, name: str, arr: Any) -> None:
        """Register a persistent non-parameter array (e.g. running stats)."""
        self._buffers.setdefault(name, None)
        object.__setattr__(self, name, arr)

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def forward(self, *inputs):
        """Compute the module's forward pass."""
        raise NotImplementedError()

    # -- parameters --------------------------------------------------------

    def _members(self, kind: type, prefix: str = ""):
        for name in self._params:
            try:
                value = object.__getattribute__(self, name)
            except AttributeError:
                continue
            if isinstance(value, kind):
                yield f"{prefix}{name}", value
            elif isinstance(value, Module):
                yield from value._members(kind, f"{prefix}{name}.")

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        """Yield ``(name, parameter)`` pairs with dotted names."""
        yield from self._members(Parameter, prefix)

    def parameters(self) -> Iterator[Parameter]:
        """Yield all registered parameters recursively."""
        for _, p in self.named_parameters():
            yield p

    def named_buffers(self, prefix: str = "") -> Iterator[Tuple[str, Any]]:
        """Yield ``(name, buffer)`` pairs with dotted names."""
        for name in self._buffers:
            yield f"{prefix}{name}", object.__getattribute__(self, name)
        for name in self._params:
            try:
                value = object.__getattribute__(self, name)
            except AttributeError:
                continue
            if isinstance(value, Module):
                yield from value.named_buffers(f"{prefix}{name}.")

    @property
    def params(self) -> Dict[str, Parameter]:
        """Dict view of ``named_parameters`` kept for backward compatibility."""
        return dict(self.named_parameters())

    def cleargrads(self) -> None:
        """Reset the gradient of every parameter to ``None``."""
        for p in self.parameters():
            p.cleargrad()

    # ``zero_grad`` alias kept alongside the classic name.
    zero_grad = cleargrads

    # -- device transfer ----------------------------------------------------

    def to_cpu(self) -> "Module":
        """Move all parameters and buffers to host (NumPy) memory."""
        for p in self.parameters():
            p.to_cpu()
        for name, buf in self.named_buffers():
            owner_name = name.rsplit(".", 1)
            owner = self._find_owner(owner_name[0]) if len(owner_name) == 2 else self
            object.__setattr__(owner, owner_name[-1], backend.to_cpu(buf))
        return self

    def to_gpu(self) -> "Module":
        """Move all parameters and buffers to GPU (CuPy) memory."""
        for p in self.parameters():
            p.to_gpu()
        for name, buf in self.named_buffers():
            owner_name = name.rsplit(".", 1)
            owner = self._find_owner(owner_name[0]) if len(owner_name) == 2 else self
            object.__setattr__(owner, owner_name[-1], backend.to_gpu(buf))
        return self

    def _find_owner(self, dotted: str) -> "Module":
        module = self
        for part in dotted.split("."):
            module = getattr(module, part)
        return module

    # -- train / eval --------------------------------------------------------

    def train(self) -> "Module":
        """Enable training mode globally (sets ``Config.train = True``)."""
        Config.train = True
        return self

    def eval(self) -> "Module":
        """Enable evaluation mode globally (sets ``Config.train = False``)."""
        Config.train = False
        return self

    # -- serialization --------------------------------------------------------

    def state_dict(self) -> Dict[str, np.ndarray]:
        """Return ``{name: numpy array}`` for all parameters and buffers."""
        state = {}
        for name, p in self.named_parameters():
            state[name] = np.asarray(backend.to_cpu(p.data))
        for name, buf in self.named_buffers():
            state[name] = np.asarray(backend.to_cpu(buf))
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load a ``state_dict`` mapping into parameters and buffers."""
        targets = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        for name, value in state.items():
            arr = xp.asarray(value)
            if name in targets:
                targets[name].data = arr.astype(targets[name].data.dtype)
            elif name in buffers:
                owner_name = name.rsplit(".", 1)
                owner = (
                    self._find_owner(owner_name[0])
                    if len(owner_name) == 2
                    else self
                )
                object.__setattr__(owner, owner_name[-1], arr)
            else:
                raise KeyError(f"unexpected key in state_dict: {name}")

    def save_weights(self, path: str) -> None:
        """Serialize ``state_dict`` to a compressed ``.npz`` file."""
        np.savez_compressed(path, **self.state_dict())

    def load_weights(self, path: str) -> None:
        """Load weights saved with :meth:`save_weights`."""
        with np.load(path) as data:
            self.load_state_dict({k: data[k] for k in data.files})
