"""Container modules: ``Sequential`` and ``MLP``."""

from typing import List, Sequence

from dptiny import functions as F
from dptiny.core import no_grad
from dptiny.nn.layers import Linear, ReLU
from dptiny.nn.module import Module


class Sequential(Module):
    """A sequence of modules applied in order.

    Children are registered as attributes ``l0``, ``l1``, ... so their
    parameters appear in ``named_parameters`` as ``l0.W`` etc.
    """

    def __init__(self, *layers: Module):
        super().__init__()
        self._layers: List[Module] = []
        for i, layer in enumerate(layers):
            setattr(self, f"l{i}", layer)
            self._layers.append(layer)

    def __getitem__(self, index: int) -> Module:
        return self._layers[index]

    def __iter__(self):
        return iter(self._layers)

    def __len__(self):
        return len(self._layers)

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class MLP(Module):
    """Multi-layer perceptron built from ``Linear`` layers and an activation.

    ``activation`` may be a ``Module`` instance or a ``Module`` class; a new
    instance is created per hidden layer.  Linear layers are exposed as
    ``self.l1``, ``self.l2``, ... and named parameters look like ``l1.W``.
    """

    def __init__(
        self,
        in_size: int,
        hidden_sizes: Sequence[int],
        out_size: int,
        activation=ReLU,
    ):
        super().__init__()
        self.layers: List[Module] = []

        def _act() -> Module:
            return activation() if isinstance(activation, type) else activation

        sizes = [in_size] + list(hidden_sizes) + [out_size]
        for i in range(len(sizes) - 1):
            layer = Linear(sizes[i], sizes[i + 1])
            setattr(self, f"l{i + 1}", layer)
            self.layers.append(layer)
            if i != len(sizes) - 2:
                self.layers.append(_act())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x):
        """Return class predictions without tracking gradients."""
        with no_grad():
            y = self.forward(x)
            return y.data.argmax(axis=1)

    def accuracy(self, x, t):
        """Classification accuracy for inputs ``x`` and labels ``t``."""
        y = self.forward(x)
        return F.accuracy(y, t).data
