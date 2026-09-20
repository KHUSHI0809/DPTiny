"""Core autograd data structures for DPTiny."""

import contextlib
import heapq
import weakref
from abc import ABC, abstractmethod
from typing import Any, Optional

from dptiny.backend import xp


class Config:
    """Global runtime configuration."""

    enable_backprop: bool = True
    train: bool = True


@contextlib.contextmanager
def using_config(name: str, value: Any):
    """Temporarily set a ``Config`` attribute inside a ``with`` block."""
    original = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, original)


def no_grad():
    """Context manager that disables gradient bookkeeping."""
    return using_config("enable_backprop", False)


def test_mode():
    """Context manager that switches layers like Dropout/BatchNorm to eval."""
    return using_config("train", False)


class Function(ABC):
    """Base class for differentiable operations.

    Subclasses implement ``forward`` and ``backward``.  The base
    ``__call__`` builds the computational graph automatically when
    ``Config.enable_backprop`` is ``True``.
    """

    def __call__(self, *inputs: Any):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max(x.generation for x in inputs)
            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs[0] if len(outputs) == 1 else outputs

    @abstractmethod
    def forward(self, *xs: Any) -> Any:
        """Compute the forward pass and return raw arrays."""
        raise NotImplementedError()

    @abstractmethod
    def backward(self, *gys: Any) -> Any:
        """Compute gradients with respect to inputs."""
        raise NotImplementedError()


class Variable:
    """A node in the computational graph.

    ``Variable`` wraps an array and stores its gradient plus a reference to the
    ``Function`` that produced it.  Operator overloads delegate to the
    corresponding autograd ``Function`` so that operations build a graph
    automatically.
    """

    __array_priority__ = 200

    def __init__(self, data: Any, name: Optional[str] = None):
        if data is not None and not isinstance(data, xp.ndarray):
            raise TypeError(
                f"Variable expects an array, got {type(data).__name__}"
            )
        self.data = data
        self.name = name
        self._grad: Optional[Any] = None
        self.creator: Optional[Function] = None
        self.generation: int = 0

    def set_creator(self, func: Function) -> None:
        """Attach ``func`` as this variable's creator function."""
        self.creator = func
        self.generation = func.generation + 1

    def unchain(self) -> None:
        """Drop the reference to this variable's creator function."""
        self.creator = None

    def unchain_backward(self) -> None:
        """Walk the graph backwards, clearing every creator link."""
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)

        if self.creator is not None:
            add_func(self.creator)

        while funcs:
            f = funcs.pop()
            for x in f.inputs:
                if x.creator is not None:
                    add_func(x.creator)
                    x.unchain()

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return self.transpose()

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        if value is not None and not isinstance(value, xp.ndarray):
            value = xp.array(value)
        self._grad = value

    def cleargrad(self):
        """Reset the gradient to ``None``."""
        self.grad = None

    def reshape(self, *shape):
        from dptiny.functions import reshape

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape(self, shape)

    def transpose(self, *axes):
        from dptiny.functions import transpose

        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            axes = axes[0]
        return transpose(self, axes)

    def sum(self, axis=None, keepdims=False):
        from dptiny.functions import sum

        return sum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        from dptiny.functions import mean

        return mean(self, axis, keepdims)

    def max(self, axis=None, keepdims=False):
        from dptiny.functions import max

        return max(self, axis, keepdims)

    def to_cpu(self):
        """Move ``data`` and ``grad`` to host (NumPy) memory."""
        from dptiny import backend

        if self.data is not None:
            self.data = backend.to_cpu(self.data)
        if self.grad is not None:
            self.grad = backend.to_cpu(self.grad)
        return self

    def to_gpu(self):
        """Move ``data`` and ``grad`` to GPU (CuPy) memory."""
        from dptiny import backend

        if self.data is not None:
            self.data = backend.to_gpu(self.data)
        if self.grad is not None:
            self.grad = backend.to_gpu(self.grad)
        return self

    def backward(self, retain_grad: bool = False, create_graph: bool = False):
        """Run backpropagation from this variable to all leaves.

        Gradients are stored as plain arrays, not ``Variable``s, so higher-order
        differentiation is not built. ``create_graph`` is accepted for API
        compatibility: it only toggles ``Config.enable_backprop`` around the
        internal ``Function.backward`` calls, which is currently a no-op.
        """
        if self.grad is None:
            self.grad = xp.ones_like(self.data)

        heap = []
        seen_set = set()
        counter = 0

        def add_func(func):
            nonlocal counter
            if func not in seen_set:
                counter += 1
                # Use generation as primary key; counter breaks ties.
                heapq.heappush(heap, (-func.generation, counter, func))
                seen_set.add(func)

        if self.creator is not None:
            add_func(self.creator)

        while heap:
            _, _, func = heapq.heappop(heap)
            gys = [output().grad for output in func.outputs]
            with using_config("enable_backprop", create_graph):
                gxs = func.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(func.inputs, gxs):
                if gx is None:
                    continue
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in func.outputs:
                    y().grad = None

    def __repr__(self) -> str:
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        name = f" {self.name}" if self.name else ""
        return f"variable({p}){name}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, slices):
        from dptiny.functions import get_item

        return get_item(self, slices)

    def __matmul__(self, other):
        from dptiny.functions import matmul

        return matmul(self, other)

    def __add__(self, other):
        from dptiny.functions import add

        return add(self, other)

    def __radd__(self, other):
        from dptiny.functions import add

        return add(other, self)

    def __mul__(self, other):
        from dptiny.functions import mul

        return mul(self, other)

    def __rmul__(self, other):
        from dptiny.functions import mul

        return mul(other, self)

    def __neg__(self):
        from dptiny.functions import neg

        return neg(self)

    def __sub__(self, other):
        from dptiny.functions import sub

        return sub(self, other)

    def __rsub__(self, other):
        from dptiny.functions import sub

        return sub(other, self)

    def __truediv__(self, other):
        from dptiny.functions import div

        return div(self, other)

    def __rtruediv__(self, other):
        from dptiny.functions import div

        return div(other, self)

    def __pow__(self, exponent):
        from dptiny.functions import pow as pow_func

        return pow_func(self, exponent)


class Parameter(Variable):
    """A ``Variable`` that is automatically registered as a model parameter."""


def as_array(x: Any) -> Any:
    """Convert a Python scalar to a 0-D array; leave arrays untouched."""
    if xp.isscalar(x):
        return xp.array(x)
    return x


def as_variable(obj: Any) -> "Variable":
    """Wrap raw data in a ``Variable`` if it is not already one."""
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)
