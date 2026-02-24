import numpy as np
import weakref
import contextlib
import heapq


class Config:
    enable_backprop = True


@contextlib.contextmanager
def no_grad():
    Config.enable_backprop = False
    try:
        yield
    finally:
        Config.enable_backprop = True


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
        self.data = data
        self.name = name
        self._grad = None
        self.creator = None
        self.generation = 0

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
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        if value is not None:
            if not isinstance(value, np.ndarray):
                value = np.array(value)
        self._grad = value

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        heap = []
        seen_set = set()
        counter = 0

        def add_func(f):
            nonlocal counter
            if f not in seen_set:
                counter += 1
                heapq.heappush(heap, (-f.generation, counter, f))
                seen_set.add(f)

        add_func(self.creator)

        while heap:
            _, _, f = heapq.heappop(heap)
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

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


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)
