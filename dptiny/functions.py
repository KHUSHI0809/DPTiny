import numpy as np
from dptiny.core import Variable, as_array, as_variable, Config
import weakref
from abc import ABC, abstractmethod


class Function(ABC):
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.creator = self

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs[0] if len(outputs) == 1 else outputs

    @abstractmethod
    def forward(self, *xs):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, *gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        # Handle broadcasting properly
        self._x0_shape = x0.shape
        self._x1_shape = x1.shape
        return x0 + x1

    def backward(self, gy):
        # Handle broadcasting in backward pass
        gx0, gx1 = gy, gy

        # Sum gradients along broadcasted dimensions if needed
        if self._x0_shape != self._x1_shape:
            if np.ndim(gx0) > len(self._x0_shape):
                axes = tuple(range(np.ndim(gx0) - len(self._x0_shape)))
                gx0 = gx0.sum(axis=axes, keepdims=True)

            if np.ndim(gx1) > len(self._x1_shape):
                axes = tuple(range(np.ndim(gx1) - len(self._x1_shape)))
                gx1 = gx1.sum(axis=axes, keepdims=True)

            # Handle broadcasting dimensions
            for i, (dim0, dim1) in enumerate(
                zip(self._x0_shape[::-1], self._x1_shape[::-1])
            ):
                axis = -i - 1
                if dim0 == 1:
                    gx0 = gx0.sum(axis=axis, keepdims=True)
                if dim1 == 1:
                    gx1 = gx1.sum(axis=axis, keepdims=True)

        return gx0, gx1


class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1.data, gy * x0.data


class MatMul(Function):
    def forward(self, x, W):
        self.x_shape = x.shape
        self.W_shape = W.shape
        return x @ W

    def backward(self, gy):
        x, W = self.inputs
        gx = gy @ W.data.T
        gW = x.data.T @ gy

        # Handle reshaping if needed due to broadcasting
        if gx.shape != self.x_shape:
            gx = gx.reshape(self.x_shape)
        if gW.shape != self.W_shape:
            gW = gW.reshape(self.W_shape)

        return gx, gW


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        return gy, -gy


class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy / x1.data, gy * (-x0.data / x1.data**2)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * y.data


class Log(Function):
    def forward(self, x):
        return np.log(x)

    def backward(self, gy):
        x = self.inputs[0]
        return gy / x.data


class Pow(Function):
    def forward(self, x, exponent):
        self._exponent = exponent
        return x**exponent

    def backward(self, gy):
        x = self.inputs[0]
        return gy * self._exponent * (x.data ** (self._exponent - 1))


class Sigmoid(Function):
    def forward(self, x):
        self._positive_mask = x >= 0
        y = np.where(
            self._positive_mask, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x))
        )
        self._y = y
        return y

    def backward(self, gy):
        return gy * self._y * (1 - self._y)


class ReLU(Function):
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, gy):
        x = self.inputs[0]
        mask = x.data > 0
        return gy * mask


class Softmax(Function):
    def forward(self, x):
        y = x - x.max(axis=1, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=1, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y.data * gy
        sumdx = gx.sum(axis=1, keepdims=True)
        gx -= y.data * sumdx
        return gx


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        self.t = t
        self.x_shape = x.shape

        batch_size = x.shape[0]
        if self.t.size == self.t.shape[0]:  # if labels are not one-hot
            self.t_oh = np.eye(x.shape[1], dtype=x.dtype)[self.t]
        else:
            self.t_oh = self.t

        # Compute softmax with improved numerical stability
        x_max = x.max(axis=1, keepdims=True)
        x_shifted = x - x_max
        exp_x = np.exp(x_shifted)
        sum_exp_x = exp_x.sum(axis=1, keepdims=True)
        self.y = exp_x / sum_exp_x

        # Compute cross entropy with improved numerical stability
        log_y = x_shifted - np.log(sum_exp_x)  # More stable than np.log(self.y)
        batch_log_y = np.sum(self.t_oh * log_y, axis=1)
        loss = -np.sum(batch_log_y) / batch_size
        return loss

    def backward(self, gy):
        batch_size = self.x_shape[0]
        dx = (self.y - self.t_oh) * gy / batch_size
        return dx


def add(x0, x1):
    return Add()(x0, x1)


def mul(x0, x1):
    return Mul()(x0, x1)


def neg(x):
    return Neg()(x)


def sub(x0, x1):
    return Sub()(x0, x1)


def div(x0, x1):
    return Div()(x0, x1)


def rsub(x0, x1):
    return sub(x1, x0)


def rdiv(x0, x1):
    return div(x1, x0)


def exp(x):
    return Exp()(x)


def log(x):
    return Log()(x)


def sigmoid(x):
    return Sigmoid()(x)


def relu(x):
    return ReLU()(x)


def softmax(x):
    return Softmax()(x)


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def matmul(x, W):
    return MatMul()(x, W)


def pow(x, exponent):
    return Pow()(x, exponent)
