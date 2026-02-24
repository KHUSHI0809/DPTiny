import numpy as np
from dptiny.core import Variable, no_grad
from dptiny.functions import relu, matmul
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self):
        self.params = {}
        self.grads = {}

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        return outputs

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, grads):
        raise NotImplementedError()

    def cleargrads(self):
        """Clear gradients of all parameters."""
        for param in self.params.values():
            if param is not None:
                param.cleargrad()


class Linear(Layer):
    def __init__(self, in_size, out_size, nobias=False, init_type="he"):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        if init_type == "he":
            W_data = np.random.randn(in_size, out_size).astype(np.float32) * np.sqrt(
                2.0 / in_size
            )
        elif init_type == "xavier":
            W_data = np.random.randn(in_size, out_size).astype(np.float32) * np.sqrt(
                1.0 / in_size
            )
        elif init_type == "xavier_uniform":
            limit = np.sqrt(6.0 / (in_size + out_size))
            W_data = np.random.uniform(-limit, limit, (in_size, out_size)).astype(
                np.float32
            )
        else:
            W_data = np.random.randn(in_size, out_size).astype(np.float32) * 0.01

        self.W = Variable(W_data)

        if nobias:
            self.b = None
        else:
            self.b = Variable(np.zeros(out_size, dtype=np.float32))

        self.params = {"W": self.W, "b": self.b}

    def forward(self, x):
        y = matmul(x, self.W)
        if self.b is not None:
            y = y + self.b  # Broadcasting will work correctly with shape (out_size)
        return y

    def backward(self, grads):
        raise NotImplementedError()


class ReLU(Layer):
    def forward(self, x):
        return relu(x)

    def backward(self, grads):
        raise NotImplementedError()


class MLP(Layer):
    def __init__(self, in_size, hidden_sizes, out_size):
        super().__init__()
        self.layers = []
        self.params = {}

        sizes = [in_size] + hidden_sizes + [out_size]
        for i in range(len(sizes) - 1):
            layer = Linear(sizes[i], sizes[i + 1])
            setattr(self, f"l{i + 1}", layer)
            self.layers.append(layer)

            # Add layer parameters to the MLP parameters
            for key, param in layer.params.items():
                self.params[f"l{i + 1}.{key}"] = param

            if i != len(sizes) - 2:  # Not the last layer
                self.layers.append(ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grads):
        raise NotImplementedError()

    def predict(self, x):
        with no_grad():
            y = self.forward(x)
            return y.data.argmax(axis=1)

    def accuracy(self, x, t):
        y = self.predict(x)
        if t.ndim != 1:
            t = t.argmax(axis=1)
        accuracy = (y == t).mean()
        return float(accuracy)


class SGD:
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update(self, params):
        for name, param in params.items():
            if param is None or param.grad is None:
                continue

            if name not in self.vs:
                self.vs[name] = np.zeros_like(param.data)

            grad = param.grad
            if grad.shape != param.data.shape:
                grad = grad.reshape(param.data.shape)

            v = self.vs[name]
            np.multiply(v, self.momentum, out=v)
            np.add(v, -self.lr * grad, out=v)
            np.add(param.data, v, out=param.data)

    def set_lr(self, lr):
        self.lr = lr


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}
        self.t = 0

    def update(self, params):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)

        for name, param in params.items():
            if param is None or param.grad is None:
                continue

            if name not in self.ms:
                self.ms[name] = np.zeros_like(param.data)
                self.vs[name] = np.zeros_like(param.data)

            grad = param.grad
            if grad.shape != param.data.shape:
                grad = grad.reshape(param.data.shape)

            m = self.ms[name]
            v = self.vs[name]

            np.multiply(m, self.beta1, out=m)
            np.add(m, (1 - self.beta1) * grad, out=m)

            np.multiply(v, self.beta2, out=v)
            np.add(v, (1 - self.beta2) * grad**2, out=v)

            np.add(param.data, -lr_t * m / (np.sqrt(v) + self.eps), out=param.data)

    def set_lr(self, lr):
        self.lr = lr
