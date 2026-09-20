# DPTiny

A minimal implementation of a deep learning framework for educational purposes.

## Features

- Automatic differentiation (autograd) system
- Basic mathematical operations with operator overloading
- Convolution layer with learnable kernels
- NumPy-based computation
- Optional GPU acceleration via CuPy

## Installation

```bash
pip install -e .
```

For GPU acceleration install the matching CuPy wheel for your CUDA version, e.g.:

```bash
pip install -e ".[gpu]"
# or explicitly, e.g. for CUDA 12.x:
pip install cupy-cuda12x
```

## GPU Usage

DPTiny can transparently run on an NVIDIA GPU when CuPy is installed.
Enable the GPU backend and move your data to the device before training:

```python
from dptiny import use_gpu, to_gpu, Variable, MLP, get_mnist, DataLoader

use_gpu()  # raises RuntimeError if CuPy/CUDA is unavailable

X_train, X_test, y_train, y_test = get_mnist()
X_train = to_gpu(X_train.astype("float32"))
y_train = to_gpu(y_train)

model = MLP(784, [100, 100], 10)
```

All framework operations (autograd, layers, optimizers) then execute on the GPU
using the same NumPy-like API.

See `examples/mnist_gpu.py` for a complete GPU training example.

## Convolution Example

```python
import numpy as np
from dptiny import Variable, Conv2d

x = Variable(np.random.randn(8, 1, 28, 28).astype(np.float32))
conv = Conv2d(in_channels=1, out_channels=16, kernel_size=3, pad=1)
y = conv(x)
print(y.shape)  # (8, 16, 28, 28)
```

See `examples/conv_example.py` for a runnable example.

## Usage Example

```python
import numpy as np
from dptiny import Variable

# Create variables
x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

# Perform operations
z = x * y
print(z.data)  # 6.0

# Compute gradients
z.backward()
print(x.grad)  # 3.0
print(y.grad)  # 2.0
```

## Architecture

```mermaid
classDiagram
    class Config {
        +bool enable_backprop
    }

    class Variable {
        -np.ndarray _grad
        +np.ndarray data
        +str name
        +int generation
        +Function creator
        +tuple shape
        +int ndim
        +int size
        +grad
        +cleargrad()
        +backward(retain_grad)
        +__matmul__(other)
    }

    class Function {
        <<abstract>>
        +int generation
        +list inputs
        +list outputs
        +__call__(*inputs)
        +forward(*xs)*
        +backward(*gys)*
    }

    class Layer {
        <<abstract>>
        +dict params
        +dict grads
        +__call__(*inputs)
        +forward(inputs)*
        +backward(grads)*
        +cleargrads()
    }

    class Add {
        +forward(x0, x1)
        +backward(gy)
    }

    class Mul {
        +forward(x0, x1)
        +backward(gy)
    }

    class MatMul {
        +forward(x, W)
        +backward(gy)
    }

    class Conv2d {
        +forward(x, W)
        +backward(gy)
    }

    class Reshape {
        +forward(x, shape)
        +backward(gy)
    }

    class Exp {
        +forward(x)
        +backward(gy)
    }

    class Log {
        +forward(x)
        +backward(gy)
    }

    class Sigmoid {
        +forward(x)
        +backward(gy)
    }

    class ReLU {
        +forward(x)
        +backward(gy)
    }

    class Softmax {
        +forward(x)
        +backward(gy)
    }

    class SoftmaxCrossEntropy {
        +forward(x, t)
        +backward(gy)
    }

    class Neg {
        +forward(x)
        +backward(gy)
    }

    class Sub {
        +forward(x0, x1)
        +backward(gy)
    }

    class Div {
        +forward(x0, x1)
        +backward(gy)
    }

    class Linear {
        +int in_size
        +int out_size
        +Variable W
        +Variable b
        +forward(x)
        +backward(grads)
    }

    class Conv2d {
        +int in_channels
        +int out_channels
        +tuple kernel_size
        +Variable W
        +Variable b
        +forward(x)
        +backward(grads)
    }

    class MLP {
        +list layers
        +forward(x)
        +backward(grads)
        +predict(x)
        +accuracy(x, t)
    }

    class SGD {
        +float lr
        +float momentum
        +dict vs
        +update(params)
        +set_lr(lr)
    }

    class DataLoader {
        +tuple dataset
        +int batch_size
        +bool shuffle
        +int max_iter
        +int iteration
        +__iter__()
        +__next__()
        +reset()
    }

    Variable --> Function : creator
    Function --> Variable : inputs/outputs
    Function <|-- Add
    Function <|-- Mul
    Function <|-- MatMul
    Function <|-- Conv2d
    Function <|-- Reshape
    Function <|-- Exp
    Function <|-- Log
    Function <|-- Sigmoid
    Function <|-- ReLU
    Function <|-- Softmax
    Function <|-- SoftmaxCrossEntropy
    Function <|-- Neg
    Function <|-- Sub
    Function <|-- Div
    Layer <|-- Linear
    Layer <|-- Conv2d
    Layer <|-- ReLU
    Layer <|-- MLP
    Layer --> Variable : params
    MLP --> Linear : contains
    MLP --> ReLU : contains
```

An interactive HTML visualization of this architecture and the compute graph is
available at [`docs/architecture.html`](docs/architecture.html).

## Requirements

- NumPy
- Optional: CuPy for GPU acceleration
