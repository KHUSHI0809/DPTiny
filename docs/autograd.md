# Understanding Automatic Differentiation: A Deep Dive into DPTiny

Automatic differentiation (autograd) is the backbone of modern deep learning frameworks. In this article, we'll explore how it works by examining the implementation in **DPTiny**, a minimal deep learning framework built for educational purposes.

## Table of Contents

1. [What is Automatic Differentiation?](#what-is-automatic-differentiation)
2. [The Computational Graph](#the-computational-graph)
3. [Core Components](#core-components)
4. [Forward Pass](#forward-pass)
5. [Backward Pass (Backpropagation)](#backward-pass-backpropagation)
6. [Function Implementations](#function-implementations)
7. [Building Neural Networks](#building-neural-networks)
8. [Putting It All Together](#putting-it-all-together)

---

## What is Automatic Differentiation?

Automatic differentiation is a technique to compute derivatives of functions expressed as computer programs. Unlike:

- **Symbolic differentiation**: Manipulates mathematical expressions (like Mathematica)
- **Numerical differentiation**: Uses finite differences (like `f(x+h) - f(x) / h`)

**Automatic differentiation** computes exact derivatives by propagating values through a computational graph.

### Why Do We Need It?

In deep learning, we need to compute gradients to update model parameters:

```python
loss = (y_pred - y_true) ** 2
# We need: d(loss)/d(weights) to perform gradient descent
```

Autograd computes these gradients automatically!

---

## The Computational Graph

When we perform operations on variables, we build a **computational graph** that tracks dependencies:

```python
x = Variable(np.array(2.0))
y = Variable(np.array(3.0))
z = x * y  # z depends on x and y
```

This creates a graph:

```
    x    y
     \  /
      *
      |
      z
```

### Generation: Ordering the Backward Pass

Each node has a **generation** number that determines the order of backpropagation:

```
x (gen=0)  y (gen=0)
     \     /
      * (gen=1)
       |
      z (gen=1)
```

When we call `z.backward()`, we process nodes from higher to lower generation, ensuring gradients flow correctly.

---

## Core Components

### 1. The `Variable` Class

The `Variable` class wraps data and tracks gradient information:

```python
class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')
        self.data = data          # The actual value
        self.name = name
        self._grad = None         # Gradient (computed during backward)
        self.creator = None       # The Function that created this variable
        self.generation = 0       # For ordering backward pass
```

**Key attributes:**

| Attribute | Purpose |
|-----------|---------|
| `data` | The NumPy array holding the value |
| `grad` | The gradient (derivative of loss w.r.t. this variable) |
| `creator` | The function that produced this variable |
| `generation` | Depth in computational graph |

### 2. The `Function` Class

Functions transform variables and remember how to compute gradients:

```python
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
```

**The `__call__` method:**
1. Converts inputs to `Variable` objects
2. Extracts raw data and calls `forward()`
3. Wraps outputs in new `Variable` objects
4. Records connections for backpropagation (if enabled)

---

## Forward Pass

During the forward pass, we compute values and build the computational graph.

### Example: Multiplication

```python
class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1.data, gy * x0.data
```

**Forward pass:**
```
z = x * y

Given: x = 2, y = 3
Result: z = 6
```

**The graph records:**
- `z.creator = Mul()`
- `Mul.inputs = [x, y]`

### Operator Overloading

DPTiny overloads Python operators for natural syntax:

```python
class Variable:
    # ... inside Variable class ...
    
    def __add__(self, other):
        from dptiny.functions import add
        return add(self, other)
    
    def __mul__(self, other):
        from dptiny.functions import mul
        return mul(self, other)
```

Now we can write:

```python
x = Variable(np.array(2.0))
y = Variable(np.array(3.0))
z = x * y + x  # Natural syntax!
```

---

## Backward Pass (Backpropagation)

### The Chain Rule

The chain rule is the foundation of backpropagation:

```
dy/dx = dy/du * du/dx
```

In neural networks, we compute:

```
d(loss)/d(x) = d(loss)/d(z) * d(z)/d(x)
```

### Implementation

```python
def backward(self, retain_grad=False):
    if self.grad is None:
        self.grad = np.ones_like(self.data)  # Start with gradient of 1

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
        gxs = f.backward(*gys)  # Call the function's backward
        if not isinstance(gxs, tuple):
            gxs = (gxs,)

        for x, gx in zip(f.inputs, gxs):
            if x.grad is None:
                x.grad = gx
            else:
                x.grad = x.grad + gx  # Accumulate gradients

            if x.creator is not None:
                add_func(x.creator)

        if not retain_grad:
            for y in f.outputs:
                y().grad = None
```

**Key steps:**

1. **Initialize**: Set gradient of output to 1 (d(loss)/d(loss) = 1)
2. **Process functions by generation**: Using a max-heap (negated for min-heap)
3. **Get output gradients**: `gys = [output().grad for output in f.outputs]`
4. **Compute input gradients**: `gxs = f.backward(*gys)`
5. **Accumulate gradients**: `x.grad = x.grad + gx` (important for variables used multiple times)
6. **Continue recursively**: Add creator functions to the heap

### Visualization

For `z = x * y`:

```
Forward:
x(2.0) ----\
            * ---> z(6.0)
y(3.0) ----/

Backward (after z.backward()):
x.grad = d(z)/d(x) = y = 3.0
y.grad = d(z)/d(y) = x = 2.0
```

### Gradient Accumulation

When a variable is used multiple times:

```python
x = Variable(np.array(2.0))
y = x + x  # x is used twice
y.backward()
# x.grad = 2 (1 + 1)
```

This is why we use `x.grad = x.grad + gx` instead of `x.grad = gx`.

---

## Function Implementations

### Basic Operations

#### Addition with Broadcasting

```python
class Add(Function):
    def forward(self, x0, x1):
        self._x0_shape = x0.shape
        self._x1_shape = x1.shape
        return x0 + x1

    def backward(self, gy):
        gx0, gx1 = gy, gy
        
        # Handle broadcasting in backward pass
        if self._x0_shape != self._x1_shape:
            # Sum gradients along broadcasted dimensions
            if np.ndim(gx0) > len(self._x0_shape):
                axes = tuple(range(np.ndim(gx0) - len(self._x0_shape)))
                gx0 = gx0.sum(axis=axes, keepdims=True)
            
            for i, (dim0, dim1) in enumerate(
                zip(self._x0_shape[::-1], self._x1_shape[::-1])
            ):
                axis = -i - 1
                if dim0 == 1:
                    gx0 = gx0.sum(axis=axis, keepdims=True)
                if dim1 == 1:
                    gx1 = gx1.sum(axis=axis, keepdims=True)
                    
        return gx0, gx1
```

**Broadcasting example:**

```python
x = Variable(np.array([[1, 2, 3]]))  # Shape: (1, 3)
b = Variable(np.array([10, 20, 30])) # Shape: (3,)
y = x + b  # Broadcasting: (1, 3) + (3,) -> (1, 3)
```

In backward, we need to sum the gradient along the broadcasted dimensions.

### Matrix Multiplication

```python
class MatMul(Function):
    def forward(self, x, W):
        self.x_shape = x.shape
        self.W_shape = W.shape
        return x @ W

    def backward(self, gy):
        x, W = self.inputs
        gx = gy @ W.data.T
        gW = x.data.T @ gy
        
        if gx.shape != self.x_shape:
            gx = gx.reshape(self.x_shape)
        if gW.shape != self.W_shape:
            gW = gW.reshape(self.W_shape)
            
        return gx, gW
```

**Mathematical derivation:**

For `Y = X @ W`:

```
d(L)/d(X) = d(L)/d(Y) @ W^T
d(L)/d(W) = X^T @ d(L)/d(Y)
```

### Activation Functions

#### Sigmoid (Numerically Stable)

```python
class Sigmoid(Function):
    def forward(self, x):
        self._positive_mask = x >= 0
        # Numerically stable implementation
        y = np.where(
            self._positive_mask,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
        self._y = y
        return y

    def backward(self, gy):
        return gy * self._y * (1 - self._y)
```

**Why the stable version?**

Standard: `sigmoid(x) = 1 / (1 + exp(-x))`

- For large positive `x`: `exp(-x)` → 0, OK
- For large negative `x`: `exp(-x)` → ∞, overflow!

Stable version uses:
- `1 / (1 + exp(-x))` for x ≥ 0
- `exp(x) / (1 + exp(x))` for x < 0

#### ReLU

```python
class ReLU(Function):
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, gy):
        x = self.inputs[0]
        mask = x.data > 0
        return gy * mask
```

Derivative: 1 if x > 0, else 0.

### Softmax Cross-Entropy Loss

```python
class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        self.t = t
        self.x_shape = x.shape
        batch_size = x.shape[0]
        
        # Convert to one-hot if needed
        if self.t.size == self.t.shape[0]:
            self.t_oh = np.eye(x.shape[1], dtype=x.dtype)[self.t]
        else:
            self.t_oh = self.t

        # Numerically stable softmax
        x_max = x.max(axis=1, keepdims=True)
        x_shifted = x - x_max
        exp_x = np.exp(x_shifted)
        sum_exp_x = exp_x.sum(axis=1, keepdims=True)
        self.y = exp_x / sum_exp_x

        # Stable log-softmax
        log_y = x_shifted - np.log(sum_exp_x)
        batch_log_y = np.sum(self.t_oh * log_y, axis=1)
        loss = -np.sum(batch_log_y) / batch_size
        return loss

    def backward(self, gy):
        batch_size = self.x_shape[0]
        dx = (self.y - self.t_oh) * gy / batch_size
        return dx
```

**The beautiful result:**

```
∂(cross_entropy(softmax(x), t)) / ∂x = (softmax(x) - t) / batch_size
```

This elegant gradient is why softmax and cross-entropy are often combined.

---

## Building Neural Networks

### Linear Layer

```python
class Linear(Layer):
    def __init__(self, in_size, out_size, nobias=False, init_type='he'):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        # He initialization for ReLU networks
        if init_type == 'he':
            W_data = np.random.randn(in_size, out_size).astype(np.float32) * np.sqrt(2.0 / in_size)
        elif init_type == 'xavier':
            W_data = np.random.randn(in_size, out_size).astype(np.float32) * np.sqrt(1.0 / in_size)
        
        self.W = Variable(W_data)
        self.b = Variable(np.zeros(out_size, dtype=np.float32))
        self.params = {'W': self.W, 'b': self.b}
        
    def forward(self, x):
        y = matmul(x, self.W)
        if self.b is not None:
            y = y + self.b
        return y
```

### Multi-Layer Perceptron (MLP)

```python
class MLP(Layer):
    def __init__(self, in_size, hidden_sizes, out_size):
        super().__init__()
        self.layers = []
        self.params = {}
        
        sizes = [in_size] + hidden_sizes + [out_size]
        for i in range(len(sizes) - 1):
            layer = Linear(sizes[i], sizes[i + 1])
            self.layers.append(layer)
            
            # Collect parameters
            for key, param in layer.params.items():
                self.params[f'l{i+1}.{key}'] = param
            
            if i != len(sizes) - 2:  # Add ReLU between layers
                self.layers.append(ReLU())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

---

## Putting It All Together

### Complete Training Example

```python
import numpy as np
from dptiny import Variable, MLP, SGD, get_mnist, DataLoader, softmax_cross_entropy

# Load data
X_train, X_test, y_train, y_test = get_mnist()

# Create model
model = MLP(784, [100, 100], 10)

# Optimizer
optimizer = SGD(lr=0.1, momentum=0.9)

# Data loader
data_loader = DataLoader((X_train, y_train), batch_size=128)

# Training loop
for epoch in range(10):
    for x, t in data_loader:
        x = Variable(x)
        
        # Forward
        y = model.forward(x)
        loss = softmax_cross_entropy(y, t)
        
        # Clear gradients
        model.cleargrads()
        
        # Backward
        loss.backward()
        
        # Update
        optimizer.update(model.params)
    
    print(f'Epoch {epoch+1}, Loss: {loss.data:.4f}')
```

### How Gradients Flow

```
Input (784)
    |
Linear(784, 100) + bias
    |
  ReLU
    |
Linear(100, 100) + bias
    |
  ReLU
    |
Linear(100, 10) + bias
    |
SoftmaxCrossEntropy
    |
  Loss
```

When `loss.backward()` is called:

1. Loss gradient = 1.0
2. Flows through SoftmaxCrossEntropy → `d(loss)/d(logits)`
3. Flows through Linear layers → `d(loss)/d(W)`, `d(loss)/d(b)`
4. Optimizer updates: `W = W - lr * d(loss)/d(W)`

---

## Advanced Topics

### Disabling Gradient Computation

For inference, we can disable backprop:

```python
with no_grad():
    predictions = model.predict(x)
```

### Memory Management with WeakRef

DPTiny uses `weakref` to prevent memory leaks:

```python
self.outputs = [weakref.ref(output) for output in outputs]
```

This allows Python to garbage-collect intermediate variables when they're no longer needed.

---

## Summary

**Automatic differentiation in DPTiny works by:**

1. **Forward pass**: Building a computational graph of `Variable` and `Function` nodes
2. **Recording dependencies**: Each function stores its inputs and outputs
3. **Backward pass**: Using the chain rule to propagate gradients from output to input
4. **Generation ordering**: Processing functions in correct order using a priority queue
5. **Gradient accumulation**: Properly handling variables used multiple times

The elegance of autograd is that we write forward computations naturally, and the framework automatically computes gradients. This is what enables training deep neural networks with millions of parameters!

---

## Further Reading

- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)
- [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
- [PyTorch Autograd Documentation](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

---

*DPTiny is inspired by the [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3) project and aims to make automatic differentiation accessible and understandable.*
