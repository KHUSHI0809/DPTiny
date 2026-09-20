import numpy as np
from dptiny import Variable, Conv2d

# Random 8 images of 1x28x28 (MNIST-like shape: NCHW)
x_data = np.random.randn(8, 1, 28, 28).astype(np.float32)
x = Variable(x_data)

# Conv layer: 1 -> 16 channels, 3x3 kernel, padding=1 to keep spatial size
conv = Conv2d(in_channels=1, out_channels=16, kernel_size=3, pad=1)
y = conv(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Kernel shape:", conv.W.shape)
print("Bias shape:", conv.b.shape if conv.b is not None else None)

# Backward
y.backward()
print("Gradient of input shape:", x.grad.shape)
print("Gradient of kernel shape:", conv.W.grad.shape)
