import numpy as np

from dptiny import Variable, matmul

# Create input matrices
x_data = np.random.randn(3, 4).astype(np.float32)
W_data = np.random.randn(4, 5).astype(np.float32)

# Convert to Variables
x = Variable(x_data)
W = Variable(W_data)

# Perform matrix multiplication
y = matmul(x, W)

# Print results
print("Input matrix x shape:", x.shape)
print("Weight matrix W shape:", W.shape)
print("Output matrix y shape:", y.shape)
print("\nOutput matrix y data:")
print(y.data)

# Compute gradients
y.grad = np.ones_like(y.data)
y.backward()

# Print gradients
print("\nGradient of x:")
print(x.grad)
print("\nGradient of W:")
print(W.grad)

# Verify gradients with numpy
expected_x_grad = np.ones_like(y.data) @ W_data.T
expected_W_grad = x_data.T @ np.ones_like(y.data)

print("\nExpected gradient of x:")
print(expected_x_grad)
print("\nExpected gradient of W:")
print(expected_W_grad)

# Check if gradients match
x_grad_match = np.allclose(x.grad, expected_x_grad)
W_grad_match = np.allclose(W.grad, expected_W_grad)
print("\nGradients match expected values:", x_grad_match and W_grad_match)
