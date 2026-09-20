import time

import numpy as np

from dptiny import (
    MLP,
    SGD,
    DataLoader,
    Variable,
    get_mnist,
    no_grad,
    softmax_cross_entropy,
)

# Load MNIST dataset
print("Loading MNIST dataset...")
X_train, X_test, y_train, y_test = get_mnist()

# Normalize data
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
# Standardize using mean and std of training data
mean = np.mean(X_train)
std = np.std(X_train)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Create model with smaller hidden layers for faster training
model = MLP(784, [100, 100], 10)  # input_size=784, hidden_sizes=[100, 100], output_size=10

# Training settings
batch_size = 128  # Increased batch size for better stability
max_epoch = 10
initial_learning_rate = 0.1
momentum = 0.9  # Add momentum for faster convergence

print("Training MLP with architecture: 784 -> 100 -> 100 -> 10")
print(f"Batch size: {batch_size}, Initial learning rate: {initial_learning_rate}, Momentum: {momentum}")

# Create data loader
data_loader = DataLoader((X_train, y_train), batch_size)
test_loader = DataLoader((X_test, y_test), batch_size, shuffle=False)

# Initialize optimizer
optimizer = SGD(lr=initial_learning_rate, momentum=momentum)

# Training loop
start_time = time.time()
for epoch in range(max_epoch):
    # Learning rate scheduling - reduce learning rate over time
    learning_rate = initial_learning_rate * (0.1 ** (epoch // 3))
    optimizer.set_lr(learning_rate)

    sum_loss = 0
    count = 0

    for i, (x, t) in enumerate(data_loader):
        # Convert to Variable
        x = Variable(x)

        # Forward
        y = model.forward(x)

        # Compute loss
        loss = softmax_cross_entropy(y, t)

        # Clear gradients
        model.cleargrads()

        # Backward
        loss.backward()

        # Update parameters using the optimizer
        optimizer.update(model.params)

        # Accumulate loss
        if loss.data is not None:
            sum_loss += float(loss.data) * len(t)
            count += len(t)

        # Print progress
        if (i + 1) % 20 == 0:
            avg_loss = sum_loss / count if count > 0 else float('inf')
            elapsed_time = time.time() - start_time
            print(f'epoch: {epoch+1}, batch: {i+1}, loss: {avg_loss:.4f}, lr: {learning_rate:.6f}, time: {elapsed_time:.2f}s')

    # Compute average loss for the epoch
    avg_loss = sum_loss / count if count > 0 else float('inf')

    # Evaluate on test set
    sum_acc = 0
    count = 0
    with no_grad():
        for x, t in test_loader:
            y = model.forward(Variable(x))
            pred = y.data.argmax(axis=1)
            acc = (pred == t).sum() / len(t)
            sum_acc += acc * len(t)
            count += len(t)

    test_acc = sum_acc / count if count > 0 else 0.0

    elapsed_time = time.time() - start_time
    print(f'epoch: {epoch+1}, final loss: {avg_loss:.4f}, accuracy: {test_acc:.4f}, time: {elapsed_time:.2f}s')

# Final evaluation
with no_grad():
    sum_acc = 0
    count = 0
    for x, t in test_loader:
        y = model.forward(Variable(x))
        pred = y.data.argmax(axis=1)
        acc = (pred == t).sum() / len(t)
        sum_acc += acc * len(t)
        count += len(t)

    final_acc = sum_acc / count
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
    print(f"Final test accuracy: {final_acc:.4f}")
