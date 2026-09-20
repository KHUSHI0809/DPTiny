import time
from dptiny import (
    Variable,
    MLP,
    SGD,
    get_mnist,
    DataLoader,
    softmax_cross_entropy,
    no_grad,
    use_gpu,
    to_gpu,
    is_gpu,
    is_available,
)

# Enable GPU if CuPy is installed and a CUDA device is available.
if is_available():
    use_gpu()
    print("GPU enabled for training.")
else:
    print("GPU not available; falling back to CPU.")

# Load MNIST dataset (always returned as NumPy arrays).
print("Loading MNIST dataset...")
X_train, X_test, y_train, y_test = get_mnist()

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Move data to the active device when the GPU backend is selected.
if is_gpu():
    X_train = to_gpu(X_train)
    X_test = to_gpu(X_test)
    y_train = to_gpu(y_train)
    y_test = to_gpu(y_test)

model = MLP(784, [100, 100], 10)

batch_size = 128
max_epoch = 10
initial_learning_rate = 0.1
momentum = 0.9

print(f"Training MLP with architecture: 784 -> 100 -> 100 -> 10")
print(f"Batch size: {batch_size}, Initial learning rate: {initial_learning_rate}, Momentum: {momentum}")

data_loader = DataLoader((X_train, y_train), batch_size)
test_loader = DataLoader((X_test, y_test), batch_size, shuffle=False)

optimizer = SGD(lr=initial_learning_rate, momentum=momentum)

start_time = time.time()
for epoch in range(max_epoch):
    learning_rate = initial_learning_rate * (0.1 ** (epoch // 3))
    optimizer.set_lr(learning_rate)

    sum_loss = 0.0
    count = 0

    for i, (x, t) in enumerate(data_loader):
        x = Variable(x)
        y = model.forward(x)
        loss = softmax_cross_entropy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update(model.params)

        if loss.data is not None:
            sum_loss += float(loss.data) * len(t)
            count += len(t)

        if (i + 1) % 20 == 0:
            avg_loss = sum_loss / count if count > 0 else float("inf")
            elapsed_time = time.time() - start_time
            print(f"epoch: {epoch+1}, batch: {i+1}, loss: {avg_loss:.4f}, lr: {learning_rate:.6f}, time: {elapsed_time:.2f}s")

    avg_loss = sum_loss / count if count > 0 else float("inf")

    sum_acc = 0.0
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
    print(f"epoch: {epoch+1}, final loss: {avg_loss:.4f}, accuracy: {test_acc:.4f}, time: {elapsed_time:.2f}s")

with no_grad():
    sum_acc = 0.0
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
