import time

from dptiny import (
    Variable,
    get_mnist,
    is_available,
    is_gpu,
    no_grad,
    softmax_cross_entropy,
    test_mode,
    to_gpu,
    use_gpu,
)
from dptiny.data import DataLoader
from dptiny.nn import (
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)
from dptiny.optim import Adam

if is_available():
    use_gpu()
    print("GPU enabled for training.")
else:
    print("GPU not available; training on CPU.")

print("Loading MNIST dataset...")
X_train, X_test, y_train, y_test = get_mnist(flatten=False)

if is_gpu():
    X_train = to_gpu(X_train)
    X_test = to_gpu(X_test)
    y_train = to_gpu(y_train)
    y_test = to_gpu(y_test)

model = Sequential(
    Conv2d(1, 16, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Conv2d(16, 32, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(32 * 7 * 7, 128),
    ReLU(),
    Dropout(0.3),
    Linear(128, 10),
)
if is_gpu():
    model.to_gpu()

batch_size = 128
max_epoch = 2
data_loader = DataLoader((X_train, y_train), batch_size)
test_loader = DataLoader((X_test, y_test), batch_size, shuffle=False)

optimizer = Adam(model, lr=0.001)

start_time = time.time()
for epoch in range(max_epoch):
    sum_loss = 0.0
    count = 0
    model.train()
    for i, (x, t) in enumerate(data_loader):
        x = Variable(x)
        y = model(x)
        loss = softmax_cross_entropy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        count += len(t)

        if (i + 1) % 50 == 0:
            print(
                f"epoch: {epoch + 1}, batch: {i + 1}, "
                f"loss: {sum_loss / count:.4f}, "
                f"time: {time.time() - start_time:.2f}s"
            )

    avg_loss = sum_loss / count
    sum_acc = 0.0
    count = 0
    with test_mode(), no_grad():
        for x, t in test_loader:
            y = model(Variable(x))
            pred = y.data.argmax(axis=1)
            acc = (pred == t).sum() / len(t)
            sum_acc += float(acc) * len(t)
            count += len(t)
    print(
        f"epoch: {epoch + 1}, loss: {avg_loss:.4f}, "
        f"test acc: {sum_acc / count:.4f}, "
        f"time: {time.time() - start_time:.2f}s"
    )

sum_acc = 0.0
count = 0
with test_mode(), no_grad():
    for x, t in test_loader:
        y = model(Variable(x))
        pred = y.data.argmax(axis=1)
        acc = (pred == t).sum() / len(t)
        sum_acc += float(acc) * len(t)
        count += len(t)

print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
print(f"Final test accuracy: {sum_acc / count:.4f}")
