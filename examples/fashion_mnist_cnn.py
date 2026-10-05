import time

import numpy as np

from dptiny import (
    Variable,
    is_available,
    is_gpu,
    no_grad,
    softmax_cross_entropy,
    test_mode,
    to_cpu,
    to_gpu,
    use_gpu,
)
from dptiny.backend import xp
from dptiny.data import DataLoader
from dptiny.data.fashion_mnist import CLASSES, get_fashion_mnist
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

# Random seed for reproducibility
np.random.seed(42)

if is_available():
    use_gpu()
    print("GPU enabled for training.")
else:
    print("GPU not available; training on CPU.")
xp.random.seed(42)

print("Loading Fashion-MNIST dataset...")

X_train, X_test, y_train, y_test = get_fashion_mnist(
    normalize=True,
    flatten=False,
)

print("Training data:", X_train.shape, y_train.shape)
print("Test data:", X_test.shape, y_test.shape)
print("Data type:", X_train.dtype, y_train.dtype)

if is_gpu():
    X_train = to_gpu(X_train)
    X_test = to_gpu(X_test)
    y_train = to_gpu(y_train)
    y_test = to_gpu(y_test)

# Calculate mean and standard deviation from the training set only
train_mean = X_train.mean()
train_std = X_train.std()

print(f"Training mean: {float(train_mean):.6f}")
print(f"Training std:  {float(train_std):.6f}")

# Standardize training and test data using training statistics
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

print("\nStandardization complete.")
print(f"X_train mean: {float(X_train.mean()):.6f}")
print(f"X_train std:  {float(X_train.std()):.6f}")
print(f"X_test mean:  {float(X_test.mean()):.6f}")
print(f"X_test std:   {float(X_test.std()):.6f}")

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

print("CNN model created successfully.")

batch_size = 64
max_epoch = 10

data_loader = DataLoader(
    (X_train, y_train),
    batch_size,
)

test_loader = DataLoader(
    (X_test, y_test),
    batch_size,
    shuffle=False,
)

optimizer = Adam(model, lr=0.001)

print(f"Batch size: {batch_size}")
print(f"Epochs: {max_epoch}")
print("Optimizer: Adam")
print("Learning rate: 0.001")

start_time = time.time()

for epoch in range(max_epoch):

    # Training
    sum_loss = 0.0
    train_correct = 0
    train_count = 0

    model.train()

    for i, (x, t) in enumerate(data_loader):
        x = Variable(x)

        # Forward pass
        y = model(x)

        # Calculate loss
        loss = softmax_cross_entropy(y, t)

        # Backpropagation and parameter update
        model.cleargrads()
        loss.backward()
        optimizer.update()

        # Track training loss
        batch_size_actual = len(t)
        sum_loss += float(loss.data) * batch_size_actual
        train_count += batch_size_actual

        # Track training accuracy
        pred = y.data.argmax(axis=1)
        train_correct += int((pred == t).sum())

    train_loss = sum_loss / train_count
    train_acc = train_correct / train_count

    # Testing
    sum_acc = 0.0
    test_count = 0

    with test_mode(), no_grad():
        for x, t in test_loader:
            y = model(Variable(x))

            pred = y.data.argmax(axis=1)
            acc = (pred == t).sum()

            sum_acc += float(acc)
            test_count += len(t)

    test_acc = sum_acc / test_count

    print(
        f"epoch: {epoch + 1:2d}/{max_epoch}, "
        f"training loss: {train_loss:.4f}, "
        f"training acc: {train_acc:.4f}, "
        f"test acc: {test_acc:.4f}, "
        f"time: {time.time() - start_time:.2f}s"
    )

print(f"\nTraining completed in {time.time() - start_time:.2f} seconds.")

# 10 x 10 confusion matrix
confusion_matrix = np.zeros((10, 10), dtype=np.int64)

model.eval()

with test_mode(), no_grad():
    for x, t in test_loader:
        y = model(Variable(x))
        pred = y.data.argmax(axis=1)

        # Converting to NumPy arrays for the confusion matrix
        pred_cpu = to_cpu(pred)
        t_cpu = to_cpu(t)

        for true_label, predicted_label in zip(t_cpu, pred_cpu):
            confusion_matrix[int(true_label), int(predicted_label)] += 1

print("\n10 x 10 Confusion Matrix:")
print(confusion_matrix)

# per-class accuracy
class_correct = np.diag(confusion_matrix)
class_total = confusion_matrix.sum(axis=1)

class_accuracy = class_correct / class_total

print("\nPer-Class Accuracy:")

for i, class_name in enumerate(CLASSES):
    print(
        f"{i}: {class_name:<12} "
        f"{class_correct[i]:4d}/{class_total[i]:4d} "
        f"= {class_accuracy[i] * 100:.2f}%"
    )

# Most Confused Class Pair
# Adding both directions together
pair_confusion = confusion_matrix + confusion_matrix.T

# Ignoring the diagonal because a class confused with itself
np.fill_diagonal(pair_confusion, 0)

# Finding the pair with the largest combined confusion
class_a, class_b = np.unravel_index(
    np.argmax(pair_confusion),
    pair_confusion.shape
)

if class_a > class_b:
    class_a, class_b = class_b, class_a

combined_count = pair_confusion[class_a, class_b]

print("\nMost Confused Class Pair:")
print(
    f"{class_a} ({CLASSES[class_a]}) <-> "
    f"{class_b} ({CLASSES[class_b]})"
)
print(f"Combined confusion count: {combined_count}")

print("\nDirectional confusion:")
print(
    f"{CLASSES[class_a]} -> {CLASSES[class_b]}: "
    f"{confusion_matrix[class_a, class_b]}"
)
print(
    f"{CLASSES[class_b]} -> {CLASSES[class_a]}: "
    f"{confusion_matrix[class_b, class_a]}"
)
