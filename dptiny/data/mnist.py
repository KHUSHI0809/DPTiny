"""MNIST dataset loader."""

import os
from typing import Optional, Tuple

from sklearn.datasets import fetch_openml

from dptiny.backend import xp

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dptiny")


def get_mnist(
    normalize: bool = True,
    flatten: bool = True,
    data_home: Optional[str] = None,
) -> Tuple["xp.ndarray", "xp.ndarray", "xp.ndarray", "xp.ndarray"]:
    """Load the MNIST dataset.

    Args:
        normalize: If ``True``, scale pixel values to ``[0, 1]``.
        flatten: If ``True``, return images as ``(N, 784)`` vectors.
        data_home: Directory used by scikit-learn to cache the download.

    Returns:
        ``(X_train, X_test, y_train, y_test)`` arrays.
    """
    cache_dir = data_home if data_home is not None else _CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        data_home=cache_dir,
        parser="auto",
    )
    X = mnist.data.astype(xp.float32)
    # fetch_openml may return string labels; convert to integer class indices.
    y = mnist.target.astype(xp.int32)

    if normalize:
        X = X / 255.0

    if not flatten:
        X = X.reshape(-1, 1, 28, 28)

    train_size = 60000
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test
