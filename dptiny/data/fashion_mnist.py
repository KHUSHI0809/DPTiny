"""Fashion-MNIST dataset loader."""

import gzip
import os
import tempfile
import urllib.request
from typing import Optional, Tuple

import numpy as np

from dptiny.backend import xp

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dptiny")

_BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"

_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def _download(url: str, destination: str) -> None:
    """Download a file atomically."""
    if os.path.exists(destination):
        return

    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)

    fd, temporary = tempfile.mkstemp(dir=directory)
    os.close(fd)

    try:
        urllib.request.urlretrieve(url, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _parse_idx(path: str) -> np.ndarray:
    """Parse a gzipped IDX file."""
    with gzip.open(path, "rb") as f:
        data = f.read()

    if len(data) < 4:
        raise ValueError("Invalid IDX file: header is too short.")

    if data[0:2] != b"\x00\x00" or data[2] != 0x08:
        raise ValueError("Invalid IDX file: expected unsigned-byte data.")

    ndim = data[3]
    header_size = 4 + 4 * ndim

    if len(data) < header_size:
        raise ValueError("Invalid IDX file: header is incomplete.")

    shape = tuple(
        int.from_bytes(
            data[4 + 4 * i:8 + 4 * i],
            byteorder="big",
        )
        for i in range(ndim)
    )

    expected_size = header_size + int(np.prod(shape))

    if len(data) != expected_size:
        raise ValueError(
            f"Invalid IDX file: expected {expected_size} bytes, "
            f"found {len(data)}."
        )

    return np.frombuffer(
        data,
        dtype=np.uint8,
        offset=header_size,
    ).reshape(shape)


def get_fashion_mnist(
    normalize: bool = True,
    flatten: bool = True,
    data_home: Optional[str] = None,
) -> Tuple["xp.ndarray", "xp.ndarray", "xp.ndarray", "xp.ndarray"]:

    cache_dir = data_home if data_home is not None else _CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    paths = {}

    for key, filename in _FILES.items():
        path = os.path.join(cache_dir, filename)
        url = f"{_BASE_URL}/{filename}"
        _download(url, path)
        paths[key] = path

    X_train = _parse_idx(paths["train_images"])
    y_train = _parse_idx(paths["train_labels"])
    X_test = _parse_idx(paths["test_images"])
    y_test = _parse_idx(paths["test_labels"])

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    if flatten:
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)
    else:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    X_train = xp.asarray(X_train)
    X_test = xp.asarray(X_test)
    y_train = xp.asarray(y_train)
    y_test = xp.asarray(y_test)

    return X_train, X_test, y_train, y_test