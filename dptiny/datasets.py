"""Backward-compatibility shim for the old ``dptiny.datasets`` module."""

from dptiny.data import DataLoader, Dataset, TensorDataset, get_mnist

__all__ = ["DataLoader", "Dataset", "TensorDataset", "get_mnist"]
