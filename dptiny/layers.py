"""Backward-compatibility shim for the old ``dptiny.layers`` module."""

from dptiny.nn import *  # noqa: F401,F403
from dptiny.nn import Module  # noqa: F401
from dptiny.optim import SGD, Adam  # noqa: F401

# The old module named its base class ``Layer``.
Layer = Module
