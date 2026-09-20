"""Backend abstraction for NumPy/CuPy array computation.

DPTiny can run on the CPU (NumPy) or on an NVIDIA GPU (CuPy).  This module
provides a tiny switching layer so that the rest of the framework does not
need to know which array library is actually doing the work.

Usage
-----
Use the active backend as a NumPy-compatible module::

    from dptiny.backend import xp
    x = xp.array([1.0, 2.0, 3.0])

Move arrays between host and device::

    from dptiny.backend import to_gpu, to_cpu
    x_gpu = to_gpu(x_cpu)
    x_cpu = to_cpu(x_gpu)

Switch backends globally::

    from dptiny.backend import use_gpu, use_cpu
    use_gpu()   # raises RuntimeError if CuPy/CUDA is unavailable
    use_cpu()
"""

from typing import Any, Union

import numpy as _np

try:
    import cupy as _cp
    _cupy_available = True
except ImportError:
    _cupy_available = False
    _cp = None


class _Backend:
    """Lightweight proxy that forwards attribute access to the active library."""

    def __init__(self):
        self._module = _np

    def use_gpu(self) -> None:
        """Switch the active backend to CuPy.

        Raises:
            RuntimeError: If CuPy is not installed or no CUDA device is found.
        """
        if not _cupy_available:
            raise RuntimeError(
                "CuPy is not installed. Install it with 'pip install cupy-cudaXX' "
                "where XX matches your CUDA toolkit version."
            )
        try:
            _cp.cuda.Device(0).use()
        except Exception as exc:
            raise RuntimeError("No CUDA device available") from exc
        self._module = _cp

    def use_cpu(self) -> None:
        """Switch the active backend back to NumPy."""
        self._module = _np

    @property
    def is_gpu(self) -> bool:
        """Return ``True`` if CuPy is currently selected."""
        return self._module is _cp

    @property
    def is_available(self) -> bool:
        """Return ``True`` if a GPU backend (CuPy + CUDA) is available."""
        return _cupy_available

    def __getattr__(self, name: str):
        return getattr(self._module, name)


# Public singleton used by the rest of the framework.
xp = _Backend()


def use_gpu() -> None:
    """Enable GPU computation globally."""
    xp.use_gpu()


def use_cpu() -> None:
    """Enable CPU computation globally (the default)."""
    xp.use_cpu()


def is_gpu() -> bool:
    """Return ``True`` if the framework is currently using the GPU."""
    return xp.is_gpu


def is_available() -> bool:
    """Return ``True`` if GPU acceleration is available on this machine."""
    return xp.is_available


def get_array_module(x: Any) -> Any:
    """Return the array module that owns ``x`` (numpy or cupy)."""
    if _cupy_available and isinstance(x, _cp.ndarray):
        return _cp
    return _np


_NDARRAY_TYPES: tuple = (_np.ndarray,)
if _cupy_available:
    _NDARRAY_TYPES = (_np.ndarray, _cp.ndarray)


def to_gpu(x: Any) -> Any:
    """Move a NumPy array or a ``Variable`` to the current GPU device.

    If ``x`` is already a CuPy array it is returned as-is.
    """
    if not _cupy_available:
        raise RuntimeError("CuPy is not installed; cannot move data to GPU")
    if isinstance(x, _NDARRAY_TYPES):
        return _cp.asarray(x)
    x.data = _cp.asarray(x.data)
    return x


def to_cpu(x: Any) -> Any:
    """Move a CuPy array or a ``Variable`` back to the host.

    If ``x`` is already a NumPy array it is returned as-is.
    """
    if isinstance(x, _NDARRAY_TYPES):
        if _cupy_available and isinstance(x, _cp.ndarray):
            return _cp.asnumpy(x)
        return x
    if _cupy_available and isinstance(x.data, _cp.ndarray):
        x.data = _cp.asnumpy(x.data)
    return x
