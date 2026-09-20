"""Array and autograd utility helpers for DPTiny."""

from typing import Any, Optional, Sequence, Tuple, Union

from dptiny.backend import xp


def pair(value: Union[int, Sequence[int]]) -> Tuple[int, int]:
    """Convert an integer or a length-2 sequence into a 2-tuple."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("value must be a scalar or a pair")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def sum_to(x: Any, shape: Tuple[int, ...]) -> Any:
    """Sum elements of ``x`` so that the result has ``shape``.

    This is the inverse of ``broadcast_to`` used in backward passes.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))
    axis = tuple(i + lead for i, sx in enumerate(shape) if sx == 1)
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        return y.squeeze(lead_axis)
    return y


def reshape_sum_backward(
    gy: Any,
    x_shape: Tuple[int, ...],
    axis: Optional[Union[int, Tuple[int, ...]]],
    keepdims: bool,
) -> Any:
    """Reshape a gradient so it can be broadcast back to ``x_shape``."""
    ndim = len(x_shape)
    if axis is None:
        tupled_axis = tuple(range(ndim))
    elif isinstance(axis, int):
        tupled_axis = (axis,)
    else:
        tupled_axis = tuple(a if a >= 0 else a + ndim for a in axis)

    if not (ndim == 0 or tupled_axis == () or keepdims):
        shape = list(gy.shape)
        actual = [a if a >= 0 else a + ndim for a in tupled_axis]
        for a in sorted(actual):
            shape.insert(a, 1)
        gy = gy.reshape(shape)
    return gy


def im2col_array(
    img: Any,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    pad: Union[int, Tuple[int, int]],
    to_matrix: bool = True,
) -> Any:
    """Extract sliding local blocks from an ``(N, C, H, W)`` image batch."""
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    img = xp.pad(
        img, ((0, 0), (0, 0), (PH, PH), (PW, PW)), mode="constant"
    )
    col = xp.zeros((N, C, KH, KW, OH, OW), dtype=img.dtype)

    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)
    return col


def col2im_array(
    col: Any,
    img_shape: Tuple[int, int, int, int],
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    pad: Union[int, Tuple[int, int]],
    to_matrix: bool = True,
) -> Any:
    """Inverse of :func:`im2col_array`; scatter-add columns back to images."""
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    img = xp.zeros(
        (N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1), dtype=col.dtype
    )
    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]

    return img[:, :, PH : H + PH, PW : W + PW]


def gradient_check(
    f,
    *inputs,
    eps: float = 1e-4,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> bool:
    """Compare analytic gradients from ``backward`` with central differences.

    ``inputs`` are ``Variable``s or raw arrays; they are converted to float64
    ``Variable``s.  The check compares ``d sum(f(x)) / dx`` with the analytic
    gradients and returns ``True`` when all of them match.
    """
    from dptiny.core import Variable, as_variable

    xs = [as_variable(x) for x in inputs]
    xs = [Variable(x.data.astype("float64")) for x in xs]
    y = f(*xs)
    y.backward()

    for x in xs:
        x_grad = x.grad
        if x_grad is None:
            continue
        num_grad = xp.zeros_like(x.data)
        it = xp.nditer(x.data, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = x.data[idx].copy()

            x.data[idx] = orig + eps
            y_pos = f(*xs).data.sum()

            x.data[idx] = orig - eps
            y_neg = f(*xs).data.sum()

            x.data[idx] = orig
            num_grad[idx] = (y_pos - y_neg) / (2 * eps)
            it.iternext()

        if not xp.allclose(x_grad, num_grad, rtol=rtol, atol=atol):
            return False
    return True
