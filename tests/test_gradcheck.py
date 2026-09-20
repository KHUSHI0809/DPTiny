import numpy as np

import dptiny.functions as F
from dptiny import test_mode
from dptiny.backend import xp
from dptiny.utils import gradient_check


def check(f, *inputs):
    assert gradient_check(f, *inputs)


def test_add_broadcast():
    check(lambda a, b: a + b, np.random.randn(3, 4), np.random.randn(4))


def test_sub_broadcast():
    check(lambda a, b: a - b, np.random.randn(3, 4), np.random.randn(3, 1))


def test_mul_broadcast():
    check(lambda a, b: a * b, np.random.randn(3, 4), np.random.randn(4))


def test_div_broadcast():
    check(
        lambda a, b: a / b,
        np.random.randn(3, 4),
        np.random.randn(4) + 2.0,
    )


def test_matmul():
    check(lambda a, b: a @ b, np.random.randn(3, 4), np.random.randn(4, 5))


def test_linear():
    check(
        F.linear,
        np.random.randn(4, 6),
        np.random.randn(6, 3),
        np.random.randn(3),
    )


def test_exp():
    check(F.exp, np.random.randn(3, 3))


def test_log():
    check(F.log, np.random.rand(3, 3) + 0.5)


def test_tanh():
    check(F.tanh, np.random.randn(3, 3))


def test_sigmoid():
    check(F.sigmoid, np.random.randn(3, 3))


def test_relu():
    x = np.random.randn(4, 4)
    x[np.abs(x) < 0.2] = 0.5  # keep away from the kink at 0
    check(F.relu, x)


def test_sum():
    check(lambda x: x.sum(), np.random.randn(3, 4))
    check(lambda x: x.sum(axis=1), np.random.randn(3, 4))
    check(lambda x: x.sum(axis=0, keepdims=True), np.random.randn(3, 4))


def test_mean():
    check(lambda x: x.mean(), np.random.randn(3, 4))
    check(lambda x: x.mean(axis=0), np.random.randn(3, 4))


def test_max():
    check(lambda x: x.max(axis=1), np.random.randn(3, 4))
    check(lambda x: x.max(), np.random.randn(3, 4))


def test_reshape():
    check(lambda x: x.reshape(2, 6), np.random.randn(3, 4))


def test_transpose():
    check(lambda x: x.transpose(), np.random.randn(3, 4))
    check(lambda x: x.transpose(1, 0), np.random.randn(3, 4))


def test_get_item():
    check(lambda x: x[1], np.random.randn(3, 4))
    check(lambda x: x[:, 1:3], np.random.randn(3, 4))


def test_broadcast_to():
    check(lambda x: F.broadcast_to(x, (3, 4)), np.random.randn(4))


def test_sum_to():
    check(lambda x: F.sum_to(x, (4,)), np.random.randn(3, 4))


def test_conv2d_stride1_pad0():
    x = np.random.randn(2, 3, 6, 6)
    W = np.random.randn(4, 3, 3, 3)
    check(lambda a, w: F.conv2d(a, w, stride=1, pad=0), x, W)


def test_conv2d_stride2_pad1():
    x = np.random.randn(2, 3, 6, 6)
    W = np.random.randn(4, 3, 3, 3)
    check(lambda a, w: F.conv2d(a, w, stride=2, pad=1), x, W)


def test_max_pool2d():
    x = np.random.randn(2, 3, 6, 6)
    check(lambda a: F.max_pool2d(a, 2), x)


def test_avg_pool2d():
    x = np.random.randn(2, 3, 6, 6)
    check(lambda a: F.avg_pool2d(a, 2), x)


def test_softmax_cross_entropy():
    x = np.random.randn(4, 5)
    t = np.eye(5)[np.array([0, 2, 1, 4])]
    check(lambda a: F.softmax_cross_entropy(a, t), x)


def test_mean_squared_error():
    check(
        F.mean_squared_error,
        np.random.randn(4, 3),
        np.random.randn(4, 3),
    )


def test_batch_norm_train():
    x = np.random.randn(8, 5) * 3 + 1
    gamma = np.random.randn(5)
    beta = np.random.randn(5)
    rm = np.zeros(5)
    rv = np.ones(5)
    check(
        lambda a, g, b: F.batch_norm(a, g, b, rm, rv),
        x,
        gamma,
        beta,
    )
    # running stats were updated in train mode
    assert not np.allclose(rm, 0)


def test_batch_norm_eval_mode():
    x = np.random.randn(8, 5)
    gamma = np.random.randn(5)
    beta = np.random.randn(5)
    rm = np.zeros(5)
    rv = np.ones(5)
    with test_mode():
        check(lambda a, g, b: F.batch_norm(a, g, b, rm, rv), x, gamma, beta)
    assert np.allclose(rm, 0)  # unchanged in eval mode
