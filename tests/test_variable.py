import numpy as np

from dptiny import Variable, no_grad


def test_generation_ordering():
    x = Variable(np.array(2.0))
    y = x + x
    assert y.generation == 1  # Variable(0) -> Add(0) -> y(1)
    y.backward()
    assert np.allclose(x.grad, 2.0)


def test_diamond_graph():
    x = Variable(np.array(3.0))
    a = x * x
    b = x * x
    z = a + b
    z.backward()
    # dz/dx = 4x = 12
    assert np.allclose(x.grad, 12.0)


def test_repr():
    x = Variable(np.array([1.0, 2.0]))
    r = repr(x)
    assert "variable(" in r
    assert "1." in r


def test_reshape_and_T():
    x = Variable(np.arange(6).reshape(2, 3).astype(np.float64))
    assert x.reshape(3, 2).shape == (3, 2)
    assert x.T.shape == (3, 2)
    assert x.shape == (2, 3)
    assert x.ndim == 2
    assert len(x) == 2


def test_getitem():
    x = Variable(np.arange(6).reshape(2, 3).astype(np.float64))
    y = x[0]
    assert np.allclose(y.data, [0, 1, 2])
    z = y.sum()
    z.backward()
    assert np.allclose(x.grad, [[1, 1, 1], [0, 0, 0]])


def test_unchain_backward():
    x = Variable(np.array(2.0))
    y = x * x
    z = y * y
    z.unchain_backward()
    assert y.creator is None
    assert x.creator is None


def test_no_grad():
    x = Variable(np.array(2.0))
    with no_grad():
        y = x * x
    assert y.creator is None


def test_backward_clears_intermediate_grads():
    x = Variable(np.array(2.0))
    y = x * x
    z = y + x
    z.backward()
    assert y.grad is None
    assert x.grad is not None
