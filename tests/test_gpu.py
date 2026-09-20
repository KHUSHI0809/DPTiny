import numpy as np
import pytest

import dptiny
from dptiny import Variable
from dptiny.backend import xp

pytestmark = pytest.mark.skipif(
    not dptiny.is_available(), reason="GPU (CuPy) not available"
)


@pytest.fixture
def gpu():
    dptiny.use_gpu()
    yield
    dptiny.use_cpu()


def test_mlp_gpu_matches_cpu(gpu):
    np.random.seed(0)
    x_np = np.random.randn(8, 10).astype(np.float32)

    # CPU reference
    dptiny.use_cpu()
    np.random.seed(1)
    cpu_model = dptiny.nn.MLP(10, [16], 4)
    cpu_state = cpu_model.state_dict()
    y_cpu = cpu_model(Variable(x_np.copy()))
    loss_cpu = y_cpu.sum()
    loss_cpu.backward()
    cpu_grads = {n: p.grad.copy() for n, p in cpu_model.named_parameters()}

    # GPU
    dptiny.use_gpu()
    gpu_model = dptiny.nn.MLP(10, [16], 4)
    gpu_model.load_state_dict(cpu_state)
    gpu_model.to_gpu()
    x_gpu = Variable(xp.asarray(x_np))
    y_gpu = gpu_model(x_gpu)
    assert isinstance(y_gpu.data, type(xp.asarray(0)))
    loss_gpu = y_gpu.sum()
    loss_gpu.backward()

    assert np.allclose(dptiny.to_cpu(y_gpu.data), y_cpu.data, atol=1e-5)
    for name, p in gpu_model.named_parameters():
        assert np.allclose(
            dptiny.to_cpu(p.grad), cpu_grads[name], atol=1e-5
        ), name


def test_conv2d_gpu_matches_cpu(gpu):
    np.random.seed(0)
    x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
    w_np = np.random.randn(4, 3, 3, 3).astype(np.float32)

    dptiny.use_cpu()
    x = Variable(x_np.copy())
    w = Variable(w_np.copy())
    y = dptiny.conv2d(x, w, pad=1)
    y.sum().backward()
    gx_cpu, gw_cpu = x.grad.copy(), w.grad.copy()
    y_cpu = y.data.copy()

    dptiny.use_gpu()
    xg = Variable(xp.asarray(x_np))
    wg = Variable(xp.asarray(w_np))
    yg = dptiny.conv2d(xg, wg, pad=1)
    yg.sum().backward()

    assert np.allclose(dptiny.to_cpu(yg.data), y_cpu, atol=1e-5)
    assert np.allclose(dptiny.to_cpu(xg.grad), gx_cpu, atol=1e-5)
    assert np.allclose(dptiny.to_cpu(wg.grad), gw_cpu, atol=1e-5)


def test_variable_to_gpu_and_back(gpu):
    x = Variable(xp.asarray(np.array([1.0, 2.0], dtype=np.float32)))
    x.to_gpu()
    assert dptiny.is_gpu()
    x.to_cpu()
    assert isinstance(x.data, np.ndarray)
