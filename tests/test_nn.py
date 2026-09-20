import numpy as np

import dptiny
from dptiny import Config, Variable, no_grad, test_mode
from dptiny.nn import (
    AvgPool2d,
    BatchNorm,
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    MLP,
    Module,
    ReLU,
    Sequential,
)


def test_param_registration_and_names():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 3))
    names = dict(model.named_parameters())
    assert set(names) == {"l0.W", "l0.b", "l2.W", "l2.b"}
    assert len(list(model.parameters())) == 4


def test_params_dict_compat():
    model = MLP(10, [20], 3)
    assert set(model.params) == {"l1.W", "l1.b", "l2.W", "l2.b"}
    model.cleargrads()  # should not raise
    model.zero_grad()


def test_mlp_forward_shape():
    model = MLP(10, [20, 20], 3)
    y = model(Variable(np.random.randn(5, 10).astype(np.float32)))
    assert y.shape == (5, 3)


def test_state_dict_roundtrip(tmp_path):
    model = MLP(4, [8], 2)
    path = str(tmp_path / "weights.npz")
    model.save_weights(path)

    model2 = MLP(4, [8], 2)
    model2.load_weights(path)
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model2.named_parameters()
    ):
        assert n1 == n2
        assert np.allclose(p1.data, p2.data)


def test_state_dict_includes_buffers():
    bn = BatchNorm(4)
    sd = bn.state_dict()
    assert "running_mean" in sd and "running_var" in sd
    assert "gamma" in sd and "beta" in sd


def test_sequential_indexable():
    s = Sequential(Linear(4, 4), ReLU())
    assert isinstance(s[0], Linear)
    assert len(s) == 2


def test_dropout_train_eval():
    d = Dropout(0.5)
    x = Variable(np.ones((100, 10), dtype=np.float32))
    Config.train = True
    y = d(x)
    assert (y.data == 0).any()
    Config.train = False
    y2 = d(x)
    assert np.allclose(y2.data, x.data)
    Config.train = True


def test_dropout_eval_via_test_mode():
    d = Dropout(0.9)
    x = Variable(np.ones((50,), dtype=np.float32))
    with test_mode():
        y = d(x)
        assert np.allclose(y.data, x.data)


def test_batch_norm_running_stats():
    bn = BatchNorm(5)
    x = Variable(np.random.randn(16, 5).astype(np.float32) * 4 + 2)
    Config.train = True
    bn(x)
    assert not np.allclose(bn.running_mean, 0)
    Config.train = True


def test_conv2d_output_shape():
    conv = Conv2d(1, 16, 3, pad=1)
    y = conv(Variable(np.random.randn(8, 1, 28, 28).astype(np.float32)))
    assert y.shape == (8, 16, 28, 28)


def test_pool_output_shapes():
    x = Variable(np.random.randn(2, 4, 8, 8).astype(np.float32))
    assert MaxPool2d(2)(x).shape == (2, 4, 4, 4)
    assert AvgPool2d(2)(x).shape == (2, 4, 4, 4)
    assert Flatten()(x).shape == (2, 4 * 8 * 8)


def test_to_cpu_noop():
    model = MLP(4, [4], 2)
    model.to_cpu()


def test_cnn_sequential_forward():
    model = Sequential(
        Conv2d(1, 4, 3, pad=1),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(4 * 14 * 14, 10),
    )
    x = Variable(np.random.randn(2, 1, 28, 28).astype(np.float32))
    y = model(x)
    assert y.shape == (2, 10)
    y.sum().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)


def test_module_is_module():
    assert isinstance(Linear(1, 1), Module)
