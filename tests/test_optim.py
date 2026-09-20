import numpy as np
import pytest

from dptiny import Variable
from dptiny.optim import (
    SGD,
    Adam,
    AdamW,
    ClipGrad,
    CosineAnnealingLR,
    ExponentialLR,
    RMSprop,
    StepLR,
    WeightDecay,
)


def _quadratic(opt):
    p = Variable(np.array([5.0, -3.0], dtype=np.float64))
    losses = []
    for _ in range(50):
        loss = (p * p).sum()
        p.cleargrad()
        loss.backward()
        opt.update([p])
        losses.append(float(loss.data))
    return losses


@pytest.mark.parametrize(
    "make_opt",
    [
        lambda: SGD(lr=0.1),
        lambda: SGD(lr=0.05, momentum=0.9),
        lambda: Adam(lr=0.1),
        lambda: AdamW(lr=0.1),
        lambda: RMSprop(lr=0.05),
    ],
)
def test_optimizer_decreases_loss(make_opt):
    losses = _quadratic(make_opt())
    assert losses[-1] < losses[0] * 0.1


def test_update_with_module():
    model = dptiny_nn_mlp()
    x = Variable(np.random.randn(4, 4).astype(np.float32))
    t = np.array([0, 1, 2, 1])
    import dptiny.functions as F

    opt = SGD(model, lr=0.1)
    for _ in range(3):
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        opt.update()


def dptiny_nn_mlp():
    from dptiny.nn import MLP

    return MLP(4, [8], 3)


def test_weight_decay_hook():
    p = Variable(np.array([1.0]))
    opt = SGD(lr=0.0)
    opt.add_hook(WeightDecay(0.5))
    p.grad = np.zeros(1)
    for hook in opt.hooks:
        hook([p])
    assert np.allclose(p.grad, 0.5)


def test_clip_grad_hook():
    p = Variable(np.array([3.0, 4.0]))
    p.grad = np.array([3.0, 4.0])
    ClipGrad(1.0)([p])
    assert np.isclose(float((p.grad**2).sum() ** 0.5), 1.0)


def test_step_lr():
    opt = SGD(lr=1.0)
    sch = StepLR(opt, step_size=2, gamma=0.5)
    sch.step()
    assert np.isclose(opt.lr, 1.0)
    sch.step()
    assert np.isclose(opt.lr, 0.5)


def test_exponential_lr():
    opt = SGD(lr=1.0)
    sch = ExponentialLR(opt, gamma=0.9)
    sch.step()
    sch.step()
    assert np.isclose(opt.lr, 0.81)


def test_cosine_annealing_lr():
    opt = SGD(lr=1.0)
    sch = CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
    for _ in range(10):
        sch.step()
    assert np.isclose(opt.lr, 0.0)


def test_setup_and_step_alias():
    p = Variable(np.array([1.0]))
    p.grad = np.array([1.0])
    opt = SGD(lr=0.5)
    opt.setup({"p": p})
    opt.step()
    assert np.allclose(p.data, 0.5)
