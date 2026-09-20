import numpy as np

from dptiny.data import DataLoader, Dataset, TensorDataset
from dptiny.data.transforms import Compose, Flatten, Normalize, ToFloat


def test_batch_count():
    x = np.random.randn(100, 4)
    t = np.arange(100)
    loader = DataLoader((x, t), batch_size=32, shuffle=False)
    batches = list(loader)
    assert len(batches) == 4  # 100 = 3*32 + 4
    assert batches[-1][0].shape[0] == 4


def test_drop_last():
    x = np.random.randn(100, 4)
    t = np.arange(100)
    loader = DataLoader((x, t), batch_size=32, shuffle=False, drop_last=True)
    batches = list(loader)
    assert len(batches) == 3


def test_tuple_compat_and_fancy_index():
    x = np.arange(20).reshape(10, 2)
    t = np.arange(10)
    loader = DataLoader((x, t), batch_size=5, shuffle=False)
    bx, bt = next(iter(loader))
    assert np.array_equal(bx, x[:5])
    assert np.array_equal(bt, t[:5])


def test_shuffle_reproducible():
    x = np.arange(20).reshape(10, 2)
    t = np.arange(10)
    np.random.seed(0)
    first = list(DataLoader((x, t), batch_size=5, shuffle=True))
    np.random.seed(0)
    second = list(DataLoader((x, t), batch_size=5, shuffle=True))
    for (a, _), (b, _) in zip(first, second):
        assert np.array_equal(a, b)


def test_reshuffles_each_iter():
    x = np.arange(200).reshape(100, 2)
    t = np.arange(100)
    loader = DataLoader((x, t), batch_size=10, shuffle=True)
    np.random.seed(1)
    order1 = np.concatenate([b[1] for b in loader])
    np.random.seed(2)
    order2 = np.concatenate([b[1] for b in loader])
    assert not np.array_equal(order1, order2)
    assert sorted(order1.tolist()) == list(range(100))


def test_tensor_dataset():
    x = np.random.randn(10, 4)
    t = np.arange(10)
    ds = TensorDataset(x, t)
    assert len(ds) == 10
    xi, ti = ds[3]
    assert np.array_equal(xi, x[3])
    assert ti == t[3]


def test_dataset_with_transform():
    class D(Dataset):
        def __getitem__(self, i):
            return self._apply_transforms(i, i)

        def __len__(self):
            return 5

    ds = D(transform=lambda v: v * 2)
    assert ds[3] == (6, 3)


def test_transforms():
    t = Compose([ToFloat(), Normalize(0.5, 0.5), Flatten()])
    out = t(np.ones((2, 2)))
    assert out.shape == (4,)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.0)


def test_generic_dataset_loader():
    class D(Dataset):
        def __init__(self):
            super().__init__()
            self.x = np.arange(12).reshape(6, 2)
            self.t = np.arange(6)

        def __getitem__(self, i):
            return self.x[i], self.t[i]

        def __len__(self):
            return 6

    loader = DataLoader(D(), batch_size=4, shuffle=False)
    bx, bt = next(iter(loader))
    assert bx.shape == (4, 2)
    assert bt.shape == (4,)
