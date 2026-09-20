"""Mini-batch data loader."""

from typing import Any, Iterator, Tuple

from dptiny.backend import xp
from dptiny.data.dataset import Dataset, TensorDataset


class DataLoader:
    """Iterate over a dataset in mini-batches.

    ``dataset`` may be a ``Dataset`` instance or a plain ``(X, y)`` tuple of
    arrays (wrapped into a ``TensorDataset`` for backward compatibility).
    Every call to ``__iter__`` produces a fresh shuffle when ``shuffle`` is
    enabled.
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        if isinstance(dataset, tuple):
            dataset = TensorDataset(dataset[0], dataset[1])
        self.dataset: Dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.data_size = len(dataset)

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        if self.shuffle:
            index = xp.random.permutation(self.data_size)
        else:
            index = xp.arange(self.data_size)

        max_iter = self.data_size // self.batch_size
        if not self.drop_last and self.data_size % self.batch_size != 0:
            max_iter += 1

        for i in range(max_iter):
            batch_index = index[i * self.batch_size: (i + 1) * self.batch_size]
            yield self._collate(batch_index)

    def __len__(self) -> int:
        n = self.data_size // self.batch_size
        if not self.drop_last and self.data_size % self.batch_size != 0:
            n += 1
        return n

    def _collate(self, batch_index):
        # Fast path: fancy indexing straight into the backing arrays.
        if isinstance(self.dataset, TensorDataset):
            return (
                self.dataset.x[batch_index],
                self.dataset.t[batch_index],
            )
        batch = [self.dataset[int(i)] for i in batch_index]
        xs, ts = zip(*batch)
        return xp.stack(list(xs)), xp.stack(list(ts))
