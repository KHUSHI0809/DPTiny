"""Dataset abstractions for DPTiny."""

from typing import Any, Callable, Optional, Tuple


class Dataset:
    """Base dataset class.

    Subclasses implement ``__getitem__`` returning a single ``(x, t)`` pair
    and ``__len__``.  ``transform`` and ``target_transform`` are applied to
    each sample automatically by :meth:`_apply_transforms`.
    """

    def __init__(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.transform = transform
        self.target_transform = target_transform

    def _apply_transforms(self, x: Any, t: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            t = self.target_transform(t)
        return x, t

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class TensorDataset(Dataset):
    """Dataset wrapping a pair of ``(X, t)`` arrays aligned on axis 0."""

    def __init__(self, x: Any, t: Any, **kwargs):
        super().__init__(**kwargs)
        assert len(x) == len(t)
        self.x = x
        self.t = t

    def __getitem__(self, index):
        return self._apply_transforms(self.x[index], self.t[index])

    def __len__(self):
        return len(self.x)
