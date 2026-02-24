import numpy as np
from dptiny.core import Variable
from dptiny.core import Config
from dptiny.core import no_grad
from dptiny.core import as_array, as_variable
from dptiny.functions import (
    Function,
    add,
    mul,
    neg,
    sub,
    div,
    rsub,
    rdiv,
    sigmoid,
    relu,
    softmax,
    softmax_cross_entropy,
    matmul,
    pow,
)
from dptiny.layers import Linear, MLP, SGD, Adam
from dptiny.datasets import get_mnist, DataLoader

__version__ = "0.1.0"
