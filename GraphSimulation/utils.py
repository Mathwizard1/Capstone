import numpy.random as random

from torch import (
    cuda,
    device,
    float32
)

SEED = 42
RND_GEN = random.default_rng(SEED)

LARGE_NUMBER = 1e9
EPS = 1e-8

cuda_is_available = cuda.is_available()
DEVICE = device('cuda' if cuda_is_available else 'cpu')

DTYPE = float32