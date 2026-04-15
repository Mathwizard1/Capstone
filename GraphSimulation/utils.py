import numpy.random as random

SEED = 42
RND_GEN = random.default_rng(SEED)

from torch import (
    cuda,
    device,
    float32
)

cuda_is_available = cuda.is_available()
DEVICE = device('cuda' if cuda_is_available else 'cpu')

DTYPE = float32