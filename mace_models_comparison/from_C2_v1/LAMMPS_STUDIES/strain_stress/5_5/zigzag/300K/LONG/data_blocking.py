import numpy as np


def compute_blocks(data: np.ndarray, block_size: int):
    N = len(data)

    discard = N % block_size
    if discard != 0:
        print("WARNING: discarding first", discard, "data")

    data = data[discard:]

    nblocks = N // block_size

    # Reshape instead of manual slicing (cleaner and faster)
    blocks = data.reshape(nblocks, block_size)

    return np.mean(blocks, axis=1)


def meanstd(means):
    return np.mean(means), np.std(means, ddof=1) / np.sqrt(len(means))
