"""Fast exact Shapley values and LOO for voting ensembles.

Enumerates the value function over all 2^n coalitions with a Gray-code
walk (each step adds or removes a single model from the running vote
counts), then computes exact Shapley values by a vectorized sweep over
coalition masks. Equivalent to WeightFinding.shapley_pytorch /
loo_pytorch (plurality value function, empty-coalition value 0.1), but
usable up to n around 20 models.
"""

from math import comb

import numpy as np


def coalition_values(votes, labels, n_classes, empty_value=0.1):
    """Plurality-accuracy value v(S) for every coalition S.

    votes: (n_models, n_samples) hard votes; labels: (n_samples,).
    Returns an array of size 2^n_models indexed by coalition bitmask.
    Ties are broken toward the smallest class index, as in
    scipy.stats.mode used by WeightFinding.
    """
    n, S = votes.shape
    values = np.empty(2 ** n)
    values[0] = empty_value
    counts = np.zeros((S, n_classes), dtype=np.int32)
    rows = np.arange(S)
    prev = 0
    for k in range(1, 2 ** n):
        gray = k ^ (k >> 1)
        bit = gray ^ prev
        i = bit.bit_length() - 1
        sign = 1 if gray & bit else -1
        np.add.at(counts, (rows, votes[i]), sign)
        pred = np.argmax(counts, axis=1)
        values[gray] = np.mean(pred == labels)
        prev = gray
    return values


def shapley_from_values(values, n):
    """Exact Shapley values from the full coalition-value table."""
    masks = np.arange(2 ** n, dtype=np.int64)
    sizes = np.zeros(2 ** n, dtype=np.int64)
    for i in range(n):
        sizes += (masks >> i) & 1
    coeff = np.array([1.0 / (n * comb(n - 1, r)) for r in range(n)])
    shap = np.zeros(n)
    for i in range(n):
        bit = 1 << i
        without = masks[(masks & bit) == 0]
        r = sizes[without]
        shap[i] = np.sum(coeff[r] * (values[without | bit] - values[without]))
    return shap


def shapley(votes, labels, n_classes, empty_value=0.1):
    """Exact plurality-value Shapley values, Gray-code enumeration."""
    values = coalition_values(votes, labels, n_classes, empty_value)
    return shapley_from_values(values, votes.shape[0])


def loo(votes, labels, n_classes):
    """Leave-one-out values v(N) - v(N \\ {i}) for the plurality value."""
    n, S = votes.shape
    counts = np.zeros((S, n_classes), dtype=np.int32)
    rows = np.arange(S)
    for i in range(n):
        np.add.at(counts, (rows, votes[i]), 1)
    total = np.mean(np.argmax(counts, axis=1) == labels)
    out = np.empty(n)
    for i in range(n):
        np.add.at(counts, (rows, votes[i]), -1)
        out[i] = total - np.mean(np.argmax(counts, axis=1) == labels)
        np.add.at(counts, (rows, votes[i]), 1)
    return out
