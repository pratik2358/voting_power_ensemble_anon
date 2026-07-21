"""Main (non-private) MNIST voting-power experiments of the paper.

Reconstruction of the experiments behind the MNIST line plots and box
plots: multinomial logistic regression models trained on random
(possibly overlapping) 5,000-sample subsets of MNIST with label-flipping
noise drawn uniformly in [0,1], ensembles of 3-16 models (75 runs each:
15 label-flip assignments x 5 experiments) plus larger ensembles for the
scalability study (15 runs each, without Shapley values, whose exact
computation is infeasible there). Voting powers: equal, accuracy,
inverse entropy, CRH, regression, Shapley (exact, Gray-code
implementation from vpe.fast_values), LOO; evaluation with both Borda
and plurality voting on the 10,000-sample test set.

The value function of Shapley/LOO is the accuracy of equal-power
plurality voting on the auxiliary set, as defined in the paper.

Run from the repository root (takes a few hours on a multicore CPU):
    python3 experiments/mnist_main.py [--workers 8]
Results: results/mnist_main/size_<n>.json
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "mnist_main")
SEED = 20260713
SUBSET = 5000
VALID_SIZE = 10000
EPOCHS = 10
DETAIL_SIZES = list(range(3, 17))
LARGE_SIZES = [25, 50, 100, 150, 200]
DETAIL_RUNS = 75          # 15 flip assignments x 5 experiments
LARGE_RUNS = 15
METHODS = ["unweighted", "accuracy", "regression", "shapley", "loo",
           "crh", "entropy"]

_DATA = {}


def load_mnist():
    from sklearn.datasets import fetch_openml
    data_home = os.path.join(os.path.dirname(__file__), "..", "data")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                        as_frame=False, data_home=data_home)
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)
    X_train, y_train = X[:50000], y[:50000]
    X_val, y_val = X[50000:60000], y[50000:60000]
    X_test, y_test = X[60000:], y[60000:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_teacher(X, y, seed):
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(seed)
    model = nn.Linear(784, 10)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(X), torch.tensor(y)
    n = len(Xt)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, 256):
            idx = perm[start:start + 256]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    import torch.nn.functional as F
    with torch.no_grad():
        w = model.weight.numpy().copy()
        b = model.bias.numpy().copy()
    return w, b


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def one_run(args):
    n_models, run_idx, with_shapley = args
    from vpe.voting_utils import WeightFinding, voting
    from vpe import fast_values

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = _DATA["d"]
    # 15 flip assignments, each used in 5 experiments
    assignment = run_idx // 5
    rng_flip = np.random.default_rng((SEED, n_models, assignment))
    flips = rng_flip.uniform(0, 1, n_models)
    rng = np.random.default_rng((SEED, n_models, run_idx, 1))

    val_preds = np.empty((n_models, len(X_val), 10))
    test_preds = np.empty((n_models, len(X_test), 10))
    for i in range(n_models):
        idx = rng.choice(len(X_train), SUBSET, replace=False)
        yi = y_train[idx].copy()
        n_flip = int(flips[i] * SUBSET)
        fidx = rng.choice(SUBSET, n_flip, replace=False)
        yi[fidx] = (yi[fidx] + rng.integers(1, 10, n_flip)) % 10
        w, b = train_teacher(X_train[idx], yi,
                             int(rng.integers(0, 2 ** 31)))
        val_preds[i] = softmax_np(X_val @ w.T + b)
        test_preds[i] = softmax_np(X_test @ w.T + b)

    wf = WeightFinding.from_predictions(val_preds, y_val)
    votes_val = np.argmax(val_preds, axis=2)
    weights = {
        "unweighted": np.ones(n_models),
        "accuracy": wf.accuracy_weights(),
        "entropy": wf.entropy_weights(),
        "crh": wf.crh_pytorch(),
        "regression": wf.regression_pytorch(num_iterations=5000),
        "loo": fast_values.loo(votes_val, y_val, 10),
    }
    if with_shapley:
        weights["shapley"] = fast_values.shapley(votes_val, y_val, 10)

    out = {}
    for m, w in weights.items():
        out[m] = {rule: float(voting(test_preds, y_test,
                                     np.asarray(w, float), rule))
                  for rule in ["borda", "plurality"]}
    return out


def init_worker():
    _DATA["d"] = load_mnist()


def main(workers):
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for n in DETAIL_SIZES:
        jobs += [(n, r, True) for r in range(DETAIL_RUNS)]
    for n in LARGE_SIZES:
        jobs += [(n, r, False) for r in range(LARGE_RUNS)]
    expected = {n: DETAIL_RUNS for n in DETAIL_SIZES}
    expected.update({n: LARGE_RUNS for n in LARGE_SIZES})

    def flush(n, runs):
        agg = {m: {rule: [run[m][rule] for run in runs if m in run]
                   for rule in ["borda", "plurality"]}
               for m in METHODS}
        with open(os.path.join(OUT, f"size_{n}.json"), "w") as f:
            json.dump(agg, f)
        print(f"size {n}: {len(runs)} runs saved")

    results = {}
    with Pool(workers, initializer=init_worker) as pool:
        for (n, r, _), res in zip(jobs, pool.imap(one_run, jobs)):
            results.setdefault(n, []).append(res)
            if len(results[n]) == expected[n]:
                flush(n, results.pop(n))
    for n, runs in results.items():   # safety: flush any leftovers
        flush(n, runs)
    print("saved per-size results to", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    main(args.workers)
