"""Entropy/CRH anomaly and correlation experiments on MNIST (Section 6).

Reconstruction of the label-interchange study: five logistic-regression
models are trained on five disjoint 5,000-sample subsets of MNIST; model
i has the labels of k_i in {0, 5, 6, 7, 8} classes cyclically
interchanged in its training data (a confidently *wrong* model keeps low
prediction entropy, so inverse-entropy powers ignore the corruption).
Two variants of the interchange pattern:
  - random: each model draws its own set of classes and its own
    interchange (models disagree on the wrongly learned classes);
  - deterministic: all models of a run share the same interchange
    (models agree on the wrong labels, which defeats CRH).

Outputs (results/mnist_anomaly/anomaly.json):
  - instance: per-model accuracy and entropy/CRH powers of one random
    run (bar figure of the paper);
  - corr: Pearson correlation between model accuracies and their
    entropy/CRH powers, over 20 runs of each variant (boxplot figure).

Run from the repository root (a few minutes on a multicore CPU):
    python3 experiments/mnist_anomaly.py [--workers 8]
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mnist_main import load_mnist, train_teacher, softmax_np  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results",
                   "mnist_anomaly")
SEED = 20260715
SUBSET = 5000
K_LIST = [0, 5, 6, 7, 8]
RUNS = 20

_DATA = {}


def interchange(y, order, k):
    """Cyclically interchange the labels of the first k classes of
    `order`: order[j] -> order[(j+1) % k]."""
    if k == 0:
        return y
    out = y.copy()
    for j in range(k):
        out[y == order[j]] = order[(j + 1) % k]
    return out


def one_run(args):
    mode, run_idx = args
    from vpe.voting_utils import WeightFinding

    (X_train, y_train), (X_val, y_val), _ = _DATA["d"]
    rng = np.random.default_rng(
        (SEED, {"random": 0, "deterministic": 1}[mode], run_idx))
    perm = rng.permutation(len(X_train))
    shared_order = rng.permutation(10)

    n = len(K_LIST)
    val_preds = np.empty((n, len(X_val), 10))
    for i, k in enumerate(K_LIST):
        idx = perm[i * SUBSET:(i + 1) * SUBSET]
        order = shared_order if mode == "deterministic" \
            else rng.permutation(10)
        yi = interchange(y_train[idx], order, k)
        w, b = train_teacher(X_train[idx], yi,
                             int(rng.integers(0, 2 ** 31)))
        val_preds[i] = softmax_np(X_val @ w.T + b)

    wf = WeightFinding.from_predictions(val_preds, y_val)
    acc = 100.0 * np.mean(np.argmax(val_preds, axis=2) == y_val, axis=1)
    powers = {"entropy": wf.entropy_weights(), "crh": wf.crh_pytorch()}
    return {"mode": mode, "run": run_idx,
            "accuracy": acc.tolist(),
            "powers": {m: np.asarray(p, float).tolist()
                       for m, p in powers.items()},
            "corr": {m: float(np.corrcoef(acc, p)[0, 1])
                     for m, p in powers.items()}}


def init_worker():
    _DATA["d"] = load_mnist()


def main(workers):
    os.makedirs(OUT, exist_ok=True)
    jobs = [(mode, r) for mode in ["random", "deterministic"]
            for r in range(RUNS)]
    with Pool(workers, initializer=init_worker) as pool:
        runs = pool.map(one_run, jobs)

    instance = next(r for r in runs
                    if r["mode"] == "random" and r["run"] == 0)
    results = {
        "k_list": K_LIST, "runs": RUNS,
        "instance": {"k": K_LIST, "accuracy": instance["accuracy"],
                     "entropy_w": instance["powers"]["entropy"],
                     "crh_w": instance["powers"]["crh"]},
        "corr": {mode: {m: [r["corr"][m] for r in runs
                            if r["mode"] == mode]
                        for m in ["entropy", "crh"]}
                 for mode in ["random", "deterministic"]},
    }
    with open(os.path.join(OUT, "anomaly.json"), "w") as f:
        json.dump(results, f)
    print("saved", os.path.join(OUT, "anomaly.json"))
    for mode in ["random", "deterministic"]:
        for m in ["entropy", "crh"]:
            c = results["corr"][mode][m]
            print(f"corr({mode:13s}, {m:7s}): median {np.median(c):+.3f}"
                  f" range [{min(c):+.3f}, {max(c):+.3f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args().workers)
