"""Running-time comparison of the voting-power methods (Figure 3).

Times the computation of the voting powers (not the training of the
models) for MNIST ensembles of 3-16 logistic-regression teachers, on the
10,000-sample auxiliary set, with the same implementations and settings
as the main experiments (experiments/mnist_main.py; in particular
regression uses 5000 Adam iterations there). Equal power is omitted (no
computation). Every measurement is repeated REPS times; the figure uses
the median.

Timings are single-threaded (torch.set_num_threads(1)) for
reproducibility; the host and CPU model are recorded in the output.

Run from the repository root (well under an hour on any recent CPU; see
slurm/mnist_timing.sbatch for running it on a cluster):
    python3 experiments/mnist_timing.py
Results: results/timing/mnist_times.json
"""

import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mnist_main import load_mnist, train_teacher, softmax_np  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "timing")
SEED = 20260715
SUBSET = 5000
N_MAX = 16
SIZES = list(range(3, N_MAX + 1))
REPS = 5


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main():
    import torch
    torch.set_num_threads(1)
    from vpe.voting_utils import WeightFinding
    from vpe import fast_values

    os.makedirs(OUT, exist_ok=True)
    (X_train, y_train), (X_val, y_val), _ = load_mnist()
    rng = np.random.default_rng(SEED)

    print(f"training {N_MAX} teachers...")
    val_preds = np.empty((N_MAX, len(X_val), 10))
    for i in range(N_MAX):
        idx = rng.choice(len(X_train), SUBSET, replace=False)
        yi = y_train[idx].copy()
        n_flip = int(i / N_MAX * SUBSET)
        fidx = rng.choice(SUBSET, n_flip, replace=False)
        yi[fidx] = (yi[fidx] + rng.integers(1, 10, n_flip)) % 10
        w, b = train_teacher(X_train[idx], yi,
                             int(rng.integers(0, 2 ** 31)))
        val_preds[i] = softmax_np(X_val @ w.T + b)

    def timers(wf, votes):
        return {
            "accuracy": lambda: wf.accuracy_weights(),
            "entropy": lambda: wf.entropy_weights(),
            "crh": lambda: wf.crh_pytorch(),
            "regression": lambda: wf.regression_pytorch(
                num_iterations=5000),
            "loo": lambda: fast_values.loo(votes, y_val, 10),
            "shapley": lambda: fast_values.shapley(votes, y_val, 10),
        }

    results = {m: {} for m in
               ["accuracy", "entropy", "crh", "regression", "loo",
                "shapley"]}
    for n in SIZES:
        wf = WeightFinding.from_predictions(val_preds[:n], y_val)
        votes = np.argmax(val_preds[:n], axis=2)
        for method, fn in timers(wf, votes).items():
            ts = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t0)
            results[method][n] = ts
            print(f"n={n:2d} {method:10s} median"
                  f" {np.median(ts):.4g}s")

    out = {"sizes": SIZES, "reps": REPS,
           "host": platform.node(), "cpu": cpu_model(),
           "torch_threads": 1,
           "times": results,
           "median": {m: [float(np.median(results[m][n])) for n in SIZES]
                      for m in results}}
    with open(os.path.join(OUT, "mnist_times.json"), "w") as f:
        json.dump(out, f)
    print("saved", os.path.join(OUT, "mnist_times.json"))


if __name__ == "__main__":
    main()
