"""CINIC-10 main-experiment evaluation from the prediction pickles.

CPU-only post-processing of the WeightFinding pickles produced by
cinic_predictions.py: for each run and noise regime, computes all
voting powers on the validation predictions (with the corrected code)
and evaluates weighted Borda and plurality voting over 20 repetitions,
each on a random 10,000-image subset of each of the cifar and imagenet
test halves (as in notebooks/cinic/voting_*_repeat.ipynb; subsetting
the stored prediction tensor is equivalent to re-running the models on
the subset). Results are written in the format used by the analysis
scripts:
    results/cinic_main/{fl,di}_accuracy_dict_ci{,_2,_3}.json

Run from the directory containing the weight_finders_run*/ folders
(notebooks/cinic/ when produced by the slurm scripts):
    python3 ../../experiments/cinic_eval.py [--runs 1 2 3]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vpe.voting_utils import WeightFinding, voting
from vpe import fast_values

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "cinic_main")
SEED = 20260714
REPS = 20
SUBSET = 10000       # per test half, as in the voting notebooks
SUFFIX = {1: "", 2: "_2", 3: "_3"}


def compute_weights(wf_val):
    votes = np.argmax(wf_val.y_preds, axis=2)
    n = len(wf_val.y_preds)
    weights = {
        "unweighted": np.ones(n),
        "accuracy": wf_val.accuracy_weights(),
        "entropy": wf_val.entropy_weights(),
        "crh": wf_val.crh_pytorch(),
        "regression": wf_val.regression_pytorch(num_iterations=5000),
        "loo": fast_values.loo(votes, wf_val.labels, 10),
    }
    if n <= 20:
        weights["shapley"] = fast_values.shapley(votes, wf_val.labels, 10)
    return weights


def main(runs):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for run in runs:
        for regime in ["fl", "di"]:
            base = f"weight_finders_run{run}/wf_{regime}"
            if not os.path.exists(base + "_val.pkl"):
                print("missing", base + "_val.pkl", "- skipping")
                continue
            wf_val = WeightFinding.load(base + "_val.pkl")
            wf_test = WeightFinding.load(base + "_test.pkl")
            weights = compute_weights(wf_val)
            # the two test halves are concatenated in the stored tensor:
            # cifar first, then imagenet (loader order in predictions)
            n_test = wf_test.y_preds.shape[1]
            half = n_test // 2
            acc = {m: {"borda": [], "plurality": []} for m in weights}
            for _ in range(REPS):
                idx = np.concatenate([
                    rng.choice(half, min(SUBSET, half), replace=False),
                    half + rng.choice(n_test - half,
                                      min(SUBSET, n_test - half),
                                      replace=False)])
                yp = wf_test.y_preds[:, idx]
                yl = wf_test.labels[idx]
                for m, w in weights.items():
                    for rule in ["borda", "plurality"]:
                        acc[m][rule].append(float(voting(
                            yp, yl, np.asarray(w, float), rule)))
            path = os.path.join(
                OUT, f"{regime}_accuracy_dict_ci{SUFFIX[run]}.json")
            json.dump(acc, open(path, "w"))
            print("saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    main(args.runs)
