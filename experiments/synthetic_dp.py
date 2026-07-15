"""Synthetic comparison of basic DP mechanisms for (unweighted) voting.

Reconstruction of the simulation behind the "basic DP comparison" figure
of the original submission: V voters with softmax preference vectors
sampled uniformly from [-10, 10]^C, ground truth = noiseless majority
vote, and three mechanisms (vote-count perturbation, output perturbation,
randomized response) evaluated for eps in [0.1, 10]. Kept as a sanity
check; the paper's Section 5 experiments now run on real ensembles
(see mnist_dp.py).

Run: python3 experiments/synthetic_dp.py  ->  results/dp/synthetic.json
"""

import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "dp")
SEED = 20260713
V = 20          # voters
C = 10          # classes
N = 1_000_000   # vote entries in total -> N // V instances


def main():
    rng = np.random.default_rng(SEED)
    n_inst = N // V
    x = rng.uniform(-10, 10, (V, n_inst, C))
    y = np.exp(x) / np.exp(x).sum(axis=2, keepdims=True)   # softmax
    votes = np.argmax(y, axis=2)                            # (V, n_inst)
    counts = np.zeros((n_inst, C))
    np.add.at(counts, (np.arange(n_inst)[None, :].repeat(V, 0), votes), 1)
    truth = np.argmax(counts, axis=1)

    eps_grid = np.round(np.arange(0.1, 10.05, 0.1), 2)
    res = {"eps": eps_grid.tolist(), "V": V, "C": C, "instances": n_inst}

    acc_count, acc_output, acc_rr = [], [], []
    for eps in eps_grid:
        # vote-count perturbation
        noisy = counts + rng.laplace(0, 2 / eps, counts.shape)
        acc_count.append(float(np.mean(np.argmax(noisy, 1) == truth)))
        # output perturbation on the probability vectors
        noisy_y = y + rng.laplace(0, 1 / eps, y.shape)
        acc_output.append(float(np.mean(
            np.argmax(noisy_y.sum(axis=0), 1) == truth)))
        # randomized response on the hard votes
        p_keep = np.exp(eps) / (np.exp(eps) + C - 1)
        flip = rng.random(votes.shape) > p_keep
        rand_votes = np.where(
            flip, (votes + rng.integers(1, C, votes.shape)) % C, votes)
        rc = np.zeros((n_inst, C))
        np.add.at(rc, (np.arange(n_inst)[None, :].repeat(V, 0), rand_votes), 1)
        acc_rr.append(float(np.mean(np.argmax(rc, 1) == truth)))

    res["vote_count"] = acc_count
    res["output_pert"] = acc_output
    res["randomized_response"] = acc_rr
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "synthetic.json"), "w") as f:
        json.dump(res, f)
    print("saved", os.path.join(OUT, "synthetic.json"))


if __name__ == "__main__":
    main()
