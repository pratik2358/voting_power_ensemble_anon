"""Differentially private weighted voting on the CINIC-10 ensembles.

Pure post-processing on the prediction tensors already materialized by
notebooks/cinic/cinic_models_weights*.ipynb: no model re-inference is
needed. Point --wf-val / --wf-test to pickled WeightFinding objects (the
wf_*.pkl files under weight_finders*/), holding the per-model probability
vectors on the validation (auxiliary) and test sets respectively.

Caveat: the CINIC teachers of the paper's main experiments are trained on
overlapping data distributions but disjoint image sets, so the
add/remove-one-example adjacency of Section 5 applies to them directly.

Run from the repository root:
    python3 experiments/cinic_dp.py --wf-val weight_finders/wf_fl_ci.pkl \
        --wf-test <pickle with test-set predictions>
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vpe.voting_utils import WeightFinding, voting
from vpe import dp_mechanisms as dpm
from vpe import fast_values

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "dp", "cinic")
SEED = 20260713


def main(wf_val_path, wf_test_path):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    wf_val = WeightFinding.load(wf_val_path)
    wf_test = WeightFinding.load(wf_test_path)
    val_preds, test_preds = wf_val.y_preds, wf_test.y_preds
    y_te = wf_test.labels
    n_models = len(val_preds)

    votes_val = np.argmax(val_preds, axis=2)
    weights = {
        "equal": np.ones(n_models),
        "accuracy": wf_val.accuracy_weights(),
        "entropy": wf_val.entropy_weights(),
        "entropy_conf": np.clip(
            1 - dpm.mean_entropies(val_preds.astype(np.float64))
            / np.log(val_preds.shape[2]), 0, 1),
        "crh": wf_val.crh_pytorch(),
        "regression": wf_val.regression_pytorch(num_iterations=5000),
        "loo": fast_values.loo(votes_val, wf_val.labels, 10),
    }
    if n_models <= 20:
        weights["shapley"] = fast_values.shapley(votes_val, wf_val.labels,
                                                 10)

    clip = {"accuracy": (0, 1), "entropy": (0, 5), "entropy_conf": (0, 1),
            "crh": (0, 1), "regression": (0, 1), "shapley": (0, 1),
            "loo": (0, 0.1)}
    joint = {"crh", "regression", "shapley", "loo"}
    eps_grid = np.round(np.logspace(np.log10(0.05), np.log10(10), 25), 4)
    reps, eps_w = 50, 5.0
    results = {"eps_grid": eps_grid.tolist(), "reps": reps, "eps_w": eps_w,
               "nonprivate": {s: voting(test_preds, y_te,
                                        dpm.normalize_weights(np.asarray(w, float), "max"),
                                        "plurality")
                              for s, w in weights.items()}}

    mechs = {"vote_count": dpm.noisy_vote_count,
             "output_pert": dpm.output_perturbation,
             "randomized_response": dpm.randomized_response}
    results["D1"] = {}
    eq = np.ones(n_models)
    for name, mech in mechs.items():
        accs = [dpm.accuracy(mech(test_preds, eq, e, rng, reps=reps), y_te)
                for e in eps_grid]
        results["D1"][name] = {"mean": [float(np.mean(a)) for a in accs],
                               "std": [float(np.std(a)) for a in accs]}
        print("D1", name, "done")

    results["D2"] = {"released_weights": {}, "curves": {}}
    for s, w in weights.items():
        if s == "equal":
            w_rel = eq
        else:
            w_rel = dpm.normalize_weights(dpm.release_weights(
                np.asarray(w, float), eps_w, rng, clip=clip[s],
                joint=s in joint), "max")
        results["D2"]["released_weights"][s] = w_rel.tolist()
        accs = [dpm.accuracy(dpm.noisy_vote_count(
            test_preds, w_rel, e, rng, reps=reps), y_te) for e in eps_grid]
        results["D2"]["curves"][f"{s}|vote_count"] = {
            "mean": [float(np.mean(a)) for a in accs],
            "std": [float(np.std(a)) for a in accs]}
        print("D2", s, "done")

    json.dump(results, open(os.path.join(OUT, "dp_results.json"), "w"))
    print("saved", os.path.join(OUT, "dp_results.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf-val", required=True,
                    help="pickled WeightFinding with validation predictions")
    ap.add_argument("--wf-test", required=True,
                    help="pickled WeightFinding with test predictions")
    args = ap.parse_args()
    main(args.wf_val, args.wf_test)
