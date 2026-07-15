"""Main (non-private) DMOZ voting-power experiments of the paper,
rerun with the corrected code (no extra softmax on predict_proba, no
double softmax in the entropy weights).

Faithful to notebooks/dmoz/dmoz.ipynb: stratified 70/15/15
train/test/validation split; for each repetition, each of the n
Multinomial Naive Bayes teachers is trained on its own stratified 80%
subsample of the training split with complement label flipping at rate
i*(1/n - 0.02); all voting powers computed on the validation split;
evaluation with Borda and plurality on the test split. Ensembles of
3, 5, 7, and 10 models, 20 repetitions each.

Run from the repository root (takes a few hours):
    python3 experiments/dmoz_main.py
Results: results/dmoz_main/accuracy_dict_<n>.json
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vpe.voting_utils import WeightFinding_sklearn, voting
from dmoz_dp import build_data     # same preprocessing as the paper

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "dmoz_main")
SEED = 20260714
SIZES = [3, 5, 7, 10]
REPS = 20


def stratified_subsample(y, frac, rng):
    idx = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        idx.append(rng.choice(cls, int(frac * len(cls)), replace=False))
    return np.concatenate(idx)


def one_rep(X_tr, y_tr, X_val, y_val, X_te, y_te, n_models, rng):
    from sklearn.naive_bayes import MultinomialNB
    n_classes = len(np.unique(y_tr))
    p = 1 / n_models - 0.02
    models = []
    for i in range(n_models):
        idx = stratified_subsample(y_tr, 0.8, rng)
        yi = y_tr[idx].copy()
        n_flip = int(i * p * len(yi))
        fidx = rng.choice(len(yi), n_flip, replace=False)
        yi[fidx] = n_classes - 1 - yi[fidx]
        m = MultinomialNB()
        m.fit(X_tr[idx], yi)
        models.append(m)

    wf = WeightFinding_sklearn(models, (X_val, y_val), "cpu")
    wf_test = WeightFinding_sklearn(models, (X_te, y_te), "cpu")
    weights = {
        "unweighted": np.ones(n_models),
        "accuracy": wf.accuracy_weights(),
        "entropy": wf.entropy_weights(),
        "crh": wf.crh(),
        "regression": wf.regression(),
        "shapley": wf.shapley(method="plurality"),
        "loo": wf.loo(method="plurality"),
    }
    return {m: {rule: float(voting(wf_test.y_preds, wf_test.labels,
                                   np.asarray(w, float), rule))
                for rule in ["borda", "plurality"]}
            for m, w in weights.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    csv = os.path.join(os.path.dirname(__file__), "..", "data",
                       "URL Classification.csv")
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = build_data(csv, rng)
    for n in SIZES:
        runs = []
        for r in range(REPS):
            runs.append(one_rep(X_tr, y_tr, X_val, y_val, X_te, y_te,
                                n, rng))
            print(f"n={n}: rep {r + 1}/{REPS} done")
        agg = {m: {rule: [run[m][rule] for run in runs]
                   for rule in ["borda", "plurality"]}
               for m in runs[0]}
        with open(os.path.join(OUT, f"accuracy_dict_{n}.json"), "w") as f:
            json.dump(agg, f)
        print(f"saved size {n}")


if __name__ == "__main__":
    main()
