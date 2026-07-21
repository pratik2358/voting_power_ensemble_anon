"""Main (non-private) phishing voting-power experiments of the paper,
reconstructed (the ensemble notebook that produced the archived
results was not part of the released repository) and run with the
corrected code.

Setting, following the paper and the surviving single-model notebook
(notebooks/phishing/webpage_class.ipynb): the Kaggle web-page phishing
dataset (11,430 pages, 87 features, binary legitimate-vs-phishing
target), split 60/20/20 into train/validation/test (stratified,
disjoint), features MinMax-scaled. For each repetition, the training
split is partitioned into n *disjoint* subsets; model M_i is a
three-layer MLP (300 and 100 hidden units, batch norm, dropout 0.1,
sigmoid output; BCE loss, Adam, learning rate 1e-3, batch size 64, 50
epochs) trained on subset i whose labels are flipped at rate
i*(1/n - 0.02). Voting powers are computed on the validation split
(Shapley and LOO with the plurality value function, exact Gray-code
computation), evaluation with Borda and plurality on the test split.
Ensembles of 5, 10, and 15 models, 20 repetitions each.

Run from the repository root:
    python3 experiments/phishing_main.py [--workers 6]
Results: results/phishing_main/accuracy_dict_<n>.json
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

OUT = os.path.join(os.path.dirname(__file__), "..", "results",
                   "phishing_main")
CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                   "dataset_phishing.csv")
SEED = 20260714
SIZES = [5, 10, 15]
REPS = 20
EPOCHS = 50

_DATA = {}


def load_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    df = pd.read_csv(CSV)
    y = pd.get_dummies(df["status"])["legitimate"].astype("int").values
    X = df.drop(columns=["url", "status"]).values.astype(np.float32)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
    scaler = MinMaxScaler().fit(X_tr)
    return ((scaler.transform(X_tr).astype(np.float32), y_tr),
            (scaler.transform(X_val).astype(np.float32), y_val),
            (scaler.transform(X_te).astype(np.float32), y_te))


def build_model(n_features):
    import torch.nn as nn

    class ChurnModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(n_features, 300)
            self.layer_2 = nn.Linear(300, 100)
            self.layer_out = nn.Linear(100, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(p=0.1)
            self.batchnorm1 = nn.BatchNorm1d(300)
            self.batchnorm2 = nn.BatchNorm1d(100)

        def forward(self, x):
            x = self.batchnorm1(self.relu(self.layer_1(x)))
            x = self.batchnorm2(self.relu(self.layer_2(x)))
            return self.sigmoid(self.layer_out(self.dropout(x)))

    return ChurnModel()


def train_teacher(X, y, seed):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = build_model(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss()
    Xt = torch.tensor(X)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Xt), generator=gen)
        for s in range(0, len(Xt), 64):
            idx = perm[s:s + 64]
            if len(idx) < 2:      # batch norm needs more than one sample
                continue
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    return model


def probs(model, X):
    """(n_samples, 2) probability vectors [phishing, legitimate]."""
    import torch
    with torch.no_grad():
        p = model(torch.tensor(X)).numpy().ravel()
    return np.stack([1 - p, p], axis=1)


def one_run(args):
    n_models, run_idx = args
    from vpe.voting_utils import WeightFinding, voting
    from vpe import fast_values

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = _DATA["d"]
    rng = np.random.default_rng((SEED, n_models, run_idx))
    perm = rng.permutation(len(X_tr))
    part = len(X_tr) // n_models
    p = 1 / n_models - 0.02

    val_preds = np.empty((n_models, len(X_val), 2))
    test_preds = np.empty((n_models, len(X_te), 2))
    for i in range(n_models):
        idx = perm[i * part:(i + 1) * part]
        yi = y_tr[idx].copy()
        n_flip = int(i * p * len(yi))
        fidx = rng.choice(len(yi), n_flip, replace=False)
        yi[fidx] = 1 - yi[fidx]
        model = train_teacher(X_tr[idx], yi,
                              int(rng.integers(0, 2 ** 31)))
        val_preds[i] = probs(model, X_val)
        test_preds[i] = probs(model, X_te)

    wf = WeightFinding.from_predictions(val_preds, y_val)
    votes_val = np.argmax(val_preds, axis=2)
    weights = {
        "unweighted": np.ones(n_models),
        "accuracy": wf.accuracy_weights(),
        "entropy": wf.entropy_weights(),
        "crh": wf.crh_pytorch(),
        "regression": wf.regression_pytorch(num_iterations=5000),
        "shapley": fast_values.shapley(votes_val, y_val, 2,
                                       empty_value=0.5),
        "loo": fast_values.loo(votes_val, y_val, 2),
    }
    return {m: {rule: float(voting(test_preds, y_te,
                                   np.asarray(w, float), rule))
                for rule in ["borda", "plurality"]}
            for m, w in weights.items()}


def init_worker():
    _DATA["d"] = load_data()


def main(workers):
    os.makedirs(OUT, exist_ok=True)
    jobs = [(n, r) for n in SIZES for r in range(REPS)]
    results = {}
    with Pool(workers, initializer=init_worker) as pool:
        for (n, r), res in zip(jobs, pool.imap(one_run, jobs)):
            results.setdefault(n, []).append(res)
            if len(results[n]) == REPS:
                agg = {m: {rule: [run[m][rule] for run in results[n]]
                           for rule in ["borda", "plurality"]}
                       for m in results[n][0]}
                with open(os.path.join(
                        OUT, f"accuracy_dict_{n}.json"), "w") as f:
                    json.dump(agg, f)
                print(f"size {n}: {REPS} runs saved", flush=True)
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    main(args.workers)
