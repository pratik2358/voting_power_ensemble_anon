"""Differentially private weighted voting on real MNIST ensembles.

Reproduces the setting of the paper's MNIST experiments (logistic
regression teachers trained on 5,000-sample subsets with label-flipping
noise), then runs the differential-privacy experiments of Section 5:

  D1  comparison of the three DP mechanisms (noisy vote counts, output
      perturbation, randomized response) on the real ensemble;
  D2  equal vs. learned voting powers under a fixed per-query budget,
      with the voting powers themselves released under DP;
  D3  allocation of a total budget across teachers trained on
      overlapping data (uniform vs. margin-aware vs. anti);
  D4  empirical study of (generalized) Proposition 1: agreement rates,
      margin decomposition, and noise tolerance per weighting scheme.

Stages are cached under results/dp/mnist/. Run from the repository root:
    python3 experiments/mnist_dp.py [--stage all|teachers|weights|dp]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vpe.voting_utils import WeightFinding, voting
from vpe import dp_mechanisms as dpm

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "dp", "mnist")
SEED = 20260713
N_MODELS = 10
SUBSET = 5000          # training samples per teacher (as in the paper)
# per-teacher label-flip rates, spanning [0, 1) as in the paper's MNIST setup
FLIP = [i / N_MODELS for i in range(N_MODELS)]
VALID_SIZE = 10000     # public auxiliary set F (held out from train)
EPOCHS = 15
DEVICE = "cpu"

WEIGHT_SCHEMES = ["equal", "accuracy", "entropy", "entropy_conf", "crh",
                  "regression", "shapley", "loo"]
# public clipping ranges for the one-time DP release of the weights
CLIP = {"accuracy": (0.0, 1.0), "entropy": (0.0, 5.0),
        "entropy_conf": (0.0, 1.0), "crh": (0.0, 1.0),
        "regression": (0.0, 1.0), "shapley": (0.0, 1.0), "loo": (0.0, 0.1)}
# weights computed jointly from all teachers (higher release sensitivity);
# per-teacher weights on public data (accuracy, entropy) need no release
JOINT = {"crh", "regression", "shapley", "loo"}


def load_mnist():
    """MNIST via OpenML, original ordering (60k train + 10k test)."""
    from sklearn.datasets import fetch_openml
    data_home = os.path.join(os.path.dirname(__file__), "..", "data")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                        as_frame=False, data_home=data_home)
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]
    # public auxiliary set F: last VALID_SIZE examples of the training set
    X_val, y_val = X_train[-VALID_SIZE:], y_train[-VALID_SIZE:]
    X_train, y_train = X_train[:-VALID_SIZE], y_train[:-VALID_SIZE]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_teacher(X, y, rng_torch, epochs=EPOCHS):
    """Multinomial logistic regression in PyTorch (single linear layer)."""
    model = nn.Linear(784, 10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.tensor(X)
    yt = torch.tensor(y)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=rng_torch)
        for start in range(0, n, 128):
            idx = perm[start:start + 128]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict_probs(models, X):
    """(n_models, n_samples, n_classes) softmax probabilities."""
    Xt = torch.tensor(X)
    out = []
    with torch.no_grad():
        for m in models:
            out.append(torch.softmax(m(Xt), dim=1).numpy())
    return np.array(out)


def flip_labels(y, frac, rng):
    y = y.copy()
    n_flip = int(frac * len(y))
    idx = rng.choice(len(y), n_flip, replace=False)
    y[idx] = (y[idx] + rng.integers(1, 10, n_flip)) % 10
    return y


def stage_teachers():
    """Train two ensembles: disjoint partitions (D1/D2/D4) and
    overlapping subsets (D3), both with the same label-flip rates."""
    rng = np.random.default_rng(SEED)
    rng_torch = torch.Generator().manual_seed(SEED)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist()

    perm = rng.permutation(len(X_train))
    ensembles = {}
    for kind in ["disjoint", "overlapping"]:
        probs = {}
        models = []
        for i in range(N_MODELS):
            if kind == "disjoint":
                idx = perm[i * SUBSET:(i + 1) * SUBSET]
            else:
                idx = rng.choice(len(X_train), SUBSET, replace=False)
            yi = flip_labels(y_train[idx], FLIP[i], rng)
            models.append(train_teacher(X_train[idx], yi, rng_torch))
            print(f"{kind} teacher {i} (flip {FLIP[i]:.2f}) trained")
        probs["val"] = predict_probs(models, X_val)
        probs["test"] = predict_probs(models, X_test)
        ensembles[kind] = probs
    np.savez_compressed(
        os.path.join(OUT, "predictions.npz"),
        y_val=y_val, y_test=y_test,
        **{f"{k}_{s}": v for k, p in ensembles.items() for s, v in p.items()})
    print("saved", os.path.join(OUT, "predictions.npz"))


def compute_weights(y_preds_val, labels_val):
    """All voting-power schemes on the public auxiliary set F."""
    wf = WeightFinding.from_predictions(y_preds_val, labels_val, DEVICE)
    weights = {
        "equal": np.ones(N_MODELS),
        "accuracy": wf.accuracy_weights(),
        "entropy": wf.entropy_weights(),
        # bounded entropy confidence 1 - H/log(m) in [0,1]: same signal as
        # inverse entropy, but with a public cap (no data-dependent
        # normalization), directly usable in the DP mechanisms
        "entropy_conf": np.clip(
            1 - dpm.mean_entropies(y_preds_val.astype(np.float64))
            / np.log(y_preds_val.shape[2]), 0, 1),
        "crh": wf.crh_pytorch(),
        "regression": wf.regression_pytorch(),
        "shapley": wf.shapley_pytorch(method="plurality"),
        "loo": wf.loo_pytorch(method="plurality"),
    }
    return {k: np.asarray(v, dtype=float) for k, v in weights.items()}


def stage_weights():
    d = np.load(os.path.join(OUT, "predictions.npz"))
    weights = compute_weights(d["disjoint_val"], d["y_val"])
    with open(os.path.join(OUT, "weights.json"), "w") as f:
        json.dump({k: v.tolist() for k, v in weights.items()}, f, indent=1)
    print("saved weights.json:")
    for k, v in weights.items():
        print(f"  {k:11s}", np.round(v, 4))


def released(weights, scheme, eps_w, rng):
    """One-time DP release of a weight vector, normalized to max 1."""
    if scheme == "equal":
        return np.ones(N_MODELS)
    w = dpm.release_weights(weights[scheme], eps_w, rng,
                            clip=CLIP[scheme], joint=scheme in JOINT)
    return dpm.normalize_weights(w, mode="max")


def stage_dp(reps=50, eps_w=5.0):
    d = {k: v.astype(np.float64) if v.ndim == 3 else v
         for k, v in np.load(os.path.join(OUT, "predictions.npz")).items()}
    y_test = d["y_test"]
    weights = {k: np.asarray(v) for k, v in
               json.load(open(os.path.join(OUT, "weights.json"))).items()}
    rng = np.random.default_rng(SEED + 1)
    eps_grid = np.round(np.logspace(np.log10(0.05), np.log10(10), 25), 4)
    results = {"eps_grid": eps_grid.tolist(), "reps": reps, "eps_w": eps_w,
               "flip_rates": FLIP}

    # Reference (non-private) accuracies
    test_preds = d["disjoint_test"]
    results["nonprivate"] = {
        s: {"borda": voting(test_preds, y_test, dpm.normalize_weights(
                weights[s] if s != "equal" else np.ones(N_MODELS), "max"),
                "borda"),
            "plurality": voting(test_preds, y_test, dpm.normalize_weights(
                weights[s] if s != "equal" else np.ones(N_MODELS), "max"),
                "plurality")}
        for s in WEIGHT_SCHEMES}
    results["teacher_accuracy"] = [
        100 * np.mean(np.argmax(test_preds[i], 1) == y_test)
        for i in range(N_MODELS)]

    # --- D1: mechanisms, equal weights --------------------------------
    print("D1: mechanisms on the disjoint-teacher ensemble")
    mechs = {"vote_count": dpm.noisy_vote_count,
             "output_pert": dpm.output_perturbation,
             "randomized_response": dpm.randomized_response}
    eq = np.ones(N_MODELS)
    results["D1"] = {}
    for name, mech in mechs.items():
        accs = [dpm.accuracy(mech(test_preds, eq, e, rng, reps=reps), y_test)
                for e in eps_grid]
        results["D1"][name] = {"mean": [float(np.mean(a)) for a in accs],
                               "std": [float(np.std(a)) for a in accs]}
        print(f"  {name} done")

    # --- D2: weighting schemes under DP (noisy vote count) ------------
    print("D2: weighting schemes under DP")
    results["D2"] = {"released_weights": {}, "curves": {}}
    for s in WEIGHT_SCHEMES:
        w_rel = released(weights, s, eps_w, rng)
        results["D2"]["released_weights"][s] = w_rel.tolist()
        for mech_name, mech in [("vote_count", dpm.noisy_vote_count),
                                ("output_pert", dpm.output_perturbation)]:
            accs = [dpm.accuracy(mech(test_preds, w_rel, e, rng, reps=reps),
                                 y_test) for e in eps_grid]
            results["D2"]["curves"][f"{s}|{mech_name}"] = {
                "mean": [float(np.mean(a)) for a in accs],
                "std": [float(np.std(a)) for a in accs]}
        # exact (non-released) weights, for reference
        w_ex = dpm.normalize_weights(
            weights[s] if s != "equal" else np.ones(N_MODELS), "max")
        accs = [dpm.accuracy(
            dpm.noisy_vote_count(test_preds, w_ex, e, rng, reps=reps), y_test)
            for e in eps_grid]
        results["D2"]["curves"][f"{s}|vote_count_exactw"] = {
            "mean": [float(np.mean(a)) for a in accs],
            "std": [float(np.std(a)) for a in accs]}
        # per-teacher weights on public data (accuracy, entropy confidence):
        # no release needed, but noise calibrated to the public cap
        if s in ("accuracy", "entropy_conf"):
            accs = [dpm.accuracy(dpm.noisy_vote_count(
                test_preds, weights[s], e, rng, reps=reps, cap=CLIP[s][1]),
                y_test) for e in eps_grid]
            results["D2"]["curves"][f"{s}|vote_count_capped"] = {
                "mean": [float(np.mean(a)) for a in accs],
                "std": [float(np.std(a)) for a in accs]}
        print(f"  {s} done")

    # weight-release fidelity vs eps_w
    fid = {}
    for s in [x for x in WEIGHT_SCHEMES if x != "equal"]:
        fid[s] = {}
        for ew in [0.5, 1, 2, 5, 10, 20]:
            rels = [released(weights, s, ew, rng) for _ in range(20)]
            corr = [float(np.corrcoef(r, weights[s])[0, 1]) if np.std(r) > 0
                    else 0.0 for r in rels]
            fid[s][str(ew)] = float(np.mean(corr))
    results["D2"]["release_fidelity"] = fid

    # --- D3: budget allocation, overlapping teachers ------------------
    print("D3: budget allocation across overlapping teachers")
    over_test = d["overlapping_test"]
    over_val = d["overlapping_val"]
    margins = dpm.mean_margins(over_val)
    # released tolerance signal (overlapping data: joint sensitivity)
    tol_rel = np.clip(dpm.release_weights(margins, eps_w, rng, clip=(0, 1),
                                          joint=True), 0.02, 1)
    results["D3"] = {"margins": margins.tolist(),
                     "released_tol": tol_rel.tolist(), "curves": {}}
    eps_tot_grid = np.round(np.logspace(np.log10(0.5), np.log10(100), 20), 4)
    results["D3"]["eps_total_grid"] = eps_tot_grid.tolist()
    # mode names: 'tolerance' puts noise proportional to the margin (the
    # "noise sink" rule: more noise on confident teachers); 'anti' puts
    # budget proportional to the margin (protecting confident teachers)
    variants = [("uniform", "uniform", margins),
                ("noise_prop_margin_oracle", "tolerance", margins),
                ("noise_prop_margin_released", "tolerance", tol_rel),
                ("budget_prop_margin_oracle", "anti", margins),
                ("budget_prop_margin_released", "anti", tol_rel)]
    for name, mode, tol in variants:
        accs = []
        for et in eps_tot_grid:
            alloc = dpm.allocate_budget(et, tol, mode)
            accs.append(dpm.accuracy(
                dpm.noisy_votes(over_test, np.ones(N_MODELS), alloc, rng,
                                reps=reps), y_test))
        results["D3"]["curves"][name] = {
            "mean": [float(np.mean(a)) for a in accs],
            "std": [float(np.std(a)) for a in accs]}
        print(f"  {name} done")

    # --- D4: Proposition 1 empirics ------------------------------------
    print("D4: margin decomposition and noise tolerance")
    results["D4"] = {}
    for s in ["equal", "entropy_conf", "accuracy", "regression"]:
        w = weights[s] if s != "equal" else np.ones(N_MODELS)
        w = dpm.normalize_weights(w, "max")
        dec = dpm.margin_decomposition(test_preds, w)
        ok = dec["Gamma"] >= dec["bound"] - 1e-9
        dis = ~dec["agree"]
        results["D4"][s] = {
            "agree_fraction": float(np.mean(dec["agree"])),
            "bound_holds": float(np.mean(ok)),
            "bound_positive_on_disagree": float(np.mean(dec["bound"][dis] > 0))
            if dis.any() else None,
            "mean_Gamma": float(np.mean(dec["Gamma"])),
            "mean_bound": float(np.mean(dec["bound"])),
            "corr_Gamma_bound_disagree": float(np.corrcoef(
                dec["Gamma"][dis], dec["bound"][dis])[0, 1]) if dis.sum() > 2
            else None,
        }
        sig_grid = np.round(np.logspace(-3, 0.5, 15), 5)
        results["D4"][s]["flip_prob"] = {
            "sigma": sig_grid.tolist(),
            "prob": [float(dpm.flip_probability(test_preds, w, sg, rng,
                                                reps=20)) for sg in sig_grid]}
        # privacy-fair variant: each teacher adds noise of scale sigma to
        # its own scores (same per-teacher budget for every scheme), and
        # the weighted aggregation is post-processing; the aggregate noise
        # then has scale sigma * ||w||_2 with w normalized to sum 1
        w_sum = w / w.sum()
        results["D4"][s]["flip_prob_per_teacher"] = {
            "sigma": sig_grid.tolist(),
            "prob": [float(dpm.flip_probability(
                test_preds, w, sg * np.linalg.norm(w_sum), rng, reps=20))
                for sg in sig_grid]}
    # per-teacher diagnostics
    val_preds = d["disjoint_val"]
    results["D4"]["per_teacher"] = {
        "mean_margin": dpm.mean_margins(val_preds).tolist(),
        "mean_entropy": dpm.mean_entropies(val_preds).tolist(),
        "admissible_sigma": dpm.admissible_sigma(
            dpm.mean_margins(val_preds), 10).tolist(),
        "entropy_weight": weights["entropy"].tolist(),
        "accuracy_weight": weights["accuracy"].tolist(),
    }

    with open(os.path.join(OUT, "dp_results.json"), "w") as f:
        json.dump(results, f)
    print("saved", os.path.join(OUT, "dp_results.json"))


def stage_extras():
    """Two additional analyses for the revision:
    - sensitivity of the voting powers and ensemble accuracy to |F|;
    - distribution of marginal contributions per coalition size,
      explaining why LOO collapses while Shapley remains informative."""
    import itertools
    from scipy.stats import mode as smode
    from sklearn.metrics import accuracy_score

    d = {k: v.astype(np.float64) if v.ndim == 3 else v
         for k, v in np.load(os.path.join(OUT, "predictions.npz")).items()}
    rng = np.random.default_rng(SEED + 2)
    val_preds, y_val = d["disjoint_val"], d["y_val"]
    test_preds, y_test = d["disjoint_test"], d["y_test"]
    out = {}

    # --- |F| sensitivity ------------------------------------------------
    fracs = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    sens = {}
    full = compute_weights(val_preds, y_val)
    for frac in fracs:
        accs = {s: [] for s in WEIGHT_SCHEMES}
        corrs = {s: [] for s in WEIGHT_SCHEMES if s != "equal"}
        for rep in range(5 if frac < 1 else 1):
            idx = rng.choice(val_preds.shape[1],
                             max(30, int(frac * val_preds.shape[1])),
                             replace=False)
            w = compute_weights(val_preds[:, idx], y_val[idx])
            for s in WEIGHT_SCHEMES:
                accs[s].append(voting(test_preds, y_test,
                                      dpm.normalize_weights(w[s], "max"),
                                      "plurality"))
                if s != "equal" and np.std(w[s]) > 0:
                    corrs[s].append(float(np.corrcoef(w[s], full[s])[0, 1]))
        sens[str(frac)] = {
            "accuracy": {s: float(np.mean(a)) for s, a in accs.items()},
            "weight_corr": {s: float(np.mean(c)) if c else None
                            for s, c in corrs.items()}}
        print(f"|F| fraction {frac} done")
    out["F_sensitivity"] = {"fracs": fracs, "results": sens,
                            "F_size": int(val_preds.shape[1])}

    # --- marginal contributions by coalition size (LOO vs Shapley) -----
    indices = list(range(N_MODELS))
    votes = np.argmax(val_preds, axis=2)
    vfun = {}
    for r in range(N_MODELS + 1):
        for S in itertools.combinations(indices, r):
            if r == 0:
                vfun[S] = 0.1
            else:
                agg = smode(votes[list(S)], axis=0)[0]
                vfun[S] = accuracy_score(y_val, agg)
    by_size = {r: [] for r in range(N_MODELS)}
    for i in indices:
        rest = [j for j in indices if j != i]
        for r in range(N_MODELS):
            for S in itertools.combinations(rest, r):
                Si = tuple(sorted(S + (i,)))
                by_size[r].append(vfun[Si] - vfun[S])
    out["marginal_by_coalition_size"] = {
        str(r): {"mean_abs": float(np.mean(np.abs(v))),
                 "std": float(np.std(v))}
        for r, v in by_size.items()}
    print("marginal-contribution analysis done")

    with open(os.path.join(OUT, "extras.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved", os.path.join(OUT, "extras.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "teachers", "weights", "dp", "extras"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.stage in ("all", "teachers"):
        stage_teachers()
    if args.stage in ("all", "weights"):
        stage_weights()
    if args.stage in ("all", "dp"):
        stage_dp()
    if args.stage in ("all", "extras"):
        stage_extras()
