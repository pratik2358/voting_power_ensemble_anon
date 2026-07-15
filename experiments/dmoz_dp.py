"""Differentially private weighted voting on DMOZ ensembles.

Same experimental design as mnist_dp.py (see that file for the setting),
on the DMOZ URL classification dataset with Multinomial Naive Bayes
teachers, using the same preprocessing as the paper's main DMOZ
experiments (notebooks/dmoz/dmoz.ipynb): the URL is split into domain
name and URI, both tokenized and stemmed, and vectorized with TF-IDF
(1-grams for the domain, 1-3-grams for the URI).

For the DP experiments the teachers are trained on *disjoint* stratified
partitions of the training split (the paper's main DMOZ experiments use
overlapping 80% subsamples instead), so that the add/remove-one-example
adjacency affects a single teacher; the label noise follows the paper's
DMOZ scheme (complement flipping with rate i*(1/n - 0.02) for teacher i).

Requires the Kaggle DMOZ file "URL Classification.csv"
(https://www.kaggle.com/datasets/shawon10/url-classification-dataset-dmoz).

Run from the repository root:
    python3 experiments/dmoz_dp.py --csv "data/URL Classification.csv"
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vpe.voting_utils import WeightFinding_sklearn, voting
from vpe import dp_mechanisms as dpm

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "dp", "dmoz")
SEED = 20260713
N_MODELS = 10


def parse_urls(urls):
    """Domain-name and URI token strings, as in the paper's notebook."""
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    cache = {}

    def stem(t):
        s = cache.get(t)
        if s is None:
            s = stemmer.stem(t)
            cache[t] = s
        return s

    pattern = re.compile(r"^(?P<protocal>.*?)://(?P<domainname>.*?)(/(?P<uri>.*?))?$")

    def tokenize(s):
        if not s:
            return ""
        return " ".join(stem(t) for t in re.split(r"\W", s) if t)

    domains, uris = [], []
    for url in urls:
        m = re.match(pattern, url) if isinstance(url, str) else None
        g = m.groupdict() if m else {"domainname": " ", "uri": " "}
        domains.append(tokenize(g.get("domainname") or ""))
        uris.append(tokenize(g.get("uri") or ""))
    return domains, uris


def build_data(csv_path, rng):
    import pandas as pd
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(csv_path, header=None, names=["index", "url", "label"])
    df = df.dropna(subset=["url", "label"])
    y = LabelEncoder().fit_transform(df["label"])
    print(f"{len(df)} URLs, {len(np.unique(y))} classes; parsing URLs...")
    domains, uris = parse_urls(df["url"].tolist())
    print("vectorizing...")
    X = hstack([
        TfidfVectorizer(analyzer="word", ngram_range=(1, 1)).fit_transform(domains),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 3)).fit_transform(uris),
    ]).tocsr()

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED)
    X_te, X_val, y_te, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)


def train_teachers(X_tr, y_tr, rng):
    from sklearn.naive_bayes import MultinomialNB
    n_classes = len(np.unique(y_tr))
    perm = rng.permutation(X_tr.shape[0])
    part = X_tr.shape[0] // N_MODELS
    p = 1 / N_MODELS - 0.02
    models = []
    for i in range(N_MODELS):
        idx = perm[i * part:(i + 1) * part]
        yi = y_tr[idx].copy()
        n_flip = int(i * p * len(yi))
        fidx = rng.choice(len(yi), n_flip, replace=False)
        yi[fidx] = n_classes - 1 - yi[fidx]     # complement flip, as in the paper
        m = MultinomialNB()
        m.fit(X_tr[idx], yi)
        models.append(m)
        print(f"teacher {i} (flip rate {i * p:.2f}) trained")
    return models


def main(csv_path, reps=50, eps_w=5.0):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = build_data(csv_path, rng)
    models = train_teachers(X_tr, y_tr, rng)

    wf_val = WeightFinding_sklearn(models, (X_val, y_val), "cpu")
    wf_te = WeightFinding_sklearn(models, (X_te, y_te), "cpu")
    val_preds = wf_val.y_preds.astype(np.float64)
    test_preds = wf_te.y_preds.astype(np.float64)
    n_classes = test_preds.shape[2]

    weights = {
        "equal": np.ones(N_MODELS),
        "accuracy": wf_val.accuracy_weights(),
        "entropy": wf_val.entropy_weights(),
        "entropy_conf": np.clip(
            1 - dpm.mean_entropies(val_preds) / np.log(n_classes), 0, 1),
        "crh": wf_val.crh(),
        "regression": wf_val.regression(),
        "shapley": wf_val.shapley(method="plurality"),
        "loo": wf_val.loo(method="plurality"),
    }
    json.dump({k: np.asarray(v).tolist() for k, v in weights.items()},
              open(os.path.join(OUT, "weights.json"), "w"), indent=1)
    results = {"teacher_accuracy": [
        100 * float(np.mean(np.argmax(test_preds[i], 1) == y_te))
        for i in range(N_MODELS)]}
    print("teacher accuracies:", np.round(results["teacher_accuracy"], 1))

    clip = {"accuracy": (0, 1), "entropy": (0, 5), "entropy_conf": (0, 1),
            "crh": (0, 1), "regression": (0, 1), "shapley": (0, 1),
            "loo": (0, 0.1)}
    joint = {"crh", "regression", "shapley", "loo"}
    eps_grid = np.round(np.logspace(np.log10(0.05), np.log10(10), 25), 4)
    results.update({"eps_grid": eps_grid.tolist(), "reps": reps,
                    "eps_w": eps_w,
                    "nonprivate": {s: float(voting(
                        test_preds, y_te,
                        dpm.normalize_weights(np.asarray(w, float), "max"),
                        "plurality")) for s, w in weights.items()}})

    mechs = {"vote_count": dpm.noisy_vote_count,
             "output_pert": dpm.output_perturbation,
             "randomized_response": dpm.randomized_response}
    results["D1"] = {}
    eq = np.ones(N_MODELS)
    for name, mech in mechs.items():
        accs = [dpm.accuracy(mech(test_preds, eq, e, rng, reps=reps), y_te)
                for e in eps_grid]
        results["D1"][name] = {"mean": [float(np.mean(a)) for a in accs],
                               "std": [float(np.std(a)) for a in accs]}
        print("D1", name, "done")

    results["D2"] = {"released_weights": {}, "curves": {}}
    for s, w in weights.items():
        w = np.asarray(w, float)
        if s == "equal":
            w_rel = eq
        else:
            w_rel = dpm.normalize_weights(dpm.release_weights(
                w, eps_w, rng, clip=clip[s], joint=s in joint), "max")
        results["D2"]["released_weights"][s] = w_rel.tolist()
        accs = [dpm.accuracy(dpm.noisy_vote_count(
            test_preds, w_rel, e, rng, reps=reps), y_te) for e in eps_grid]
        results["D2"]["curves"][f"{s}|vote_count"] = {
            "mean": [float(np.mean(a)) for a in accs],
            "std": [float(np.std(a)) for a in accs]}
        # exact weights, for reference
        w_ex = dpm.normalize_weights(w if s != "equal" else eq, "max")
        accs = [dpm.accuracy(dpm.noisy_vote_count(
            test_preds, w_ex, e, rng, reps=reps), y_te) for e in eps_grid]
        results["D2"]["curves"][f"{s}|vote_count_exactw"] = {
            "mean": [float(np.mean(a)) for a in accs],
            "std": [float(np.std(a)) for a in accs]}
        # per-teacher powers used directly with the public cap
        if s in ("accuracy", "entropy_conf"):
            accs = [dpm.accuracy(dpm.noisy_vote_count(
                test_preds, w, e, rng, reps=reps, cap=clip[s][1]), y_te)
                for e in eps_grid]
            results["D2"]["curves"][f"{s}|vote_count_capped"] = {
                "mean": [float(np.mean(a)) for a in accs],
                "std": [float(np.std(a)) for a in accs]}
        print("D2", s, "done")

    # margin decomposition (Proposition 1) on the real DMOZ ensemble
    results["D4"] = {}
    for s in ["equal", "accuracy", "entropy_conf", "regression"]:
        w = np.clip(np.asarray(weights[s], float), 0, None)
        w = dpm.normalize_weights(w, "max")
        dec = dpm.margin_decomposition(test_preds, w)
        dis = ~dec["agree"]
        results["D4"][s] = {
            "agree_fraction": float(np.mean(dec["agree"])),
            "bound_holds": float(np.mean(dec["Gamma"] >= dec["bound"] - 1e-9)),
            "bound_positive_on_disagree":
                float(np.mean(dec["bound"][dis] > 0)) if dis.any() else None,
            "mean_Gamma": float(np.mean(dec["Gamma"])),
            "mean_bound": float(np.mean(dec["bound"])),
        }

    json.dump(results, open(os.path.join(OUT, "dp_results.json"), "w"))
    print("saved", os.path.join(OUT, "dp_results.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "URL Classification.csv"))
    args = ap.parse_args()
    main(args.csv)
