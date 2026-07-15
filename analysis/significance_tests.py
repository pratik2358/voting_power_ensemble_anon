"""Pairwise significance tests on the archived accuracy results.

Each repetition of an experiment evaluates every voting-power method on
the same sampled test data (and, for DMOZ/phishing, the same trained
models), so accuracies are paired across methods within a repetition.
We therefore use the Wilcoxon signed-rank test on paired differences,
with Holm correction over the 21 method pairs within each
(dataset, ensemble size, voting rule) setting.

Outputs:
  results/significance/<setting>.csv        p-value matrices
  results/significance/summary_table.tex    LaTeX table for the paper
Run from the repository root: python3 analysis/significance_tests.py
"""

import json
import os
import pickle
from itertools import combinations

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(__file__)
PAPER = os.path.join(HERE, "..", "results", "paper")
OUT = os.path.join(HERE, "..", "results", "significance")

METHODS = ["unweighted", "accuracy", "regression", "shapley", "loo",
           "crh", "entropy"]
LABEL = {"unweighted": "Equal", "accuracy": "Accuracy",
         "regression": "Regression", "shapley": "Shapley", "loo": "LOO",
         "crh": "CRH", "entropy": "Inv.\\ entropy"}
RULES = ["borda", "plurality"]

# CINIC runs -> number of models (divs 3, 4, 2 -> 4*divs models)
CINIC_RUNS = [("", 12), ("_2", 16), ("_3", 8)]


def load_datasets():
    """List of (dataset, n_models, method -> rule -> np.array)."""
    settings = []
    # MNIST (results of experiments/mnist_main.py, if available)
    import glob
    import re
    for path in sorted(glob.glob(os.path.join(
            HERE, "..", "results", "mnist_main", "size_*.json"))):
        n = int(re.search(r"size_(\d+)", path).group(1))
        if n not in (5, 10, 16):   # keep the table readable
            continue
        d = json.load(open(path))
        settings.append(("MNIST", n, {
            m: {r: np.array(d[m][r], dtype=float) for r in RULES}
            for m in METHODS}))
    cinic_rerun = os.path.join(HERE, "..", "results", "cinic_main")
    for regime, tag in [("label flip.", "fl"), ("class imbal.", "di")]:
        for suffix, n_models in CINIC_RUNS:
            path = os.path.join(cinic_rerun,
                                f"{tag}_accuracy_dict_ci{suffix}.json")
            if not os.path.exists(path):
                path = os.path.join(PAPER, "cinic_noise_results",
                                    f"{tag}_accuracy_dict_ci{suffix}.json")
            d = json.load(open(path))
            settings.append((f"CINIC-10 ({regime})", n_models, {
                m: {r: np.array(d[m][r], dtype=float) for r in RULES}
                for m in METHODS}))
    rerun_dir = {"DMOZ": os.path.join(HERE, "..", "results", "dmoz_main"),
                 "Phishing": os.path.join(HERE, "..", "results",
                                          "phishing_main")}
    for name, sub, sizes in [("DMOZ", "dmoz_exps", [3, 5, 7, 10]),
                             ("Phishing", "phising", [5, 10, 15])]:
        for s in sizes:
            # prefer the rerun with the corrected code, if present
            rerun = os.path.join(rerun_dir[name], f"accuracy_dict_{s}.json")
            if os.path.exists(rerun):
                d = json.load(open(rerun))
            else:
                d = pickle.load(open(os.path.join(
                    PAPER, sub, f"accuracy_dict_{s}.pkl"), "rb"))
            settings.append((name, s, {
                m: {r: np.array([float(x) for x in d[m][r]]) for r in RULES}
                for m in METHODS}))
    settings.sort(key=lambda t: (t[0], t[1]))
    return settings


def holm(pvals):
    """Holm step-down adjusted p-values."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def analyze(settings):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for dname, size, dd in settings:
        for rule in RULES:
            pairs = list(combinations(METHODS, 2))
            pvals, medians = [], []
            for a, b in pairs:
                diff = dd[a][rule] - dd[b][rule]
                p = 1.0 if np.allclose(diff, 0) else wilcoxon(diff).pvalue
                pvals.append(p)
                medians.append(np.median(diff))
            adj = holm(np.array(pvals))
            mat = np.full((len(METHODS), len(METHODS)), np.nan)
            for (a, b), pa in zip(pairs, adj):
                i, j = METHODS.index(a), METHODS.index(b)
                mat[i, j] = mat[j, i] = pa
            slug = (f"{dname}_{size}_{rule}".replace(" ", "_")
                    .replace("(", "").replace(")", "").replace(".", ""))
            with open(os.path.join(OUT, slug + ".csv"), "w") as f:
                f.write("," + ",".join(METHODS) + "\n")
                for i, m in enumerate(METHODS):
                    f.write(m + "," + ",".join(
                        "" if np.isnan(v) else f"{v:.4g}" for v in mat[i])
                        + "\n")
            for (a, b), pa, md in zip(pairs, adj, medians):
                rows.append((dname, size, rule, a, b, md, pa))
    return rows


def summary_vs_equal(settings, rows):
    """LaTeX table: median accuracy gain over equal power per
    (dataset, ensemble size, voting rule), with Holm-adjusted
    Wilcoxon significance."""
    idx = {(d, s, r, a, b): (md, p) for d, s, r, a, b, md, p in rows}
    lines = [
        "\\begin{table}[tbp]",
        "  \\centering",
        "  \\caption{Median accuracy difference (in accuracy points, over"
        " paired repetitions) with respect to equal voting power, per"
        " dataset, ensemble size $n$, and voting rule."
        " Significance from Wilcoxon signed-rank tests on the paired"
        " repetitions, Holm-corrected over all method pairs of each"
        " setting; $^{**}$: adjusted $p<0.01$, $^{*}$: adjusted $p<0.05$,"
        " no star: not significant at the 0.05 level.}",
        "  \\label{tab:significance}",
        "  \\scriptsize\\setlength{\\tabcolsep}{3pt}",
        "  \\begin{tabular}{llrrrrrrr}",
        "    \\toprule",
        "    Dataset & Rule & $n$ & " + " & ".join(
            LABEL[m] for m in METHODS if m != "unweighted") + " \\\\",
        "    \\midrule",
    ]
    prev = None
    for dname, size, dd in settings:
        if prev is not None and prev != dname:
            lines.append("    \\midrule")
        for rule in RULES:
            cells = []
            for m in METHODS:
                if m == "unweighted":
                    continue
                if (dname, size, rule, "unweighted", m) in idx:
                    md, p = idx[(dname, size, rule, "unweighted", m)]
                    md = -md
                else:
                    md, p = idx[(dname, size, rule, m, "unweighted")]
                star = "^{**}" if p < 0.01 else ("^{*}" if p < 0.05 else "")
                cells.append(f"${md:+.2f}{star}$")
            rulename = "Borda" if rule == "borda" else "Plur."
            shown = dname if prev != dname else ""
            prev = dname
            lines.append(f"    {shown} & {rulename} & {size} & "
                         + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    path = os.path.join(OUT, "summary_table.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    settings = load_datasets()
    rows = analyze(settings)
    summary_vs_equal(settings, rows)
    print("\nHolm-adjusted Wilcoxon vs. equal power (median diff, p):")
    for d, s, r, a, b, md, p in rows:
        if "unweighted" in (a, b):
            other = b if a == "unweighted" else a
            sign = -md if a == "unweighted" else md
            flag = "**" if p < 0.01 else ("*" if p < 0.05 else "  ")
            print(f"  {d:22s} n={s:2d} {r:9s} {other:11s} "
                  f"{sign:+7.2f} {flag} p={p:.2g}")
