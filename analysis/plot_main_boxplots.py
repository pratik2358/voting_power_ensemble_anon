#!/usr/bin/env python3
"""Regenerate the main grouped-boxplot accuracy figures of the paper.

For each (dataset, voting rule) pair, draws grouped boxplots: one group per
ensemble size, one box per weighting method within each group, with the rank
of each method (1 = best median accuracy within the group) printed above the
upper whisker of its box.

Restyled per reviewer request: larger rank labels and a colorblind-safe
(Okabe-Ito) palette with fixed per-method colors. Also produces a standalone
horizontal legend (legend.pdf).
"""

import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results", "paper")
OUT_ACCS = "/home/pierre/git/students/pratik/dke_paper/additional_plots/accs"
OUT_PLOTS = "/home/pierre/git/students/pratik/dke_paper/plots"

METHODS = ["loo", "unweighted", "accuracy", "crh", "entropy", "shapley", "regression"]
LEGEND_LABELS = [
    "LOO",
    "Equal power",
    "Accuracy",
    "CRH",
    "Inverse entropy",
    "Shapley",
    "Regression",
]
# Okabe-Ito, colorblind-safe. For "unweighted" the nominal color is #333333,
# but the box is filled with a lighter gray so the black median stays visible.
BOX_COLORS = {
    "loo": "#D55E00",
    "unweighted": "#999999",
    "accuracy": "#E69F00",
    "crh": "#0072B2",
    "entropy": "#CC79A7",
    "shapley": "#009E73",
    "regression": "#56B4E9",
}
BOX_ALPHA = 0.85
LABEL_FONTSIZE = 12
RANK_FONTSIZE = 13
RULES = ["borda", "plurality"]


def load_cinic(prefix):
    """Return {rule: {n_models: {method: list of floats}}} for fl or di."""
    # File suffix -> ensemble size: '' = 12, '_2' = 16, '_3' = 8.
    sizes = {"_3": 8, "": 12, "_2": 16}
    data = {rule: {} for rule in RULES}
    for suffix, n in sizes.items():
        path = os.path.join(
            RESULTS, "cinic_noise_results", f"{prefix}_accuracy_dict_ci{suffix}.json"
        )
        with open(path) as f:
            d = json.load(f)
        for rule in RULES:
            data[rule][n] = {m: [float(x) for x in d[m][rule]] for m in METHODS}
    return data


def load_pickles(subdir, ns):
    """Return {rule: {n_models: {method: list of floats}}}."""
    data = {rule: {} for rule in RULES}
    for n in ns:
        path = os.path.join(RESULTS, subdir, f"accuracy_dict_{n}.pkl")
        with open(path, "rb") as f:
            d = pickle.load(f)
        for rule in RULES:
            data[rule][n] = {m: [float(x) for x in d[m][rule]] for m in METHODS}
    return data


def plot_grouped_boxes(groups, outpath):
    """groups: list of (n_models, {method: values}) in display order."""
    num_methods = len(METHODS)
    fig, ax = plt.subplots(figsize=(6.5, 4))

    tick_positions = []
    top_ys = []
    for gi, (n_models, method_values) in enumerate(groups):
        x_positions = np.arange(num_methods) + 1.5 * gi * (num_methods + 1)
        tick_positions.append(x_positions.mean())

        medians = np.array([np.median(method_values[m]) for m in METHODS])
        # Rank 1 = highest median accuracy within the group.
        order = np.argsort(-medians, kind="stable")
        ranks = np.empty(num_methods, dtype=int)
        ranks[order] = np.arange(1, num_methods + 1)

        for mi, method in enumerate(METHODS):
            bp = ax.boxplot(
                method_values[method],
                positions=[x_positions[mi]],
                widths=0.8,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1),
            )
            box = bp["boxes"][0]
            box.set_facecolor(BOX_COLORS[method])
            box.set_alpha(BOX_ALPHA)
            box.set_edgecolor("black")

            top_whisker = max(
                w.get_ydata().max() for w in bp["whiskers"]
            )
            top_ys.append(top_whisker)
            ax.text(
                x_positions[mi],
                top_whisker,
                str(ranks[mi]),
                ha="center",
                va="bottom",
                fontsize=RANK_FONTSIZE,
            )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(n) for n, _ in groups], fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=LABEL_FONTSIZE)
    ax.set_xlabel("Number of Models", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Accuracy (%)", fontsize=LABEL_FONTSIZE)

    # Leave headroom above the highest whisker for the rank labels.
    ymin, ymax = ax.get_ylim()
    top = max(top_ys)
    ax.set_ylim(ymin, top + 0.09 * (top - ymin))

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"wrote {outpath}")


def make_legend(outpath):
    handles = [
        Patch(
            facecolor=BOX_COLORS[m],
            alpha=BOX_ALPHA,
            edgecolor="black",
            label=label,
        )
        for m, label in zip(METHODS, LEGEND_LABELS)
    ]
    fig = plt.figure(figsize=(6.5, 0.4))
    fig.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    fig.savefig(outpath)
    plt.close(fig)
    print(f"wrote {outpath}")


def main():
    datasets = {
        "cinic_fl": load_cinic("fl"),
        "cinic_di": load_cinic("di"),
        "dmoz": load_pickles("dmoz_exps", [3, 5, 7, 10]),
        "phising": load_pickles("phising", [5, 10, 15]),
    }
    for name, data in datasets.items():
        for rule in RULES:
            groups = [(n, data[rule][n]) for n in sorted(data[rule])]
            plot_grouped_boxes(
                groups, os.path.join(OUT_ACCS, f"{name}_{rule}.pdf")
            )
    make_legend(os.path.join(OUT_PLOTS, "legend.pdf"))


if __name__ == "__main__":
    main()
