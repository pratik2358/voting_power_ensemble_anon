"""Regenerate the MNIST main-experiment figures of the paper
(line plots for 3-16 and up-to-200 models; grouped boxplots for 3-16)
from results/mnist_main/size_*.json, with distinct large markers and the
Okabe-Ito palette (consistent with all other figures of the revision).

Run from the repository root after experiments/mnist_main.py:
    python3 analysis/plot_mnist_main.py [--paper-dir /path/to/paper/plots]
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results", "mnist_main")

# fixed method order (same as the legend) and Okabe-Ito colors
ORDER = ["loo", "unweighted", "accuracy", "crh", "entropy", "shapley",
         "regression"]
STYLE = {  # method -> (color, marker, label)
    "loo":        ("#D55E00", "v", "LOO"),
    "unweighted": ("#333333", "o", "Equal power"),
    "accuracy":   ("#E69F00", "s", "Accuracy"),
    "crh":        ("#0072B2", "P", "CRH"),
    "entropy":    ("#CC79A7", "X", "Inverse entropy"),
    "shapley":    ("#009E73", "^", "Shapley"),
    "regression": ("#56B4E9", "D", "Regression"),
}

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#dddddd", "grid.linewidth": 0.6,
})


def load():
    data = {}
    for path in glob.glob(os.path.join(RES, "size_*.json")):
        n = int(re.search(r"size_(\d+)", path).group(1))
        data[n] = json.load(open(path))
    return dict(sorted(data.items()))


def line_plot(data, sizes, rule, outpath, skip=()):
    """Median accuracy vs. ensemble size. The y-range is fitted to the
    non-LOO medians; where LOO collapses, its curve exits the plot."""
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ref = []
    for m in ORDER:
        if m in skip:
            continue
        med = [np.median(data[n][m][rule]) for n in sizes
               if data[n][m][rule]]
        ns = [n for n in sizes if data[n][m][rule]]
        if not ns:
            continue
        if m != "loo":
            ref += med
        c, mk, lab = STYLE[m]
        ax.plot(ns, med, color=c, marker=mk, label=lab, markersize=7,
                linewidth=1.6)
    span = max(ref) - min(ref)
    pad = max(span * 0.15, 0.05)
    ax.set_ylim(min(ref) - pad, max(ref) + pad)
    ax.set_xlabel("Number of Models")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def box_plot(data, sizes, rule, outpath):
    """Grouped boxplots; the y-range is fitted to the non-LOO boxes (LOO
    can collapse to near-zero accuracy at some sizes, in which case its
    box is clipped at the bottom of the plot; its rank stays visible)."""
    fig, ax = plt.subplots(figsize=(13, 3.8))
    nt = len(ORDER)
    ref = np.concatenate([np.asarray(data[n][m][rule])
                          for n in sizes for m in ORDER
                          if m != "loo" and data[n][m][rule]])
    lo = np.percentile(ref, 0.5)
    hi = np.percentile(ref, 99.5)
    pad = (hi - lo) * 0.12
    for gi, n in enumerate(sizes):
        x0 = gi * (nt + 2)
        meds = [np.median(data[n][m][rule]) for m in ORDER]
        ranks = len(meds) - np.argsort(np.argsort(meds))
        for ti, m in enumerate(ORDER):
            arr = np.asarray(data[n][m][rule])
            c = STYLE[m][0] if m != "unweighted" else "#999999"
            bp = ax.boxplot(arr, positions=[x0 + ti], widths=0.8,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", linewidth=1))
            bp["boxes"][0].set_facecolor(c)
            bp["boxes"][0].set_alpha(0.85)
            top = min(bp["whiskers"][1].get_ydata()[1], hi + pad / 3)
            ax.text(x0 + ti, top, str(ranks[ti]), ha="center",
                    va="bottom", size=9)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks([gi * (nt + 2) + (nt - 1) / 2 for gi in range(len(sizes))])
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("Number of Models")
    ax.set_ylabel("Accuracy (%)")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=os.path.join(RES, "figures"))
    args = ap.parse_args()
    os.makedirs(args.paper_dir, exist_ok=True)
    data = load()
    detail = [n for n in data if n <= 16]
    large = [n for n in data if n > 16]
    for rule, tag in [("borda", "borda"), ("plurality", "mode")]:
        line_plot(data, detail, rule,
                  os.path.join(args.paper_dir, f"{tag}_line_16.pdf"))
        if large:
            # LOO is excluded from the large-ensemble plot: its median
            # accuracy collapses to near zero at some sizes, which would
            # crush the scale (its failure is shown in the boxplots)
            line_plot(data, sorted(set(detail[-1:]) | set(large)), rule,
                      os.path.join(args.paper_dir, f"{tag}_line_200.pdf"),
                      skip=("loo",))
        box_plot(data, detail, rule,
                 os.path.join(args.paper_dir, f"{tag}_box_loo_16.pdf"))
    print("figures written to", args.paper_dir)
