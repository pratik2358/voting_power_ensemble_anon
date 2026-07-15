"""Figures for the differential-privacy experiments (Section 5).

Reads results/dp/mnist/{dp_results.json,extras.json} and writes the
paper figures as PDF. Colors follow the Okabe-Ito colorblind-safe
palette with a fixed method-to-color assignment and distinct marker
shapes across all figures.

Run from the repository root:
    python3 analysis/plot_dp.py [--paper-dir /path/to/paper/plots]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results", "dp", "mnist")

# method -> (color, marker, label); Okabe-Ito palette, fixed assignment
METHOD_STYLE = {
    "equal":        ("#333333", "o", "Equal"),
    "accuracy":     ("#E69F00", "s", "Accuracy"),
    "regression":   ("#56B4E9", "D", "Regression"),
    "shapley":      ("#009E73", "^", "Shapley"),
    "loo":          ("#D55E00", "v", "LOO"),
    "crh":          ("#0072B2", "P", "CRH"),
    "entropy_conf": ("#CC79A7", "X", "Entropy confidence"),
}
MECH_STYLE = {
    "vote_count": ("#0072B2", "o", "Vote-count perturbation"),
    "output_pert": ("#D55E00", "s", "Output perturbation"),
    "randomized_response": ("#009E73", "^", "Randomized response"),
}
ALLOC_STYLE = {
    "uniform": ("#333333", "o", "Uniform"),
    "budget_prop_margin_oracle":
        ("#0072B2", "s", r"budget $\propto$ margin (exact)"),
    "budget_prop_margin_released":
        ("#56B4E9", "D", r"budget $\propto$ margin (released)"),
    "noise_prop_margin_oracle":
        ("#D55E00", "v", r"noise $\propto$ margin (noise sink, exact)"),
    "noise_prop_margin_released":
        ("#CC79A7", "X", r"noise $\propto$ margin (noise sink, released)"),
}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#dddddd", "grid.linewidth": 0.6,
})


def curve(ax, x, mean, std, color, marker, label, every=2):
    mean, std = np.asarray(mean), np.asarray(std)
    ax.plot(x, mean, color=color, marker=marker, label=label,
            markersize=5, markevery=every, linewidth=1.6)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15,
                    linewidth=0)


def fig_mechanisms(r, outdir):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    eps = r["eps_grid"]
    for m, (c, mk, lab) in MECH_STYLE.items():
        curve(ax, eps, r["D1"][m]["mean"], r["D1"][m]["std"], c, mk, lab)
    ax.axhline(r["nonprivate"]["equal"]["plurality"], color="#888888",
               linestyle="--", linewidth=1,
               label="Non-private ensemble")
    ax.set_xscale("log")
    ax.set_xlabel(r"privacy budget $\epsilon$ per query")
    ax.set_ylabel("test accuracy (%)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "dp_mechanisms.pdf"))
    plt.close(fig)


def fig_weights(r, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.0), sharey=True)
    eps = r["eps_grid"]
    # deployable curves: per-teacher powers used directly with the public
    # cap; jointly computed powers released under DP
    deploy = {"equal": "vote_count",
              "accuracy": "vote_count_capped",
              "entropy_conf": "vote_count_capped",
              "regression": "vote_count", "shapley": "vote_count",
              "loo": "vote_count", "crh": "vote_count"}
    for ax, key_of, title in [
            (axes[0], lambda s: f"{s}|vote_count_exactw",
             "exact voting powers (public)"),
            (axes[1], lambda s: f"{s}|{deploy[s]}",
             "deployable: capped or DP-released")]:
        for s, (c, mk, lab) in METHOD_STYLE.items():
            d = r["D2"]["curves"][key_of(s)]
            curve(ax, eps, d["mean"], d["std"], c, mk, lab)
        ax.set_xscale("log")
        ax.set_xlabel(r"privacy budget $\epsilon$ per query")
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis="y")
    axes[0].set_ylabel("test accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(outdir, "dp_weights.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def fig_allocation(r, outdir):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    eps = r["D3"]["eps_total_grid"]
    for m, (c, mk, lab) in ALLOC_STYLE.items():
        d = r["D3"]["curves"][m]
        curve(ax, eps, d["mean"], d["std"], c, mk, lab)
    ax.set_xscale("log")
    ax.set_xlabel(r"total privacy budget $\epsilon$ per query"
                  r" (sequential composition)")
    ax.set_ylabel("test accuracy (%)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "dp_allocation.pdf"))
    plt.close(fig)


def fig_margins(r, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.9))
    # (a) per-teacher entropy vs margin, annotated with flip rate
    pt = r["D4"]["per_teacher"]
    ent, mar = pt["mean_entropy"], pt["mean_margin"]
    axes[0].scatter(ent, mar, color="#0072B2", s=30, zorder=3)
    for i, fr in enumerate(r["flip_rates"]):
        axes[0].annotate(f"{fr:.1f}", (ent[i], mar[i]),
                         textcoords="offset points", xytext=(5, 3),
                         fontsize=7, color="#555555")
    axes[0].set_xlabel(r"mean prediction entropy $\bar H_i$")
    axes[0].set_ylabel(r"mean top-2 margin $\bar\gamma_i$")
    axes[0].grid(True)
    # (b) flip probability vs per-teacher noise scale per weighting
    for s in ["equal", "accuracy", "entropy_conf", "regression"]:
        c, mk, lab = METHOD_STYLE[s]
        d = r["D4"][s]["flip_prob_per_teacher"]
        sig = np.asarray(d["sigma"])
        prob = np.asarray(d["prob"])
        keep = (sig >= 5e-3) & (prob > 0)
        axes[1].plot(sig[keep], prob[keep], color=c, marker=mk, label=lab,
                     markersize=4, linewidth=1.4)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"per-teacher Gaussian noise scale $\sigma$")
    axes[1].set_ylabel("decision flip probability")
    axes[1].grid(True, axis="y")
    axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "dp_margins.pdf"))
    plt.close(fig)


def fig_loo_shapley(extras, outdir):
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    d = extras["marginal_by_coalition_size"]
    sizes = sorted(int(k) for k in d)
    mean_abs = [d[str(s)]["mean_abs"] for s in sizes]
    std = [d[str(s)]["std"] for s in sizes]
    ax.errorbar(sizes, mean_abs, yerr=std, color="#0072B2", marker="o",
                markersize=5, linewidth=1.6, capsize=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"coalition size $|S|$")
    ax.set_ylabel(r"mean $|v(S\cup\{i\})-v(S)|$")
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "loo_shapley_marginals.pdf"))
    plt.close(fig)


def table_f_sensitivity(extras, outdir):
    fs = extras["F_sensitivity"]
    fracs = fs["fracs"]
    schemes = ["accuracy", "regression", "shapley", "loo", "crh", "entropy"]
    lines = [
        "\\begin{table}[tbp]",
        "  \\centering",
        "  \\caption{Sensitivity to the size of the auxiliary set"
        " $\\mathcal{F}$ (MNIST, 10 models): Pearson correlation between"
        " the voting powers computed on a random fraction of $\\mathcal F$"
        " ($|\\mathcal F|=" + f"{fs['F_size']:,}".replace(",", "{,}")
        + "$) and those computed on all of it, averaged over 5 draws.}",
        "  \\label{tab:f-sensitivity}",
        "  \\small",
        "  \\begin{tabular}{l" + "r" * len(fracs) + "}",
        "    \\toprule",
        "    & \\multicolumn{" + str(len(fracs)) +
        "}{c}{fraction of $\\mathcal{F}$} \\\\",
        "    Method & " + " & ".join(str(f) for f in fracs) + " \\\\",
        "    \\midrule",
    ]
    label = {"accuracy": "Accuracy", "regression": "Regression",
             "shapley": "Shapley", "loo": "LOO", "crh": "CRH",
             "entropy": "Inv.\\ entropy"}
    for s in schemes:
        cells = []
        for f in fracs:
            v = fs["results"][str(f)]["weight_corr"].get(s)
            cells.append("--" if v is None else f"{v:.3f}")
        lines.append(f"    {label[s]} & " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    with open(os.path.join(outdir, "f_sensitivity_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=os.path.join(RES, "figures"))
    args = ap.parse_args()
    os.makedirs(args.paper_dir, exist_ok=True)
    r = json.load(open(os.path.join(RES, "dp_results.json")))
    fig_mechanisms(r, args.paper_dir)
    fig_weights(r, args.paper_dir)
    fig_allocation(r, args.paper_dir)
    fig_margins(r, args.paper_dir)
    extras_path = os.path.join(RES, "extras.json")
    if os.path.exists(extras_path):
        extras = json.load(open(extras_path))
        fig_loo_shapley(extras, args.paper_dir)
        table_f_sensitivity(extras, args.paper_dir)
    print("figures written to", args.paper_dir)
