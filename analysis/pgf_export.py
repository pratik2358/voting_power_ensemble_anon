"""Helpers to emit paper figures as native pgfplots .tex fragments,
so that they are typeset with the document fonts. The color/marker
styles are defined once in a style file (see STYLE) that the paper
loads in its preamble; every fragment is a standalone tikzpicture to be
\\input inside a figure environment (its width adapts to \\linewidth at
the point of use)."""

import numpy as np

# Okabe-Ito palette; style names used by all fragments
STYLE = r"""% pgfplots styles for all figures (generated, do not edit)
\usepgfplotslibrary{statistics}
\pgfplotsset{compat=1.17}
\definecolor{vpeGray}{HTML}{333333}
\definecolor{vpeGrayLight}{HTML}{999999}
\definecolor{vpeOrange}{HTML}{E69F00}
\definecolor{vpeSky}{HTML}{56B4E9}
\definecolor{vpeGreen}{HTML}{009E73}
\definecolor{vpeVermillion}{HTML}{D55E00}
\definecolor{vpeBlue}{HTML}{0072B2}
\definecolor{vpePink}{HTML}{CC79A7}
\pgfplotsset{
  vpe axis/.style={
    axis lines=left,
    ymajorgrids,
    grid style={gray!25},
    tick label style={font=\scriptsize},
    label style={font=\footnotesize},
    legend style={font=\scriptsize, draw=none, fill=none},
    legend cell align=left,
    every axis title/.append style={font=\footnotesize},
  },
  vpeEqual/.style={vpeGray, mark=*, mark size=1.6pt},
  vpeAccuracy/.style={vpeOrange, mark=square*, mark size=1.6pt},
  vpeRegression/.style={vpeSky, mark=diamond*, mark size=2pt},
  vpeShapley/.style={vpeGreen, mark=triangle*, mark size=2pt},
  vpeLOO/.style={vpeVermillion, mark=triangle*, mark size=2pt,
                 every mark/.append style={rotate=180}},
  vpeCRH/.style={vpeBlue, mark=oplus*, mark size=1.8pt},
  vpeEntropy/.style={vpePink, mark=otimes*, mark size=1.8pt},
  every axis plot/.append style={line width=0.9pt},
}
"""

# method key -> (pgf style, fill color name, label)
METHOD = {
    "loo":          ("vpeLOO", "vpeVermillion", "LOO"),
    "unweighted":   ("vpeEqual", "vpeGrayLight", "Equal power"),
    "equal":        ("vpeEqual", "vpeGrayLight", "Equal"),
    "accuracy":     ("vpeAccuracy", "vpeOrange", "Accuracy"),
    "crh":          ("vpeCRH", "vpeBlue", "CRH"),
    "entropy":      ("vpeEntropy", "vpePink", "Inverse entropy"),
    "entropy_conf": ("vpeEntropy", "vpePink", "Entropy confidence"),
    "shapley":      ("vpeShapley", "vpeGreen", "Shapley"),
    "regression":   ("vpeRegression", "vpeSky", "Regression"),
}


def _fmt(x):
    return f"{float(x):.6g}"


def coords(xs, ys):
    return " ".join(f"({_fmt(x)},{_fmt(y)})" for x, y in zip(xs, ys))


def line_axis(series, xlabel, ylabel, xmode="normal", ymin=None, ymax=None,
              width=r"0.97\linewidth", height=None, legend_pos=None,
              ytick=None, extra=()):
    """series: list of (style, fillcolor, label_or_None, xs, ys, std_or_None).
    A std band is drawn behind its curve; labels go into the axis legend
    (only if legend_pos is given)."""
    height = height or r"0.72\linewidth"
    opts = ["vpe axis", f"width={width}", f"height={height}",
            f"xlabel={{{xlabel}}}", f"ylabel={{{ylabel}}}"]
    if xmode == "log":
        opts.append("xmode=log")
        opts.append("log ticks with fixed point")
    if ymin is not None:
        opts.append(f"ymin={_fmt(ymin)}")
    if ymax is not None:
        opts.append(f"ymax={_fmt(ymax)}")
    if ytick is not None:
        opts.append("ytick={" + ytick + "}")
    if legend_pos:
        opts.append(f"legend pos={legend_pos}")
    opts += list(extra)
    out = ["\\begin{tikzpicture}",
           "\\begin{axis}[" + ", ".join(opts) + "]"]
    for style, fill, label, xs, ys, std in series:
        if std is not None:
            up = np.asarray(ys) + np.asarray(std)
            lo = np.asarray(ys) - np.asarray(std)
            band = coords(xs, up) + " " + coords(list(xs)[::-1],
                                                 list(lo)[::-1])
            out.append(f"\\addplot[draw=none, fill={fill}, fill opacity=0.15,"
                       " forget plot] coordinates {" + band + "} -- cycle;")
    for style, fill, label, xs, ys, std in series:
        out.append(f"\\addplot[{style}] coordinates {{{coords(xs, ys)}}};")
        if legend_pos and label:
            out.append(f"\\addlegendentry{{{label}}}")
    out += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(out) + "\n"


def box_stats(arr):
    """Matplotlib-convention boxplot statistics (1.5 IQR whiskers
    clipped to the data)."""
    arr = np.asarray(arr, dtype=float)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    lo = arr[arr >= q1 - 1.5 * iqr].min()
    hi = arr[arr <= q3 + 1.5 * iqr].max()
    return lo, q1, med, q3, hi


def grouped_boxplots(groups, xlabel, ylabel, width=r"0.97\linewidth",
                     height=None, rank_font=r"\scriptsize",
                     ymin=None, ymax=None, ytick=None, box_extend=0.8):
    """groups: list of (tick_label, boxes) where boxes is a list of
    (fillcolor, data_array, rank_or_None). Boxes are placed at
    consecutive positions with a 2-slot gap between groups; ranks are
    printed above the upper whisker (clipped to the axis)."""
    height = height or r"0.62\linewidth"
    n_per = max(len(b) for _, b in groups)
    ticks, tickpos, items = [], [], []
    for gi, (lab, boxes) in enumerate(groups):
        x0 = gi * (n_per + 2)
        ticks.append(str(lab))
        tickpos.append(_fmt(x0 + (n_per - 1) / 2))
        for ti, (fill, arr, rank) in enumerate(boxes):
            items.append((x0 + ti, fill, box_stats(arr), rank))
    # boxplot-prepared plots have empty coordinate lists and do not feed
    # the automatic axis limits, so explicit limits are mandatory; boxes
    # are clamped to them (values far outside also overflow TeX
    # dimensions)
    if ymin is None or ymax is None:
        allv = np.array([st for _, _, st, _ in items])
        lo_, hi_ = allv.min(), allv.max()
        pad = (hi_ - lo_) * 0.1 or 1.0
        ymin = lo_ - pad if ymin is None else ymin
        ymax = hi_ + pad if ymax is None else ymax
    body = []
    for pos, fill, stats, rank in items:
        lo, q1, med, q3, hi = (min(max(v, ymin), ymax) for v in stats)
        body.append(
            "\\addplot[boxplot prepared={draw position=" + _fmt(pos)
            + f", lower whisker={_fmt(lo)}, lower quartile={_fmt(q1)}"
            + f", median={_fmt(med)}, upper quartile={_fmt(q3)}"
            + f", upper whisker={_fmt(hi)}, box extend={box_extend}}},"
            + f" draw=black, solid, line width=0.5pt, fill={fill},"
            + " fill opacity=0.85] coordinates {};")
        if rank is not None:
            body.append(
                f"\\node[font={rank_font}, above, inner sep=1pt] at"
                f" (axis cs:{_fmt(pos)},{_fmt(hi)}) {{{rank}}};")
    last_pos = (len(groups) - 1) * (n_per + 2) + n_per - 1
    opts = ["vpe axis", f"width={width}", f"height={height}",
            f"xlabel={{{xlabel}}}", f"ylabel={{{ylabel}}}",
            "xtick={" + ",".join(tickpos) + "}",
            "xticklabels={" + ",".join(ticks) + "}",
            # explicit x-limits: with only boxplot-prepared plots (empty
            # coordinate lists), automatic limits overflow TeX dimensions
            f"xmin=-1, xmax={last_pos + 1}",
            "boxplot/draw direction=y",
            "clip=false, boxplot/every box/.style={solid}",
            "x tick label style={font=\\scriptsize}"]
    if ymin is not None:
        opts.append(f"ymin={_fmt(ymin)}")
    if ymax is not None:
        opts.append(f"ymax={_fmt(ymax)}")
    if ytick is not None:
        opts.append("ytick={" + ytick + "}")
    out = (["\\begin{tikzpicture}",
            "\\begin{axis}[" + ", ".join(opts) + "]"] + body
           + ["\\end{axis}", "\\end{tikzpicture}"])
    return "\n".join(out) + "\n"


def grouped_bars(categories, series, xlabel, ylabel,
                 width=r"0.97\linewidth", height=None,
                 legend_pos="north east", ymin=0, ymax=None):
    """Grouped bar chart: categories = x tick labels, series = list of
    (fillcolor, label, values). Bars are placed on symbolic-free numeric
    positions (one slot per category) so bar width stays predictable."""
    height = height or r"0.72\linewidth"
    xs = list(range(len(categories)))
    if ymax is None:
        ymax = 1.1 * max(max(v) for _, _, v in series)
    opts = ["vpe axis", f"width={width}", f"height={height}",
            f"xlabel={{{xlabel}}}", f"ylabel={{{ylabel}}}",
            "ybar", "bar width=9pt",
            "xtick={" + ",".join(_fmt(x) for x in xs) + "}",
            "xticklabels={" + ",".join(str(c) for c in categories) + "}",
            f"xmin=-0.6, xmax={_fmt(len(xs) - 0.4)}",
            f"ymin={_fmt(ymin)}", f"ymax={_fmt(ymax)}",
            f"legend pos={legend_pos}",
            "legend style={font=\\scriptsize, draw=none, fill=none}",
            "x tick label style={font=\\scriptsize}"]
    out = ["\\begin{tikzpicture}",
           "\\begin{axis}[" + ", ".join(opts) + "]"]
    for fill, label, values in series:
        out.append(f"\\addplot[draw={fill}!50!black, fill={fill},"
                   " fill opacity=0.85] coordinates"
                   f" {{{coords(xs, values)}}};")
        out.append(f"\\addlegendentry{{{label}}}")
    out += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(out) + "\n"


def legend_row(entries, columns=-1):
    """Standalone legend row: entries = list of (style, label). Rendered
    through a hidden 1pt axis whose only output is its legend."""
    out = ["\\begin{tikzpicture}",
           "\\begin{axis}[hide axis, scale only axis, width=1pt,"
           " height=1pt, xmin=0, xmax=1, ymin=0, ymax=1,"
           f" legend columns={columns},"
           " legend style={draw=none, fill=none, font=\\scriptsize,"
           " at={(0.5,0.5)}, anchor=center,"
           " /tikz/every even column/.append style={column sep=0.7em}}]"]
    for style, label in entries:
        out.append(f"\\addlegendimage{{{style}}}")
        out.append(f"\\addlegendentry{{{label}}}")
    out += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(out) + "\n"
