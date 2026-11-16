"""
plot_style.py — shared publication style for figures/figures_beautiful/*.

One place to fix fonts, dpi, and the color vocabulary so every "beautiful"
figure in this folder reads as one consistent system instead of N one-off
matplotlib defaults. Import and call apply() once at the top of each
*_beautiful.py script, before creating any figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

GT_COLOR = "#4C72B0"
PRED_COLOR = "#DD8452"
HO_COLOR = "#8C8C8C"

CT_COLORS = {
    "gCap":     "#2CA02C",
    "aCap":     "#D62728",
    "Pericyte": "#1F77B4",
    "VEC":      "#9467BD",
}

COND_COLORS = {
    "HO":  HO_COLOR,
    "AZI": "#17BECF",
}

COND_COLORS_HUMAN = {
    "Acute":   HO_COLOR,
    "BPD":     "#17BECF",
    "BPD-PH":  "#E45756",
}

SIG_COLOR = "#333333"

DPI = 300

# Matched to the main author's figures (measured from his assembled
# Figure_updated.pdf): every figure there is placed 7.5 in (540 pt) wide and all
# text is Arial Bold. In his figures laid out like these -- two or three graphs
# per row (Fig 2, 3-S1, 3-S2, 3-S4) -- the text is 9.4 pt on the page (median).
# His denser figures run 5-7 pt and are not the reference for these. So every
# figure here is drawn exactly FIG_W wide and saved untrimmed, and all text is
# one size, FS -- the size it will have on the page.
FIG_W = 7.5
FS = 9.5

FS_SUPTITLE = FS
FS_TITLE = FS
FS_LABEL = FS
FS_TICK = FS
FS_LEGEND = FS
FS_ANNOT = FS


def apply() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.weight": "bold",
        "font.size": FS,
        "figure.titlesize": FS,
        "figure.titleweight": "bold",
        "axes.titlesize": FS,
        "axes.titleweight": "bold",
        "axes.labelsize": FS,
        "axes.labelweight": "bold",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "legend.title_fontsize": FS,
        "legend.frameon": False,
        "svg.fonttype": "none",
    })


def two_per_row(n_panels: int, row_h: float):
    """Panels two per row on a FIG_W-wide figure (a single panel is half width).
    An odd count leaves the last row's right-hand slot empty; it is returned as
    `spare`, axis turned off, so the legend can sit in it.
    -> (fig, axes, spare)"""
    ncols = 1 if n_panels == 1 else 2
    nrows = -(-n_panels // ncols)
    legend_h = 0.6 if n_panels == 1 else 0.0   # put_legend stacks it under a lone panel
    fig, grid = plt.subplots(nrows, ncols, squeeze=False, layout="constrained",
                             figsize=(FIG_W / 2 * ncols, row_h * nrows + legend_h))
    cells = list(grid.ravel())
    spare = cells[n_panels] if len(cells) > n_panels else None
    if spare is not None:
        spare.set_axis_off()
    return fig, cells[:n_panels], spare


def put_legend(fig, spare, handles, labels) -> None:
    """Into the spare slot when there is one, otherwise under the panels."""
    if spare is not None:
        spare.legend(handles, labels, loc="center", frameon=False)
    else:
        fig.legend(handles, labels, loc="outside lower center", frameon=False,
                   ncol=3 if fig.get_figwidth() >= FIG_W else 1)


def save(fig, path) -> None:
    """Saved at exactly the size it was drawn -- no bbox trimming, which would
    change the width and so the text size once the figure is placed FIG_W wide."""
    fig.savefig(path, dpi=DPI, facecolor="white")


def out_dir(fig_dir: Path, tag: str) -> Path:
    """figures/figures_beautiful/<tag>/ , one subfolder per sampling temperature."""
    d = fig_dir / "figures_beautiful" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


SIG_ALPHA = 0.05


def sig_label(p: float) -> str:
    """The printed p-value. Stars and "ns" are deliberately not used: a reader
    comparing two arms of a figure otherwise compares star counts, which are a
    function of n as much as of effect size -- and the two arms here differ in
    n by an order of magnitude (see the power curve in autophagy_panel/reports/
    rat_topk_detectability_gcap_up_power_curve.csv). The number is printed
    always; significance is carried by weight, not by a separate token."""
    if p < 1e-3:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


sig_stars = sig_label


def sig_bracket(ax, x1: float, x2: float, y: float, h: float, p: float, color: str = SIG_COLOR,
                 fontsize: float = FS) -> None:
    """Draw one horizontal significance bracket between x1 and x2 at height y
    (bracket arms rise by h), labelled with the p-value itself -- bold when
    p < SIG_ALPHA, normal weight otherwise."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.1, color=color)
    ax.text((x1 + x2) / 2, y + h, sig_label(p), ha="center", va="bottom",
            fontsize=fontsize, color=color,
            fontweight="bold" if p < SIG_ALPHA else "normal")
