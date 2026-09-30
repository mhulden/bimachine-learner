#!/usr/bin/env python3
"""Paper Figure 3: held-out agreement and exact recovery on random target bimachines.

Reads the output of random_targets.py. One panel per family; color and marker
give the target size; solid lines = mean held-out agreement, dashed lines =
exact normalized recovery (means over five targets); log-scaled x-axis.

Usage: python3 plot_random_targets.py [--in results/random_targets.jsonl] [--out results/random_targets]
"""
import argparse
import collections
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FAMILIES = [
    ("ia=3, oa=2, c<=2", r"$ia{=}3,\ oa{=}2,\ c{\leq}2$"),
    ("ia=4, oa=2, c<=3", r"$ia{=}4,\ oa{=}2,\ c{\leq}3$"),
    ("ia=5, oa=4, c<=4", r"$ia{=}5,\ oa{=}4,\ c{\leq}4$"),
]
# Target size -> (legend label, color, marker). Colors are a colorblind-safe
# categorical set (blue, orange, aqua, violet).
SIZES = [
    ("2x2", "2×2", "#2a78d6", "o"),
    ("4x4", "4×4", "#eb6834", "s"),
    ("6x6", "6×6", "#1baf7a", "^"),
    ("10x10", "10×10", "#4a3aa7", "D"),
]
N_TRAIN = [50, 100, 200, 400]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/random_targets.jsonl")
    ap.add_argument("--out", default="results/random_targets", help="output prefix (.pdf and .png are written)")
    args = ap.parse_args()

    cells = collections.defaultdict(list)
    for line in open(args.inp):
        r = json.loads(line)
        cells[(r["family"], r["size"], r["n_train"])].append(r)

    def mean(key, fam, size, n):
        v = cells[(fam, size, n)]
        return sum(float(r[key]) for r in v) / len(v)

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7.5, "font.family": "serif", "mathtext.fontset": "cm",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#555", "xtick.color": "#555", "ytick.color": "#555",
    })
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.15), sharey=True)
    for ax, (fam, title) in zip(axes, FAMILIES):
        for key, _label, color, marker in SIZES:
            agree = [mean("heldout_agreement", fam, key, n) for n in N_TRAIN]
            exact = [mean("exact_normalized_recovery", fam, key, n) for n in N_TRAIN]
            ax.plot(N_TRAIN, agree, color=color, lw=1.6, marker=marker, ms=4, mec="white", mew=0.6, zorder=3)
            ax.plot(N_TRAIN, exact, color=color, lw=1.2, ls=(0, (3, 2)), marker=marker, ms=3.5,
                    mfc="white", mec=color, mew=0.9, zorder=2)
        ax.set_xscale("log")
        ax.set_xticks(N_TRAIN)
        ax.set_xticklabels([str(n) for n in N_TRAIN])
        ax.minorticks_off()
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(43, 470)
        ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)
        ax.set_title(title, pad=4)
        ax.set_xlabel("training strings")
    axes[0].set_ylabel("rate")

    handles = [Line2D([], [], color=c, marker=m, lw=1.6, ms=4, mec="white", label=l) for _, l, c, m in SIZES]
    handles += [
        Line2D([], [], color="#444", lw=1.6, label="held-out agreement"),
        Line2D([], [], color="#444", lw=1.2, ls=(0, (3, 2)), marker="o", mfc="white", ms=3.5, label="exact recovery"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.8)
    fig.savefig(args.out + ".pdf")
    fig.savefig(args.out + ".png", dpi=220)
    print(f"wrote {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()
