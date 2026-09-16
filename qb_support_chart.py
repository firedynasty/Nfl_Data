"""
QB chart: Pass ADOT vs catch%, colored by Game Performance Score.

Labels are placed with force-directed repulsion (adjustText): each label
is pushed away from other labels and points until nothing overlaps -- the
"forcefield" behavior -- with a thin connector line back to its dot.

Usage:
    python qb_support_chart.py                     # uses default CSV/PNG paths
    python qb_support_chart.py --csv some.csv --out chart.png

    On this machine, run with the `test` conda env:
    /Users/stanleytan/anaconda3/envs/test/bin/python qb_support_chart.py
    (base anaconda matplotlib is broken against numpy 2.x; `test` has
    matplotlib + adjustText installed)

Weekly refresh:
    python team_composite.py --seasons 2026 --qb_csv ~/Downloads/qb_adot_catchpct.csv
    python qb_support_chart.py
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import cm, colors as mcolors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/Users/stanleytan/Downloads/qb_adot_catchpct_with_game_performance.csv")
    ap.add_argument("--out", default="/Users/stanleytan/Downloads/qb_adot_catchpct_game_performance.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    fig, ax = plt.subplots(figsize=(16, 10))
    norm = mcolors.Normalize(vmin=20, vmax=90)
    cmap = matplotlib.colormaps["RdYlGn"]
    colors = [cmap(norm(s)) for s in df["game_performance_score"]]

    ax.scatter(df["adot"], df["catch_pct"] * 100, c=colors, s=120,
               edgecolors="k", linewidths=0.5, zorder=3)

    texts = [
        ax.text(r["adot"], r["catch_pct"] * 100,
                f"{r['passer_player_name']} {r['record_str']} ({r['game_performance_score']:.0f})",
                fontsize=9, zorder=4)
        for _, r in df.iterrows()
    ]
    adjust_text(
        texts, x=df["adot"].to_numpy(), y=(df["catch_pct"] * 100).to_numpy(),
        ax=ax,
        force_text=(0.6, 0.8),    # label-vs-label repulsion (the forcefield)
        force_points=(0.4, 0.6),  # label-vs-point repulsion
        expand=(1.3, 1.6),
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.6, alpha=0.7),
    )

    m, b = np.polyfit(df["adot"], df["catch_pct"] * 100, 1)
    xs = np.linspace(df["adot"].min(), df["adot"].max(), 50)
    ax.plot(xs, m * xs + b, color="gray", lw=1, zorder=2)

    ax.axvline(df["adot"].mean(), color="gray", ls="--", lw=0.8)
    ax.axhline(df["catch_pct"].mean() * 100, color="gray", ls="--", lw=0.8)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label="Game Performance Score (0-100)")

    ax.set_title("Pass ADOT & CATCH % - 2026\n(color = Game Performance Score, label = QB, record, score)",
                 fontsize=14)
    ax.set_xlabel("Pass ADOT")
    ax.set_ylabel("CATCH %")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.margins(0.06)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved {args.out}")

    # Tidy CSV of exactly what the chart plots, sorted by score
    chart_data = df[["passer_player_name", "team", "record_str", "adot",
                     "catch_pct", "game_performance_score"]].copy()
    chart_data["adot"] = chart_data["adot"].round(1)
    chart_data["catch_pct"] = (chart_data["catch_pct"] * 100).round(1)
    chart_data = chart_data.sort_values("game_performance_score", ascending=False)
    data_out = args.out.rsplit(".", 1)[0] + "_data.csv"
    chart_data.to_csv(data_out, index=False)
    print(f"saved {data_out}")
    print(chart_data.to_string(index=False))


if __name__ == "__main__":
    main()
