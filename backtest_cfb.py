"""
backtest_cfb.py -- honest walk-forward backtest of the SRS model on
college football (FBS), mirroring backtest_srs.py's --asof mode.

Purpose: estimate the constants predict_cfb_week.py needs --
HFA_POINTS, MARGIN_STD, and the cold-start blend (k, shrink) -- from
data, not guesses, exactly the way the NFL constants were estimated
(intermediate/qa_log.md).

For each season 2022-current, every week's games are predicted with
ratings built only from games BEFORE that week (no leakage), blended
against the prior season for the cold start. 2021 is loaded only as the
prior season for 2022's cold start.

USAGE
-----
  python backtest_cfb.py                 # HFA/std/accuracy/Brier + k/shrink sweep
  python backtest_cfb.py --k 4 --shrink 0.7   # single config, no sweep
"""

import argparse
import itertools
import sys

import pandas as pd

from cfb_games import load_cfb_games
from srs import blended_asof_ratings
from backtest_srs import collect_predictions, estimate_hfa, grade

BACKTEST_SEASONS = [2022, 2023, 2024, 2025, 2026]


def run(games, seasons, k, shrink):
    ratings = blended_asof_ratings(games, seasons, k=k, shrink=shrink)
    df = collect_predictions(games, seasons, lambda s, w: ratings[(s, w)])
    hfa = estimate_hfa(df)
    df, _ats, std = grade(df, hfa, rule_edge=None)
    mae = (df["actual_margin"] - df["pred_margin"]).abs().mean()
    return df, hfa, std, mae


def main():
    p = argparse.ArgumentParser(description="Walk-forward SRS backtest, college football")
    p.add_argument("--k", type=float, default=None, help="skip the sweep, use this k")
    p.add_argument("--shrink", type=float, default=None, help="skip the sweep, use this shrink")
    p.add_argument("--seasons", type=int, nargs="+", default=BACKTEST_SEASONS)
    args = p.parse_args()

    games = load_cfb_games()
    seasons = [s for s in args.seasons if s in set(games["season"])]

    if args.k is not None and args.shrink is not None:
        grid = [(args.k, args.shrink)]
    else:
        grid = list(itertools.product([2.0, 4.0, 6.0, 8.0], [0.5, 0.7, 0.9]))

    results = []
    for k, shrink in grid:
        df, hfa, std, mae = run(games, seasons, k, shrink)
        acc = df["correct"].mean()
        brier = df["brier"].mean()
        results.append({"k": k, "shrink": shrink, "hfa": hfa, "std": std,
                        "mae": mae, "winner_acc": acc, "brier": brier})
        print(f"k={k:g} shrink={shrink:g}  hfa={hfa:+.2f}  std={std:.2f}  "
              f"mae={mae:.2f}  winners={acc:.1%}  brier={brier:.3f}")

    out = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
    best = out.iloc[0]
    print(f"\n=== best by MAE over {seasons} ===")
    print(out.round(3).to_string(index=False))
    print(f"\n-> constants for predict_cfb_week.py: k={best['k']:g}, "
          f"shrink={best['shrink']:g}, HFA_POINTS={best['hfa']:.2f}, "
          f"MARGIN_STD={best['std']:.2f}")
    print("Same honesty rule as the NFL model: this measures how well SRS "
          "ranks college teams. Nothing here says it beats the market.")


if __name__ == "__main__":
    sys.exit(main())
