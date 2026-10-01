"""
Walk-forward backtest harness for SRS predictions (Phase 5 of the plan).

ROUGH VERSION, deliberately leaky: ratings are season-TOTAL SRS — the
rating has already seen the games it's predicting, so every number here
is optimistic. Phase 2's as-of-date SRS plugs into the same harness via
the `rating_fn(season, week)` hook, and the delta between the two runs is
exactly the cost of the leakage.

Per game it produces: predicted home margin (SRS diff + estimated HFA,
0 on neutral sites), win probability (normal CDF, std estimated from the
backtest's own residuals — the Phase 3/5 circular calibration the plan
calls for), edge vs the market's spread_line, and grades: win accuracy,
Brier score, ATS cover rate bucketed by |edge| (bet = the side the edge
points to, per the decided betting rule), plus the user's |edge| >= rule.

USAGE
-----
  python backtest_srs.py                        # 2021-2025 pooled, LEAKY rough read
  python backtest_srs.py --asof                 # the honest version (Phase 2 ratings)
  python backtest_srs.py --seasons 2023 2024 --asof
  python backtest_srs.py --cap 24 --rule 4
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from nfl_box_score_analysis import load_games, build_long_results  # noqa: E402
from srs import game_margins, solve_srs, blended_asof_ratings  # noqa: E402
# The sport-agnostic backtest machinery lives in shared/backtest_common.py;
# re-exported here so existing `from backtest_srs import ...` lines keep working.
from backtest_common import (  # noqa: E402,F401
    BREAKEVEN_COVER_PCT,
    EDGE_BUCKETS,
    collect_predictions,
    cover_line,
    estimate_hfa,
    grade,
    normal_cdf,
    report,
)


def season_total_ratings(games, seasons, cap=None):
    """The leaky stub: one rating per team per SEASON, computed from every
    game in that season including the ones being predicted. Phase 2
    replaces this with week-as-of ratings behind the same shape."""
    long = build_long_results(games, list(seasons))
    return {s: solve_srs(game_margins(long, s, cap=cap))[0] for s in seasons}







def main():
    p = argparse.ArgumentParser(description="SRS walk-forward backtest")
    p.add_argument("--seasons", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    p.add_argument("--cap", type=float, default=None, help="margin cap for SRS (see srs.py)")
    p.add_argument("--rule", type=float, default=4.0, help="betting-rule |edge| threshold")
    p.add_argument("--asof", action="store_true",
                   help="use no-leakage as-of ratings (Phase 2) instead of leaky season totals")
    p.add_argument("--k", type=float, default=4.0, help="cold-start blend, pseudo-games (--asof)")
    p.add_argument("--shrink", type=float, default=0.7, help="prior-season regression (--asof)")
    p.add_argument("--save_csv", action="store_true", help="save per-game grading to CSV")
    args = p.parse_args()

    games = load_games()
    if args.asof:
        ratings = blended_asof_ratings(games, args.seasons, cap=args.cap,
                                       k=args.k, shrink=args.shrink)
        rating_fn = lambda season, week: ratings[(season, week)]  # noqa: E731
        mode = f"AS-OF ratings, no leakage (k={args.k:g}, shrink={args.shrink:g})"
    else:
        ratings = season_total_ratings(games, args.seasons, cap=args.cap)
        rating_fn = lambda season, week: ratings[season]  # noqa: E731 -- the leaky stub
        mode = "ROUGH: season-total ratings, LEAKY"

    df = collect_predictions(games, args.seasons, rating_fn)
    if df.empty:
        sys.exit("no completed games found for those seasons")
    hfa = estimate_hfa(df.assign(neutral=df["neutral"]))
    df, ats, std = grade(df, hfa, args.rule)
    report(df, ats, args.seasons, f"HFA estimated from data: {hfa:+.2f} pts",
           f"margin-error std: {std:.2f} pts", args.rule, mode)

    if args.save_csv:
        out = pd.concat([df, ats[["bet_home", "ats_result", "bet_won"]]], axis=1)
        out.to_csv("backtest_srs_games.csv", index=False)
        print("\nSaved: backtest_srs_games.csv")


if __name__ == "__main__":
    main()
