"""
predict_cfb_week.py -- the college football (FBS) version of predict_week.py.

One row per game for a season/week, played or upcoming:
  - both teams' SRS as-of that week (no leakage: games before the week only)
  - model fair line: SRS diff + home-field constant, in betting notation
  - the market's line (ESPN's odds, current week only), and the edge
  - predicted winner + calibrated win probability

Data comes from cfb_games.py (ESPN scoreboard API -- nflverse doesn't cover
college), but the math is the same srs.py/blended_asof_ratings machinery,
reused through predict_week.py's detail_for_games/build_week_table with
college-estimated constants.

Constants below are from backtest_cfb.py's honest walk-forward over
2022-2026 (k=6/shrink=0.7 won the MAE sweep; HFA and std are that run's
estimates). College is higher-variance than the NFL: bigger home field,
wider margin distribution. The backtest picked 70% of winners straight up
(Brier 0.188) -- better than the NFL model's 63%, because college favorites
win more often -- but the same honesty rule applies: nothing here says the
model beats the MARKET, so the edge column is context, not a bet signal.

USAGE
-----
  python predict_cfb_week.py                     # current season + week
  python predict_cfb_week.py --week 6
  python predict_cfb_week.py --team ND           # just that team's game
  python predict_cfb_week.py --log               # append this week to the live log
"""

import argparse

import pandas as pd

from cfb_games import load_cfb_games
from srs import blended_asof_ratings
from predict_week import build_week_table, append_to_log, current_season_week

# From backtest_cfb.py walk-forward 2022-2026 (k/shrink are the MAE-sweep
# winners, HFA/std are that configuration's estimates). Re-estimate if the
# rating model changes.
K_BLEND = 6.0
SHRINK = 0.7
HFA_POINTS = 3.77
MARGIN_STD = 16.66

MODEL_VERSION = "cfb-srs-v1"
LOG_PATH = "cfb_predictions_log.csv"


def main():
    p = argparse.ArgumentParser(description="Weekly SRS predictions sheet, college football (FBS)")
    p.add_argument("--season", type=int, default=None, help="default: current season")
    p.add_argument("--week", type=int, default=None, help="default: current week")
    p.add_argument("--team", type=str, default=None,
                   help="show only games involving this ESPN abbreviation (e.g. ND)")
    p.add_argument("--k", type=float, default=K_BLEND, help="cold-start blend, pseudo-games")
    p.add_argument("--shrink", type=float, default=SHRINK, help="prior-season regression")
    p.add_argument("--cap", type=float, default=None, help="SRS margin cap")
    p.add_argument("--save_csv", action="store_true", help="save the table to a CSV")
    p.add_argument("--log", action="store_true",
                   help="append unplayed games to the live predictions log")
    args = p.parse_args()

    games = load_cfb_games()
    def_season, def_week = current_season_week(games)
    season = args.season or def_season
    week = args.week or def_week

    ratings = blended_asof_ratings(games, [season], cap=args.cap, k=args.k, shrink=args.shrink)
    df, detail = build_week_table(games, season, week, ratings, args.k, args.shrink,
                                  hfa_points=HFA_POINTS, margin_std=MARGIN_STD)

    if args.team:
        t = args.team.upper()
        df = df[df["game"].str.contains(fr"\b{t}\b", regex=True)]
        detail = detail[(detail["away"] == t) | (detail["home"] == t)]
        if df.empty:
            print(f"no {t} game in week {week}")
            return

    print(f"\n=== Week {week}, {season} — college SRS predictions "
          f"(ratings as of before week {week}) ===")
    print(f"SRS blend k={args.k:g}, shrink={args.shrink:g} | HFA {HFA_POINTS:+.1f} | "
          f"win-prob std {MARGIN_STD:.1f} (from 2022-2026 honest backtest)\n")
    print(df.to_string(index=False))
    print("\nedge: model line minus market line, home-positive (+ = model likes "
          "home more than the market). Market lines from ESPN odds, current week only.")
    print("Reminder: the backtest says how well SRS ranks college teams "
          "(70% winners straight up), not that it beats the market.")

    if args.save_csv:
        path = f"cfb_predictions_{season}_week{week:02d}.csv"
        df.to_csv(path, index=False)
        print(f"\nSaved: {path}")

    if args.log:
        append_to_log(detail, path=LOG_PATH, model_version=MODEL_VERSION)


if __name__ == "__main__":
    main()
