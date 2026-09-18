"""
predict_week.py -- the weekly "how good are these teams" sheet (Phase 4).

One row per game for a season/week, played or upcoming:
  - both teams' SRS as-of that week (no leakage: games before the week only)
  - model fair line: SRS diff + home-field constant, in betting notation
  - the market's actual line, and the edge (model - market, home-positive)
  - predicted winner + calibrated win probability

What this sheet IS: a power rating comparison in points, so "how good is
the team I'm betting for / against" is a number you can see.
What it is NOT: a bet signal. Honest walk-forward backtests 2021-2025
(spread AND moneyline) found no edge bucket that clears the 52.4%
breakeven -- see intermediate/qa_log.md. The edge column is context,
not a recommendation.

USAGE
-----
  python predict_week.py                      # current season + week
  python predict_week.py --season 2026 --week 1
  python predict_week.py --week 5 --save_csv
"""

import argparse
import sys

import pandas as pd

from nfl_box_score_analysis import load_games
from srs import blended_asof_ratings
from backtest_srs import normal_cdf

# Estimated from the honest as-of backtest, 2021-2025 (intermediate/qa_log.md).
# Re-estimate if the rating model changes.
HFA_POINTS = 2.2
MARGIN_STD = 13.44


def current_season_week(games):
    """Latest season; the most recent week that has kicked off (so the
    default is 'this week', including in-progress games in it)."""
    season = int(games["season"].max())
    sub = games[games["season"] == season].copy()
    sub["gameday"] = pd.to_datetime(sub["gameday"])
    started = sub[sub["gameday"] <= pd.Timestamp.now()]
    return season, (int(started["week"].max()) if len(started) else 1)


def line_string(home, away, margin_home_pos):
    """Betting notation: favorite -X.X (margin is home-positive points)."""
    if abs(margin_home_pos) < 0.05:
        return "PK"
    fav, pts = (home, margin_home_pos) if margin_home_pos > 0 else (away, -margin_home_pos)
    return f"{fav} -{pts:.1f}"


def build_week_table(games, season, week, ratings, k, shrink):
    sched = games[(games["season"].astype(int) == season) & (games["week"] == week)].copy()
    if sched.empty:
        sys.exit(f"no games found for season {season} week {week}")
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    sched = sched.sort_values(["gameday", "gametime"])

    r = ratings[(season, week)]
    rows = []
    for g in sched.itertuples():
        neutral = getattr(g, "location", "Home") == "Neutral"
        hfa = 0.0 if neutral else HFA_POINTS
        margin = r[g.home_team] - r[g.away_team] + hfa  # home-positive
        wp_home = normal_cdf(margin / MARGIN_STD)
        pred_home = wp_home >= 0.5
        spread = g.spread_line if pd.notna(g.spread_line) else None
        row = {
            "day": g.gameday.strftime("%a %b %-d"),
            "game": f"{g.away_team} @ {g.home_team}" + (" (neutral)" if neutral else ""),
            "away_srs": round(r[g.away_team], 1),
            "home_srs": round(r[g.home_team], 1),
            "model_line": line_string(g.home_team, g.away_team, margin),
            "market_line": (line_string(g.home_team, g.away_team, spread)
                            if spread is not None else "-"),
            "edge": round(margin - spread, 1) if spread is not None else None,
            "pred": g.home_team if pred_home else g.away_team,
            "win_pct": round(100 * max(wp_home, 1 - wp_home)),
            "final": "-",
            "pred_right": "-",
        }
        if pd.notna(g.home_score):  # already played: show what actually happened
            actual = g.home_score - g.away_score
            row["final"] = f"{g.away_score:.0f}-{g.home_score:.0f}"
            row["pred_right"] = "✓" if (actual > 0) == pred_home else "✗"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Weekly SRS predictions sheet")
    p.add_argument("--season", type=int, default=None, help="default: current season")
    p.add_argument("--week", type=int, default=None, help="default: current week")
    p.add_argument("--k", type=float, default=4.0, help="cold-start blend, pseudo-games")
    p.add_argument("--shrink", type=float, default=0.7, help="prior-season regression")
    p.add_argument("--cap", type=float, default=None, help="SRS margin cap")
    p.add_argument("--save_csv", action="store_true", help="save the table to a CSV")
    args = p.parse_args()

    games = load_games()
    def_season, def_week = current_season_week(games)
    season = args.season or def_season
    week = args.week or def_week

    ratings = blended_asof_ratings(games, [season], cap=args.cap, k=args.k, shrink=args.shrink)
    df = build_week_table(games, season, week, ratings, args.k, args.shrink)

    print(f"\n=== Week {week}, {season} — SRS predictions (ratings as of before week {week}) ===")
    print(f"SRS blend k={args.k:g}, shrink={args.shrink:g} | HFA {HFA_POINTS:+.1f} | "
          f"win-prob std {MARGIN_STD:.1f} (from 2021-2025 honest backtest)\n")
    print(df.to_string(index=False))
    print("\nedge: model line minus market line, home-positive (+ = model likes "
          "home more than the market).")
    print("Reminder: no edge bucket cleared the 52.4% breakeven in honest "
          "backtests -- this says how good the teams are, not what to bet.")

    if args.save_csv:
        path = f"predictions_{season}_week{week:02d}.csv"
        df.to_csv(path, index=False)
        print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
