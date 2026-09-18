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
  python predict_week.py --log                # append this week's picks to the live log
"""

import argparse
import os
import sys

import pandas as pd

from nfl_box_score_analysis import load_games
from srs import blended_asof_ratings
from backtest_srs import normal_cdf

# Estimated from the honest as-of backtest, 2021-2025 (intermediate/qa_log.md).
# Re-estimate if the rating model changes.
HFA_POINTS = 2.2
MARGIN_STD = 13.44

# Bump this when ANYTHING about the model changes (constants, blend, new
# features). grade_predictions.py compares versions head-to-head on the
# same weeks -- that's the honest A/B test for whether a tweak helped.
MODEL_VERSION = "srs-v1"

# Append-only live log (Phase 6). Rows are written BEFORE kickoff and never
# edited -- that's what makes live grading unfakeable.
LOG_PATH = "predictions_log.csv"

TEAMS_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/teams.csv"

QB_SUFFIXES = {"II", "III", "IV", "Jr.", "Sr.", "Jr", "Sr."}


def to_pbp_name(full):
    """'Jared Goff' -> 'J.Goff' (nflverse play-by-play passer style).
    Used both for display and for joining games.csv QB names to pbp
    passer stats. Returns None for missing names."""
    if not isinstance(full, str) or not full.strip():
        return None
    parts = full.split()
    if parts[-1] in QB_SUFFIXES:
        parts = parts[:-1]
    return f"{parts[0][0]}.{parts[-1]}"


def team_nicknames():
    """abbr -> nickname (SF -> 49ers) for readable final scores;
    falls back to abbreviations if the lookup can't load."""
    try:
        teams = pd.read_csv(TEAMS_URL)
        latest = teams.sort_values("season").drop_duplicates("team", keep="last")
        return dict(zip(latest["team"], latest["nickname"]))
    except Exception:
        return {}


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


def detail_for_games(sched, ratings):
    """One detail row per game for ANY subset of the schedule (played or
    not): both SRS ratings, predicted margin, win prob, market line, edge.
    Single source of truth for the CLI sheet, the app's Predictions view,
    the app's attach-to-cards mode, and the live log."""
    detail = []
    for g in sched.itertuples():
        r = ratings[(int(g.season), int(g.week))]
        neutral = getattr(g, "location", "Home") == "Neutral"
        hfa = 0.0 if neutral else HFA_POINTS
        margin = r[g.home_team] - r[g.away_team] + hfa  # home-positive
        wp_home = normal_cdf(margin / MARGIN_STD)
        pred_home = wp_home >= 0.5
        spread = g.spread_line if pd.notna(g.spread_line) else None
        played = pd.notna(g.home_score)
        detail.append({
            "season": int(g.season), "week": int(g.week), "game_id": g.game_id,
            "gameday": g.gameday,
            "away": g.away_team, "home": g.home_team,
            "away_srs": round(r[g.away_team], 3), "home_srs": round(r[g.home_team], 3),
            "neutral": neutral, "hfa_applied": hfa,
            "pred_margin": round(margin, 3), "home_win_prob": round(wp_home, 4),
            "spread_line": spread,
            "edge": round(margin - spread, 3) if spread is not None else None,
            "pred_winner": g.home_team if pred_home else g.away_team,
            "away_qb": getattr(g, "away_qb_name", None),
            "home_qb": getattr(g, "home_qb_name", None),
            "away_score": g.away_score if played else None,
            "home_score": g.home_score if played else None,
            "already_played": played,
        })
    return pd.DataFrame(detail)


def build_week_table(games, season, week, ratings, k, shrink, names=None):
    """(display_df, detail_df) for one season/week: the printable sheet,
    plus one row per game with everything the live log/grader needs."""
    names = names or {}
    sched = games[(games["season"].astype(int) == season) & (games["week"] == week)].copy()
    if sched.empty:
        sys.exit(f"no games found for season {season} week {week}")
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    sched = sched.sort_values(["gameday", "gametime"])
    detail = detail_for_games(sched, ratings)

    rows = []
    for d in detail.itertuples():
        row = {
            "day": d.gameday.strftime("%a %b %-d"),
            "game": f"{d.away} @ {d.home}" + (" (neutral)" if d.neutral else ""),
            "away_qb": to_pbp_name(d.away_qb) or "-",
            "home_qb": to_pbp_name(d.home_qb) or "-",
            "away_srs": round(d.away_srs, 1),
            "home_srs": round(d.home_srs, 1),
            "model_line": line_string(d.home, d.away, d.pred_margin),
            "market_line": ("-" if pd.isna(d.spread_line)
                            else line_string(d.home, d.away, d.spread_line)),
            "edge": None if pd.isna(d.edge) else round(d.edge, 1),
            "pred": d.pred_winner,
            "win_pct": round(100 * max(d.home_win_prob, 1 - d.home_win_prob)),
            "final": "-",
            "pred_right": "-",
        }
        if d.already_played:
            row["final"] = (f"{d.away_score:.0f}({names.get(d.away, d.away)})"
                            f"-{d.home_score:.0f}({names.get(d.home, d.home)})")
            row["pred_right"] = ("✓" if (d.home_score > d.away_score)
                                 == (d.pred_winner == d.home) else "✗")
        rows.append(row)
    return pd.DataFrame(rows), detail


def append_to_log(detail, path=LOG_PATH):
    """Phase 6 logging discipline: only games that haven't kicked off yet
    (logging a played game would be post-hoc), and never the same
    (season, week, version, game) twice. Append-only after that."""
    fresh = detail[~detail["already_played"]].drop(columns=["already_played"]).copy()
    skipped = int(detail["already_played"].sum())
    fresh["model_version"] = MODEL_VERSION
    fresh["logged_at"] = pd.Timestamp.now().isoformat(timespec="seconds")

    if os.path.exists(path):
        existing = pd.read_csv(path)
        dup = existing[["season", "week", "game_id", "model_version"]].drop_duplicates()
        fresh = fresh.merge(dup.assign(_dup=1), on=["season", "week", "game_id", "model_version"],
                            how="left")
        dups = int(fresh["_dup"].sum())
        fresh = fresh[fresh["_dup"].isna()].drop(columns=["_dup"])
        fresh.to_csv(path, mode="a", header=False, index=False)
    else:
        dups = 0
        fresh.to_csv(path, index=False)
    print(f"\nLogged {len(fresh)} predictions to {path} (model_version={MODEL_VERSION})"
          + (f" | skipped {skipped} already-played" if skipped else "")
          + (f" | skipped {dups} already-logged" if dups else ""))


def main():
    p = argparse.ArgumentParser(description="Weekly SRS predictions sheet")
    p.add_argument("--season", type=int, default=None, help="default: current season")
    p.add_argument("--week", type=int, default=None, help="default: current week")
    p.add_argument("--k", type=float, default=4.0, help="cold-start blend, pseudo-games")
    p.add_argument("--shrink", type=float, default=0.7, help="prior-season regression")
    p.add_argument("--cap", type=float, default=None, help="SRS margin cap")
    p.add_argument("--save_csv", action="store_true", help="save the table to a CSV")
    p.add_argument("--log", action="store_true",
                   help="append unplayed games to the live predictions log (Phase 6)")
    args = p.parse_args()

    games = load_games()
    def_season, def_week = current_season_week(games)
    season = args.season or def_season
    week = args.week or def_week

    ratings = blended_asof_ratings(games, [season], cap=args.cap, k=args.k, shrink=args.shrink)
    df, detail = build_week_table(games, season, week, ratings, args.k, args.shrink,
                                  names=team_nicknames())

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

    if args.log:
        append_to_log(detail)


if __name__ == "__main__":
    main()
