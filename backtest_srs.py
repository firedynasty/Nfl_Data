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
import math
import sys

import pandas as pd

from nfl_box_score_analysis import load_games, build_long_results
from srs import game_margins, solve_srs, blended_asof_ratings

# Standard -110 odds on both sides of a spread bet: you must win ~52.38%
# of bets just to break even after the vig. Named per the plan so no
# bucket is ever silently compared against 50%.
BREAKEVEN_COVER_PCT = 52.4

EDGE_BUCKETS = [(0, 1), (1, 2), (2, 3), (3, 99)]  # |edge|, points


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def season_total_ratings(games, seasons, cap=None):
    """The leaky stub: one rating per team per SEASON, computed from every
    game in that season including the ones being predicted. Phase 2
    replaces this with week-as-of ratings behind the same shape."""
    long = build_long_results(games, list(seasons))
    return {s: solve_srs(game_margins(long, s, cap=cap))[0] for s in seasons}


def estimate_hfa(season_games):
    """Home-field constant from data, not guessed: mean of
    (actual home margin - SRS-only predicted margin) over non-neutral
    games. Neutral-site games get HFA 0 at prediction time."""
    df = season_games[~season_games["neutral"]]
    return (df["actual_margin"] - df["srs_diff"]).mean()


def collect_predictions(games, seasons, rating_fn):
    """One row per played game with predictions and grading columns.
    rating_fn(season, week) -> {team: rating} is the only leakage-sensitive
    piece; everything downstream of it is identical across versions."""
    g = games.dropna(subset=["home_score", "away_score"]).copy()
    g["season"] = g["season"].astype(int)
    g = g[g["season"].isin(list(seasons))]
    # nflverse `location`: "Home" or "Neutral" (Super Bowls, int'l games)
    g["neutral"] = g.get("location", "Home").eq("Neutral")

    rows = []
    for season in sorted(g["season"].unique()):
        gs = g[g["season"] == season]
        for week in sorted(gs["week"].unique()):
            rating = rating_fn(season, week)
            gw = gs[gs["week"] == week]
            for _, r in gw.iterrows():
                rows.append({
                    "season": season, "week": week, "game_id": r["game_id"],
                    "away": r["away_team"], "home": r["home_team"],
                    "srs_diff": rating[r["home_team"]] - rating[r["away_team"]],
                    "neutral": r["neutral"],
                    "actual_margin": r["home_score"] - r["away_score"],
                    "spread_line": r["spread_line"],
                    "home_won": int(r["home_score"] > r["away_score"]),
                    "tied": int(r["home_score"] == r["away_score"]),
                })
    return pd.DataFrame(rows)


def grade(df, hfa, rule_edge):
    """Add HFA, edge, win prob, and all grading columns; return (df, std)."""
    df = df.copy()
    df["hfa_applied"] = df["neutral"].map({True: 0.0, False: hfa})
    df["pred_margin"] = df["srs_diff"] + df["hfa_applied"]

    # std of margin error, estimated from this backtest's own residuals --
    # the Phase 3/5 circular calibration (also feeds Brier win probs).
    std = (df["actual_margin"] - df["pred_margin"]).std()
    df["home_win_prob"] = df["pred_margin"].map(lambda m: normal_cdf(m / std))
    df["brier"] = (df["home_win_prob"] - df["home_won"]) ** 2

    df["pred_home_wins"] = df["pred_margin"] > 0
    df["correct"] = df["pred_home_wins"] == df["home_won"].astype(bool)

    # ATS: edge = predicted margin - spread (both home-positive, the
    # verified convention). Bet the side the edge points to.
    df["edge"] = df["pred_margin"] - df["spread_line"]
    df["abs_edge"] = df["edge"].abs()
    ats = df.dropna(subset=["spread_line"]).copy()
    ats = ats[ats["edge"] != 0]  # edge == 0: no opinion, no bet
    ats["bet_home"] = ats["edge"] > 0
    margin_vs_spread = ats["actual_margin"] - ats["spread_line"]
    ats["ats_result"] = "push"
    ats.loc[margin_vs_spread > 0, "ats_result"] = "home_covers"
    ats.loc[margin_vs_spread < 0, "ats_result"] = "away_covers"
    ats["bet_won"] = (
        ((ats["bet_home"]) & (ats["ats_result"] == "home_covers"))
        | ((~ats["bet_home"]) & (ats["ats_result"] == "away_covers"))
    )
    return df, ats, std


def cover_line(mask, ats, label):
    sub = ats[mask]
    decided = sub[sub["ats_result"] != "push"]
    if len(decided) == 0:
        return f"  {label}: no bets"
    pct = 100 * decided["bet_won"].mean()
    pushes = len(sub) - len(decided)
    verdict = "ABOVE breakeven" if pct >= BREAKEVEN_COVER_PCT else "below breakeven"
    push_note = f", {pushes} push" if pushes else ""
    return f"  {label}: {pct:5.1f}%  (n={len(decided)}{push_note})  -- {verdict}"


def report(df, ats, seasons, hfa_text, std_text, rule_edge, mode, footer=None):
    n = len(df)
    acc = 100 * df["correct"].mean()
    home_base = 100 * df["home_won"].mean()
    print(f"\n=== SRS walk-forward backtest — {mode} ===")
    print(f"Seasons: {min(seasons)}-{max(seasons)} | games graded: {n} | "
          f"neutral-site: {int(df['neutral'].sum())}")
    print(f"{hfa_text} | {std_text}\n")

    print(f"Win accuracy: {acc:.1f}%   (home-always baseline: {home_base:.1f}%)")
    print(f"Brier score:  {df['brier'].mean():.3f}\n")

    print(f"ATS cover rate by |edge| bucket (bet = side the edge points to; "
          f"breakeven {BREAKEVEN_COVER_PCT}%):")
    for lo, hi in EDGE_BUCKETS:
        label = f"{lo}-{hi if hi < 99 else '+'} pt" if hi < 99 else f"{lo}+  pt"
        print(cover_line((ats["abs_edge"] >= lo) & (ats["abs_edge"] < hi), ats, label))
    print(f"\nBetting rule (|edge| >= {rule_edge:g}):")
    print(cover_line(ats["abs_edge"] >= rule_edge, ats, f"|edge| >= {rule_edge:g}"))

    print("\nPer-season win accuracy:")
    per = df.groupby("season")["correct"].mean() * 100
    print("  " + "  ".join(f"{s}: {v:.1f}%" for s, v in per.items()))
    if footer:
        print(footer)
    elif "LEAKY" in mode:
        print("\nREMINDER: season-total ratings saw the games they predict -- "
              "these numbers are the OPTIMISTIC ceiling, not the truth.")
    else:
        print("\nNo leakage: every rating uses only games before the predicted "
              "week. (HFA/std are still fit on this window -- the plan's "
              "circular calibration; blend constants k/shrink untuned.)")


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
