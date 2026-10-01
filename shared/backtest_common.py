"""
backtest_common.py -- the sport-agnostic half of backtest_srs.py.

Everything here works off a games table shaped like nflverse's games.csv
(game_id, season, week, home_team, away_team, home_score, away_score,
location, spread_line), so the NFL backtest (nfl/backtest_srs.py) and the
college one (ncaa/backtest_cfb.py) share one implementation. The only
leakage-sensitive piece anywhere is the caller's rating_fn(season, week).

nfl/backtest_srs.py re-exports these names, so existing
`from backtest_srs import ...` lines keep working.
"""

import math

import pandas as pd

# Standard -110 odds on both sides of a spread bet: you must win ~52.38%
# of bets just to break even after the vig. Named per the plan so no
# bucket is ever silently compared against 50%.
BREAKEVEN_COVER_PCT = 52.4

EDGE_BUCKETS = [(0, 1), (1, 2), (2, 3), (3, 99)]  # |edge|, points


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


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
