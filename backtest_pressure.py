"""
Phase 9: the "protection tiebreaker" hypothesis -- when two passing games
are about even, does the team that allows LESS QB pressure win?

Origin: user eyeballing box scores (DET/BUF: BUF 2 sacks/5yds vs DET
4/38) and using pressure as the tiebreaker between evenly-matched QBs
("a team that gives fewer QB pressures usually wins if the two
quarterbacks are about the same"). Phase 7 fitted pressure rate
ADDITIVELY (prate_diff next to srs_diff and epa_diff in one margin
regression) and found no market edge anywhere -- but an additive fit
assumes pressure helps equally in every game. This tests the CONDITIONAL
claim: pressure rate as a tiebreaker, only when the passing games are
even. A different question, so a different harness: no new model, just
slicing already-as-of features, same discipline as backtest_segments.py.

Two "about the same" conditions, both knowable BEFORE kickoff:
  A. |epa_diff| <= t   -- the model's own trailing EPA/db says the two
     passing attacks are even (TEAM-level: includes line/weapons, so a
     good-QB-behind-a-bad-line team and its mirror both read as "even";
     starter-level Phase 7b features are the stricter follow-up if this
     shows life). t in {0.05, 0.08, 0.12} -- median |epa_diff| is ~0.10.
  B. |spread_line| <= 3 -- the MARKET says the teams are even. The
     bettable version: no QB-judging required; when Vegas hangs a near
     pick'em, does trailing pressure rate pick the side?

Graded per condition:
  win%   -- the user's literal claim ("usually win")
  ATS    -- cover rate betting the lower-prate team vs the closing spread
            (actual spread odds from games.csv, -110 fallback), vs the
            52.4% breakeven
  dose-response -- win%/ATS bucketed by |prate_diff| size; a real
            tiebreaker should get STRONGER as the protection gap widens,
            noise bounces around

Honesty caveats:
  - Features are as-of (trailing, cold-start blended -- built by
    backtest_blend.trailing_features, cached in blend_features_cache.csv),
    so no leakage. But the condition thresholds were eyeballed from the
    data distribution: any bucket that clears breakeven is a re-test
    candidate on future seasons, not a rule. Multiple cuts x one
    breakeven line = some bucket will clear by chance.
  - Phase 7b's additive fit found starter-level QB detail a wash; that
    LOWERS the prior that a conditional QB-based slice beats the close.
  - Eval restricted to 2021-2025: 2020 rows lean on the league-fallback
    prior (no 2019 pbp in this pipeline), so their "as-of" features are
    the weakest.

USAGE
-----
  python backtest_pressure.py                  # full report
  python backtest_pressure.py --even 0.08      # dose-response condition A threshold
  python backtest_pressure.py --spread 3       # dose-response condition B threshold
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from backtest_srs import BREAKEVEN_COVER_PCT
from backtest_totals import profit_100

CACHE = "blend_features_cache.csv"
EVEN_THRESHOLDS = [0.05, 0.08, 0.12]
SPREAD_THRESHOLDS = [2.5, 3.0, 4.0]
PRATE_BANDS = [(0, 0.015), (0.015, 0.03), (0.03, 0.05), (0.05, 99)]


def load_universe():
    if not os.path.exists(CACHE):
        sys.exit(f"{CACHE} missing -- build it with: python backtest_moneyline.py --rebuild")
    df = pd.read_csv(CACHE)
    df = df[(df["season"] >= 2021) & (df["season"] <= 2025)].copy()
    df = df.dropna(subset=["home_score", "away_score", "spread_line",
                           "epa_diff", "prate_diff"])
    df = df[df["prate_diff"] != 0]  # exact tie = no signal (week-1 fallbacks)
    # prate_diff = home_prate - away_prate (as-of). Lower prate = better
    # protection -> pick home when prate_diff < 0, away when > 0.
    df["pick_home"] = df["prate_diff"] < 0
    df["margin"] = df["home_score"] - df["away_score"]
    df["picked_won"] = np.where(df["pick_home"], df["margin"] > 0, df["margin"] < 0)
    beat = df["margin"] - df["spread_line"]  # >0: home covers (spread_line home-favorable)
    df["picked_covers"] = np.where(df["pick_home"], beat > 0, beat < 0)
    df["push"] = beat == 0
    odds = np.where(df["pick_home"], df["home_spread_odds"], df["away_spread_odds"])
    df["odds"] = pd.to_numeric(pd.Series(odds, index=df.index), errors="coerce").fillna(-110.0)
    df["profit"] = [0.0 if p else profit_100(o, w)
                    for p, o, w in zip(df["push"], df["odds"], df["picked_covers"])]
    return df


def line(df, label):
    decided = df[~df["push"]]
    if len(decided) == 0:
        return f"  {label}: no games"
    win_pct = 100 * df["picked_won"].mean()
    ats = 100 * decided["picked_covers"].mean()
    roi = 100 * df["profit"].sum() / (100 * len(df))
    verdict = "ABOVE breakeven" if ats >= BREAKEVEN_COVER_PCT else "below breakeven"
    return (f"  {label}: win {win_pct:5.1f}% | ATS {ats:5.1f}% "
            f"| ROI {roi:+5.1f}% | n={len(df)} ({len(decided)} decided)  -- {verdict}")


def dose_response(df, label):
    print(f"\n  dose-response within {label} (does a BIGGER protection gap win more?):")
    for lo, hi in PRATE_BANDS:
        band = df[df["prate_diff"].abs().between(lo, hi, inclusive="left")]
        tag = f"|prate diff| {lo:g}-{hi:g}" if hi < 99 else f"|prate diff| {lo:g}+"
        print("   " + line(band, tag))


def main():
    p = argparse.ArgumentParser(description="Phase 9: pressure-as-tiebreaker backtest")
    p.add_argument("--even", type=float, default=0.08,
                   help="|epa_diff| threshold for the dose-response cut")
    p.add_argument("--spread", type=float, default=3.0,
                   help="|spread_line| threshold for the dose-response cut")
    p.add_argument("--save_csv", action="store_true")
    args = p.parse_args()

    df = load_universe()
    print(f"\n=== Protection tiebreaker -- {int(df['season'].min())}-{int(df['season'].max())} ===")
    print(f"games: {len(df)} | pick = team with lower as-of pressure-allowed rate")
    print(f"ATS breakeven at -110: {BREAKEVEN_COVER_PCT}%\n")

    print("Baseline (no condition -- does pressure pick winners at all?):")
    print(line(df, "all games"))

    print("\nCondition A: passing games even by the MODEL (|epa_diff| <= t):")
    for t in EVEN_THRESHOLDS:
        print(line(df[df["epa_diff"].abs() <= t], f"|epa diff| <= {t:g}"))
    print(line(df[df["epa_diff"].abs() > EVEN_THRESHOLDS[-1]],
               f"|epa diff| >  {EVEN_THRESHOLDS[-1]:g} (control: QBs NOT even)"))

    print("\nCondition B: teams even by the MARKET (|spread| <= t):")
    for t in SPREAD_THRESHOLDS:
        print(line(df[df["spread_line"].abs() <= t], f"|spread| <= {t:g}"))

    both = df[(df["epa_diff"].abs() <= args.even) & (df["spread_line"].abs() <= args.spread)]
    print(f"\nBoth even (|epa diff| <= {args.even:g} AND |spread| <= {args.spread:g}):")
    print(line(both, "both"))

    dose_response(df[df["epa_diff"].abs() <= args.even], f"|epa diff| <= {args.even:g}")
    dose_response(df[df["spread_line"].abs() <= args.spread], f"|spread| <= {args.spread:g}")

    print("\nRead: win% answers the literal question; ATS/ROI answers whether")
    print("the market already prices it. Thresholds were picked from the data,")
    print("so treat any above-breakeven bucket as a re-test candidate, not a rule.")

    if args.save_csv:
        df.to_csv("backtest_pressure_games.csv", index=False)
        print("\nSaved: backtest_pressure_games.csv")


if __name__ == "__main__":
    main()
