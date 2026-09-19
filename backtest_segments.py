"""
Segment tests on the M1 blend's already-graded predictions: instead of a new
model, slice the SAME graded games by context columns games.csv already
carries but the model doesn't use (div_game, away_rest/home_rest) --
"is the no-edge-anywhere conclusion uniform, or hiding a pocket?"

Cheapest tier of the post-Phase-7b exploration backlog (see
intermediate/plan.md): no new features, no refit beyond what
backtest_moneyline.py's cached feature frame + M1 already produced.

Honesty note: this is exactly the kind of post-hoc slicing that invites
multiple-comparisons noise -- a handful of buckets here are being eyeballed
against one breakeven line with no correction. Treat any bucket that clears
breakeven as a LEAD to re-test on the next holdout season, not a rule to bet
today.

USAGE
-----
  python backtest_segments.py                  # eval 2022-2025 (default cache)
  python backtest_segments.py --rebuild
"""

import argparse
from argparse import Namespace

from backtest_blend import fit_predict, grade_model, FEATURES
from backtest_moneyline import build_feature_frame
from backtest_srs import BREAKEVEN_COVER_PCT


def segment_line(ats, mask, label):
    sub = ats[mask]
    n = len(sub)
    if n == 0:
        return f"  {label}: no bets"
    cover = 100 * sub["bet_won"].mean()
    verdict = "ABOVE breakeven" if cover >= BREAKEVEN_COVER_PCT else "below breakeven"
    return f"  {label}: n={n:4d} | cover {cover:5.1f}%  -- {verdict}"


def win_acc_line(df, mask, label):
    sub = df[mask]
    n = len(sub)
    if n == 0:
        return f"  {label}: no games"
    return f"  {label}: n={n:4d} | win acc {100 * sub['correct'].mean():5.1f}%"


def main():
    p = argparse.ArgumentParser(description="Segment M1's graded games by div_game and rest")
    p.add_argument("--eval_seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    p.add_argument("--k", type=float, default=4.0)
    p.add_argument("--shrink", type=float, default=0.7)
    p.add_argument("--cap", type=float, default=None)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()

    g = build_feature_frame(Namespace(eval_seasons=args.eval_seasons, k=args.k,
                                      shrink=args.shrink, cap=args.cap, rebuild=args.rebuild))
    pred, _ = fit_predict(g, args.eval_seasons, FEATURES)
    df, ats = grade_model(pred)
    rest_diff = ats["home_rest"] - ats["away_rest"]
    rest_diff_full = df["home_rest"] - df["away_rest"]

    print(f"\n=== Segment tests on M1 blend, {min(args.eval_seasons)}-{max(args.eval_seasons)} "
          f"(breakeven {BREAKEVEN_COVER_PCT}%) ===")
    print(f"Graded games: {len(df)} | ATS bets (edge != 0, spread known): {len(ats)}\n")

    print("Divisional game (win accuracy):")
    print(win_acc_line(df, df["div_game"] == 1, "division"))
    print(win_acc_line(df, df["div_game"] == 0, "non-division"))
    print("\nDivisional game (ATS cover rate):")
    print(segment_line(ats, ats["div_game"] == 1, "division"))
    print(segment_line(ats, ats["div_game"] == 0, "non-division"))

    print("\nRest-differential (ATS cover rate, home_rest - away_rest):")
    print(segment_line(ats, rest_diff <= -4, "home rest disadvantage (<=-4 days)"))
    print(segment_line(ats, (rest_diff > -4) & (rest_diff < 4), "roughly even rest (-3..3 days)"))
    print(segment_line(ats, rest_diff >= 4, "home rest advantage (>=4 days)"))

    print("\nNamed rest situations (ATS cover rate):")
    print(segment_line(ats, ats["home_rest"] <= 6, "home team on short week (<=6 days)"))
    print(segment_line(ats, ats["away_rest"] <= 6, "away team on short week (<=6 days)"))
    print(segment_line(ats, ats["home_rest"] >= 12, "home team off bye (>=12 days)"))
    print(segment_line(ats, ats["away_rest"] >= 12, "away team off bye (>=12 days)"))

    print("\nNamed rest situations (win accuracy, larger n than ATS since no "
          "spread-known filter):")
    print(win_acc_line(df, df["home_rest"] <= 6, "home team on short week"))
    print(win_acc_line(df, df["away_rest"] <= 6, "away team on short week"))
    print(win_acc_line(df, df["home_rest"] >= 12, "home team off bye"))
    print(win_acc_line(df, df["away_rest"] >= 12, "away team off bye"))

    print("\nCaution: small-n buckets here are eyeballed against one breakeven "
          "line with no multiple-comparisons correction -- a bucket clearing "
          "52.4% is a lead to re-test on the next holdout season, not a rule "
          "to bet today.")


if __name__ == "__main__":
    main()
