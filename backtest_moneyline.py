"""
Moneyline prob-edge backtest: does the model's win PROBABILITY beat the
market's moneyline-implied probability?

Second branch of the decided betting rule (the moneyline twin of the
spread edge test):
    prob edge = model win prob - moneyline-implied prob
The implied probability IS the breakeven for that bet (-250 -> 71.4%,
+250 -> 28.6%), so a bucket only has a real edge if its actual win rate
clears its mean implied rate.

Two rules graded, because they answer different questions:
  Rule A (price shop): per game bet the side with the LARGER edge --
       this systematically lands on underdogs the model rates above the
       market, i.e. it SELECTS ON MODEL ERROR (the ATS inversion's twin).
  Rule B (choose prediction at a good price): bet the model's predicted
       WINNER, but only when its own price edge clears the threshold.

Calibration is measured on the predicted-winner side with NO edge
selection -- selecting by edge contaminates it with disagreement error.

Probabilities: Phase 7 M1 blend (expanding-window fit, features as-of).
Ties refund (standard 2-way ML rules) and are excluded like ATS pushes.
The expensive feature frame is cached to blend_features_cache.csv
(--rebuild to regenerate).

USAGE
-----
  python backtest_moneyline.py                  # eval 2022-2025
  python backtest_moneyline.py --rule 5 --rebuild
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from nfl_box_score_analysis import load_games
from srs import blended_asof_ratings
from backtest_srs import normal_cdf
from backtest_blend import trailing_features, assemble_games, fit_predict, FEATURES

PROB_BUCKETS = [(0, 3), (3, 5), (5, 8), (8, 100)]  # probability-point edge
CACHE = "blend_features_cache.csv"


def implied_prob(ml):
    """American odds -> breakeven probability (raw implied, vig included --
    that's the number a real bet has to beat)."""
    ml = ml.astype(float)
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def profit_per_100(ml, won):
    """Flat $100 stake ROI accounting for American odds."""
    return np.where(won, np.where(ml < 0, 100 * 100 / -ml, ml.astype(float)), -100.0)


def build_feature_frame(args):
    """The expensive part (pbp downloads); cached to CSV."""
    if os.path.exists(CACHE) and not args.rebuild:
        print(f"  loaded cached features from {CACHE}", file=sys.stderr)
        return pd.read_csv(CACHE)
    games = load_games()
    lo, hi = min(args.eval_seasons), max(args.eval_seasons)
    tg = trailing_features(games, list(range(lo - 2, hi + 1)), k=args.k, shrink=args.shrink)
    feat_seasons = sorted(tg["season"].unique())
    srs_ratings = blended_asof_ratings(games, feat_seasons, cap=args.cap,
                                       k=args.k, shrink=args.shrink)
    g = assemble_games(games, feat_seasons, srs_ratings, tg)
    g.to_csv(CACHE, index=False)
    print(f"  cached features to {CACHE}", file=sys.stderr)
    return g


def build_bet_frame(pred):
    """Per game: both sides' implied prob, edge, moneyline, and outcome."""
    g = pred.dropna(subset=["home_moneyline", "away_moneyline"]).copy()
    g = g[g["actual_margin"] != 0]  # ties refund under 2-way ML rules
    g["impl_h"] = implied_prob(g["home_moneyline"])
    g["impl_a"] = implied_prob(g["away_moneyline"])
    g["edge_h"] = g["home_win_prob"] - g["impl_h"]
    g["edge_a"] = (1 - g["home_win_prob"]) - g["impl_a"]
    g["pred_home"] = g["pred_margin"] > 0
    return g


def grade_rule(g, bet_home_mask, label):
    """Grade one rule: bet where bet_home_mask picks home (else away),
    restricted to rows where the rule bets at all (mask pre-filtered by
    caller via edge threshold). Returns a bets frame."""
    b = pd.DataFrame({
        "season": g["season"],
        "model_prob": np.where(bet_home_mask, g["home_win_prob"], 1 - g["home_win_prob"]),
        "implied_prob": np.where(bet_home_mask, g["impl_h"], g["impl_a"]),
        "ml": np.where(bet_home_mask, g["home_moneyline"].astype(float),
                       g["away_moneyline"].astype(float)),
        "won": np.where(bet_home_mask, g["home_won"] == 1, g["home_won"] == 0),
    })
    b["edge"] = (b["model_prob"] - b["implied_prob"]) * 100
    b["profit"] = profit_per_100(b["ml"], b["won"])
    return b


def bucket_line(mask, bets, label):
    sub = bets[mask]
    if len(sub) == 0:
        return f"  {label}: no bets"
    breakeven = 100 * sub["implied_prob"].mean()
    actual = 100 * sub["won"].mean()
    roi = 100 * sub["profit"].sum() / (100 * len(sub))
    verdict = "ABOVE breakeven" if actual > breakeven else "below breakeven"
    return (f"  {label}: n={len(sub):4d} | breakeven {breakeven:5.1f}% | "
            f"actual {actual:5.1f}% | ROI {roi:+5.1f}%  -- {verdict}")


def report_rule(bets, rule_edge, title):
    print(f"\n{title}")
    print(bucket_line(bets["edge"] > -999, bets, "all bets this rule makes"))
    for lo_e, hi_e in PROB_BUCKETS:
        hi_lab = f"{hi_e}" if hi_e < 100 else "+"
        print(bucket_line((bets["edge"] >= lo_e) & (bets["edge"] < hi_e), bets,
                          f"edge {lo_e}-{hi_lab} pt"))
    print(bucket_line(bets["edge"] >= rule_edge, bets, f"RULE: edge >= {rule_edge:g}"))


def main():
    p = argparse.ArgumentParser(description="Moneyline prob-edge backtest (M1 blend probabilities)")
    p.add_argument("--eval_seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    p.add_argument("--k", type=float, default=4.0)
    p.add_argument("--shrink", type=float, default=0.7)
    p.add_argument("--cap", type=float, default=None)
    p.add_argument("--rule", type=float, default=5.0, help="prob-edge threshold (points)")
    p.add_argument("--rebuild", action="store_true", help="rebuild cached feature frame")
    args = p.parse_args()

    g = build_feature_frame(args)
    pred, _ = fit_predict(g, args.eval_seasons, FEATURES)
    pred["home_win_prob"] = [normal_cdf(m / s) for m, s in zip(pred["pred_margin"], pred["win_std"])]
    g = build_bet_frame(pred)

    lo, hi = min(args.eval_seasons), max(args.eval_seasons)
    print(f"\n=== Moneyline prob-edge backtest — M1 blend probabilities ===")
    print(f"Seasons: {lo}-{hi} | games with moneylines: {len(g)} (ties refunded)\n")

    # Calibration on the PREDICTED WINNER, no edge selection
    pw_prob = np.maximum(g["home_win_prob"], 1 - g["home_win_prob"])
    pw_won = (g["pred_home"] == (g["home_won"] == 1))
    print("Calibration (predicted winner, no edge selection):")
    for plo, phi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        m = (pw_prob >= plo) & (pw_prob < phi)
        if m.sum():
            print(f"  {int(plo*100)}-{int(min(phi,1)*100)}%: model {100*pw_prob[m].mean():5.1f}% "
                  f"| actual {100*pw_won[m].mean():5.1f}% | n={int(m.sum())}")

    # Rule A: price shop -- bet whichever side has the larger edge
    bet_home_a = g["edge_h"] >= g["edge_a"]
    bets_a = grade_rule(g, bet_home_a, "A")
    report_rule(bets_a, args.rule,
                "Rule A — price shop (larger edge; lands on model-favored UNDERDOGS):")

    # Rule B: choose prediction -- bet the predicted winner, only at a price
    bets_b = grade_rule(g, g["pred_home"], "B")
    report_rule(bets_b, args.rule,
                "Rule B — choose prediction at a good price (predicted winner only):")


if __name__ == "__main__":
    main()
