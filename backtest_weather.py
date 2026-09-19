"""
Phase 10: weather-conditioned totals -- which weather bands actually lean
under, and does adding weather to the totals model find edge vs the close?

Plan.md segment-test item 4 (unblocked by Phase 8, backtest_totals.py).
User's question: "what parameters can I get more data to make a good
educated bet on the under?" Two separate answers:

1. RAW CUTS (descriptive, no model -- 2000-2025, every game with a
   closing total): bucket by weather band and report the under-rate and
   the market's average miss in points. If the market fully prices a
   band, its under-rate sits at ~50% no matter how strongly weather
   affects scoring -- so these cuts measure PRICING, not weather's
   effect on points. Bands: wind (outdoor), temp (outdoor), roof,
   outdoor surface, plus two non-weather "under lore" cuts for contrast
   (Thursday night, division games).

2. MODEL TEST (walk-forward, no leakage -- 2021-2025 graded): Phase 8's
   pace model vs the same model + three weather features, both refit by
   expanding-window OLS on identical games:
     outdoor    = 1 if roof is outdoors/open else 0
     wind_out   = wind mph if outdoor else 0
     temp_c_out = (temp - 68) if outdoor else 0   (68 degF ~ dome climate,
                  so the outdoor flag reads as "vs a climate-controlled
                  game" and the two slopes only move outdoor games)
   Missing outdoor wind/temp (<5% of outdoor rows; 2022 has a source
   gap) imputed to the all-time outdoor mean / 68 degF -- always toward
   the dome-like center, never inventing signal.

Data caveat (from the nflverse schedules file): temp/wind are a single
game-day snapshot, airport-station level, not stadium-hour precise;
domes are NaN by design. temp/wind coverage is complete-ish for outdoor
games back to 1999.

USAGE
-----
  python backtest_weather.py                     # raw cuts + model comparison
  python backtest_weather.py --eval_seasons 2024 2025
  python backtest_weather.py --save_csv
"""

import argparse
import sys

import numpy as np
import pandas as pd

from nfl_box_score_analysis import load_games, build_long_results
from backtest_srs import BREAKEVEN_COVER_PCT
from backtest_totals import (FEATURES as PACE_FEATURES, trailing_pace,
                             assemble_games, fit_predict, grade, cover_line,
                             EDGE_BUCKETS)

WX_FEATURES = PACE_FEATURES + ["outdoor", "wind_out", "temp_c_out"]
WIND_MEAN = 8.4      # all-time outdoor mean, for imputing the rare missing row
DOME_TEMP = 68.0     # climate-controlled reference point for temp_c_out

WIND_BANDS = [(0, 6), (6, 11), (11, 16), (16, 99)]
TEMP_BANDS = [(-99, 32), (33, 45), (46, 60), (61, 75), (76, 99)]


def encode_weather(g):
    """Add outdoor / wind_out / temp_c_out. Rows with missing roof but a
    recorded wind count as outdoor."""
    g = g.copy()
    g["outdoor"] = (g["roof"].isin(["outdoors", "open"])
                    | (g["roof"].isna() & g["wind"].notna())).astype(float)
    wind = g["wind"].fillna(WIND_MEAN)
    temp = g["temp"].fillna(DOME_TEMP)
    g["wind_out"] = np.where(g["outdoor"] == 1, wind, 0.0)
    g["temp_c_out"] = np.where(g["outdoor"] == 1, temp - DOME_TEMP, 0.0)
    return g


def cut_line(df, label):
    decided = df[df["actual_total"] != df["total_line"]]
    if len(decided) < 30:
        return f"  {label:<22}: n={len(df)} (too few)"
    under_pct = 100 * (decided["actual_total"] < decided["total_line"]).mean()
    miss = (df["actual_total"] - df["total_line"]).mean()
    note = " <-- leans UNDER past breakeven" if under_pct >= BREAKEVEN_COVER_PCT else ""
    return (f"  {label:<22}: {under_pct:5.1f}% under | market miss {miss:+5.1f} pts "
            f"| n={len(df)}{note}")


def raw_cuts(games):
    g = games.dropna(subset=["home_score", "away_score", "total_line"]).copy()
    g = g[g["season"].between(2000, 2025)]
    g["actual_total"] = g["home_score"] + g["away_score"]
    g = encode_weather(g)
    out = g[g["outdoor"] == 1]

    print(f"\n=== Weather raw cuts, 2000-2025 ({len(g)} games with closing totals) ===")
    print("under-rate = how often a blind UNDER bet cashes in that band;")
    print("market miss = avg(actual - line), negative means the market posts")
    print("the band too high. ~50% under = the band is fully priced.\n")

    print("wind, outdoor games only:")
    for lo, hi in WIND_BANDS:
        band = out[out["wind"].between(lo, hi - 1e-9)] if hi < 99 else out[out["wind"] >= lo]
        print(cut_line(band, f"  wind {lo}-{hi if hi < 99 else '+'} mph"))

    print("\ntemperature, outdoor games only:")
    for lo, hi in TEMP_BANDS:
        band = out[(out["temp"] >= lo) & (out["temp"] <= hi)]
        label = f"  temp <={hi}F" if lo == -99 else f"  temp {lo}-{hi}F"
        print(cut_line(band, label))

    print("\nroof / surface:")
    print(cut_line(g[g["outdoor"] == 0], "  indoor (dome/closed)"))
    print(cut_line(out, "  outdoor"))
    print(cut_line(out[out["surface"] == "grass"], "  outdoor grass"))
    print(cut_line(out[out["surface"].isin(["fieldturf", "astroturf", "matrixturf",
                                            "sportturf", "a_turf"])],
                   "  outdoor turf"))

    print("\nnon-weather 'under lore', for contrast:")
    print(cut_line(g[g["weekday"] == "Thursday"], "  Thursday games"))
    print(cut_line(g[g["div_game"] == 1], "  division games"))
    return g


def compare_report(bets_pace, bets_wx, coefs_wx, eval_seasons, rule_edge=2.0):
    print(f"\n=== Walk-forward ATS: pace-only vs pace+weather, "
          f"{min(eval_seasons)}-{max(eval_seasons)} ===")
    print(f"games graded: {len(bets_pace)} | breakeven {BREAKEVEN_COVER_PCT}%")
    print("\nFitted weather coefficients (expanding window; wind in pts/mph,")
    print("temp_c in pts/degF-from-68, outdoor in pts vs a dome game):")
    for S, co in coefs_wx.items():
        print(f"  {S}: outdoor={co['outdoor']:+.2f}, wind={co['wind_out']:+.3f}, "
              f"temp={co['temp_c_out']:+.3f}  (train_n={co['train_n']})")

    print(f"\n{'|edge| bucket':<14}{'pace-only':<38}{'pace+weather'}")
    for lo, hi in EDGE_BUCKETS:
        label = f"{lo}-{hi if hi < 99 else '+'} pt" if hi < 99 else f"{lo}+ pt"
        m = lambda b: (b["abs_edge"] >= lo) & (b["abs_edge"] < hi)  # noqa: E731
        print(f"{label:<14}{cover_line(m(bets_pace), bets_pace, '').strip():<38}"
              f"{cover_line(m(bets_wx), bets_wx, '').strip()}")

    print(f"\nBetting rule (|edge| >= {rule_edge:g}):")
    print(" pace-only : " + cover_line(bets_pace["abs_edge"] >= rule_edge,
                                       bets_pace, "pace").strip())
    print(" pace+wx   : " + cover_line(bets_wx["abs_edge"] >= rule_edge,
                                       bets_wx, "wx").strip())

    print("\nSame buckets, UNDER bets only (pace+weather model):")
    for lo, hi in EDGE_BUCKETS:
        label = f"{lo}-{hi if hi < 99 else '+'} pt" if hi < 99 else f"{lo}+ pt"
        m = (bets_wx["abs_edge"] >= lo) & (bets_wx["abs_edge"] < hi) & (~bets_wx["bet_over"])
        print(cover_line(m, bets_wx, label))


EVAL_DEFAULT = [2021, 2022, 2023, 2024, 2025]


def main():
    p = argparse.ArgumentParser(description="Phase 10: weather-conditioned totals")
    p.add_argument("--eval_seasons", nargs="+", type=int, default=EVAL_DEFAULT)
    p.add_argument("--k", type=float, default=4.0)
    p.add_argument("--shrink", type=float, default=0.7)
    p.add_argument("--rule", type=float, default=2.0)
    p.add_argument("--save_csv", action="store_true")
    args = p.parse_args()

    print("  downloading nflverse games.csv ...", file=sys.stderr)
    games = load_games()

    raw_cuts(games)

    # Walk-forward model comparison (Phase 8 harness, identical games)
    train_from = 2000
    long = build_long_results(games, list(range(train_from - 1,
                                                max(args.eval_seasons) + 1)))
    pace = trailing_pace(long, k=args.k, shrink=args.shrink)
    g = assemble_games(games, list(range(train_from, max(args.eval_seasons) + 1)), pace)
    g = encode_weather(g)

    pred_pace, _ = fit_predict(g, args.eval_seasons, PACE_FEATURES)
    pred_wx, coefs_wx = fit_predict(g, args.eval_seasons, WX_FEATURES)
    bets_pace, bets_wx = grade(pred_pace), grade(pred_wx)
    compare_report(bets_pace, bets_wx, coefs_wx, args.eval_seasons, args.rule)

    print("\nNo leakage: pace features as-of (Phase 8), weather is pre-game")
    print("knowable, weights fit only on seasons < the graded one. Same caveat")
    print("as every script here: buckets were eyeballed, so one clearing")
    print("breakeven is a re-test candidate, not a rule.")

    if args.save_csv:
        bets_wx.to_csv("backtest_weather_games.csv", index=False)
        print("\nSaved: backtest_weather_games.csv")


if __name__ == "__main__":
    main()
