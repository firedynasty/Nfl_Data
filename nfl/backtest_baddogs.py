"""
Backtest of the Action Network "bad last season, small dog" system,
reverse-engineered from the system's actual published picks.

The pasted system description claims: dog +1 to +3.5, total under ~44,
no Thursday games, bad prior-season record. The system's OWN PAST PICKS
(verified against blend_features_cache.csv) contradict two of those:
  - spreads ran +3 to +6.5, not +1 to +3.5
  - totals ran 33.5 to 48.5 -- two of five were OVER 44
So the filters are swept as parameters instead of assumed.

A "qualifying bet" = the underdog (per spread_line, home-positive
convention from backtest_srs.py) had a losing record the PRIOR season.
Graded ATS against the closing spread in the cache; pushes refund and
are excluded from the cover rate; ROI uses the actual spread odds in
the cache (away_spread_odds/home_spread_odds) on a flat 100 stake.

Prior-season records come from the cache itself, so eval starts 2021
(2020 teams' prior records need 2019 data the cache doesn't carry).

Same honesty note as backtest_segments.py: this is a post-hoc rule
sliced to match nine observed picks. A cell clearing breakeven at
n~100 over five seasons is a LEAD, not a confirmed edge.

USAGE
-----
  python backtest_baddogs.py
"""

import argparse
import sys

import pandas as pd

from backtest_srs import BREAKEVEN_COVER_PCT

CACHE = "blend_features_cache.csv"

SPREAD_BANDS = [(1.0, 3.5), (1.0, 6.5), (2.5, 6.5), (3.0, 6.5)]
PRIOR_THRESHOLDS = [(0.500, "< .500"), (0.354, "<= 6-11"), (0.295, "<= 5-12")]
TOTAL_CAPS = [(99.0, "no total filter"), (44.5, "total < 45"), (44.0, "total <= 44")]


def season_records(g, season):
    """team -> prior-season win pct (ties count half), from REG games."""
    rec = {}
    s = g[g["season"] == season]
    for t in pd.concat([s["home_team"], s["away_team"]]).unique():
        rec[t] = [0.0, 0.0]
    for _, r in s.iterrows():
        h, a = r["home_team"], r["away_team"]
        if r["home_score"] > r["away_score"]:
            rec[h][0] += 1; rec[a][1] += 1
        elif r["home_score"] < r["away_score"]:
            rec[a][0] += 1; rec[h][1] += 1
        else:
            rec[h][0] += 0.5; rec[a][0] += 0.5; rec[h][1] += 0.5; rec[a][1] += 0.5
    return {t: w / (w + l) for t, (w, l) in rec.items() if (w + l) > 0}


def profit_100(odds, won):
    """Flat 100 stake at American odds."""
    return (100 * 100 / -odds) if (won and odds < 0) else (odds if won else -100.0)


def build_bets(g, seasons):
    """One row per qualifying dog bet, graded, for seasons where the cache
    carries the prior season too."""
    rows = []
    for season in seasons:
        prior = season_records(g, season - 1)
        s = g[g["season"] == season].dropna(subset=["spread_line", "total_line"])
        for _, r in s.iterrows():
            if r["spread_line"] == 0:
                continue  # pick'em: no dog, no bet
            bet_home = r["spread_line"] < 0  # home getting points
            dog = r["home_team"] if bet_home else r["away_team"]
            pts = abs(r["spread_line"])
            if dog not in prior:
                continue
            margin_vs = r["actual_margin"] - r["spread_line"]  # >0: home covers
            push = margin_vs == 0
            covered = (margin_vs > 0) if bet_home else (margin_vs < 0)
            odds = r["home_spread_odds"] if bet_home else r["away_spread_odds"]
            rows.append({
                "season": season, "week": r["week"], "weekday": r["weekday"],
                "gameday": r["gameday"], "dog": dog, "bet_home": bet_home,
                "pts": pts, "total_line": r["total_line"],
                "prior_pct": prior[dog],
                "opp_prior_pct": prior.get(r["away_team"] if bet_home else r["home_team"]),
                "push": push, "covered": covered,
                "profit": 0.0 if push else profit_100(float(odds), covered),
            })
    return pd.DataFrame(rows)


def line(df, mask, label):
    sub = df[mask]
    decided = sub[~sub["push"]]
    if len(decided) == 0:
        return f"  {label}: no bets"
    pct = 100 * decided["covered"].mean()
    roi = 100 * sub["profit"].sum() / (100 * len(sub))
    pushes = len(sub) - len(decided)
    verdict = "ABOVE breakeven" if pct >= BREAKEVEN_COVER_PCT else "below breakeven"
    note = f", {pushes} push" if pushes else ""
    return (f"  {label}: {pct:5.1f}% cover | ROI {roi:+5.1f}% | "
            f"n={len(decided)}{note}  -- {verdict}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seasons", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    args = p.parse_args()

    g = pd.read_csv(CACHE)
    g = g[g["game_type"] == "REG"].copy()
    bets = build_bets(g, args.seasons)
    sun_mon = bets["weekday"].isin(["Sunday", "Monday"])

    print(f"\n=== 'Bad last season, small dog' system backtest, "
          f"{min(args.seasons)}-{max(args.seasons)} ===")
    print(f"Qualifying dog bets (any band): {len(bets)} | breakeven {BREAKEVEN_COVER_PCT}%\n")

    print("Baseline slices (no total filter, all weekdays):")
    print(line(bets, bets["prior_pct"] < 0.5, "all bad-prior dogs, any spread"))

    print("\nFilter sweep (Sunday/Monday only, per the system's schedule rule):")
    for cap, cap_lab in TOTAL_CAPS:
        for lo, hi in SPREAD_BANDS:
            for thr, thr_lab in PRIOR_THRESHOLDS:
                m = (sun_mon & (bets["pts"] >= lo) & (bets["pts"] <= hi)
                     & (bets["prior_pct"] < thr) & (bets["total_line"] < cap))
                print(line(bets, m, f"+{lo:g}-{hi:g} & prior {thr_lab:8s} & {cap_lab}"))

    # The configuration the actual picks imply: +3..+6.5, prior < .500,
    # Sunday/Monday, no total filter
    core = (sun_mon & (bets["pts"] >= 3.0) & (bets["pts"] <= 6.5)
            & (bets["prior_pct"] < 0.5))
    print("\nImplied config from the 9 observed picks (+3 to +6.5, prior < .500, Sun/Mon):")
    print(line(bets, core, "all weeks"))
    print("\n  by season:")
    for s in args.seasons:
        print(line(bets, core & (bets["season"] == s), f"{s}", ).replace("  2", "    2", 1))
    print("\n  by season segment:")
    print(line(bets, core & (bets["week"] <= 6), "weeks 1-6 (recency-bias window)"))
    print(line(bets, core & (bets["week"] > 6) & (bets["week"] <= 12), "weeks 7-12"))
    print(line(bets, core & (bets["week"] > 12), "weeks 13+"))
    print("\n  road vs home dogs:")
    print(line(bets, core & ~bets["bet_home"], "road dog"))
    print(line(bets, core & bets["bet_home"], "home dog"))

    print("\nCaution: parameters above were chosen to match nine observed picks -- "
          "that selection IS the multiple-comparisons problem. A cell clearing "
          f"{BREAKEVEN_COVER_PCT}% here is a lead to re-test forward, not a rule to bet.")


if __name__ == "__main__":
    main()
