"""
Phase 8: totals (over/under) market -- untested until now (see
intermediate/plan.md, "Segment tests" #3). SRS predicts MARGIN; this is a
genuinely different, separate model for combined SCORING PACE, fit and
graded the same honest way as everything else in this project.

Model: actual_total ~ b0 + b1*home_pf + b2*home_pa + b3*away_pf + b4*away_pa
  pf = that team's own trailing points-FOR, pa = trailing points-AGAINST,
  both as-of (only games strictly before the one being predicted) and
  cold-start blended with the team's PRIOR season -- same g/(g+k) ramp as
  srs.py's blended_asof_ratings() and backtest_blend.py's trailing
  EPA/pressure features, generalized to a stat that isn't already
  zero-centered: the prior season's rate is shrunk toward that PRIOR
  season's LEAGUE-AVERAGE points (not toward 0), which collapses to the
  exact same `shrink * prior` formula those other features use once you
  notice their league average is ~0 -- points obviously aren't.

Weights are fit by EXPANDING-WINDOW OLS (season S predicted by a model
trained only on seasons < S) -- never hand-picked, same discipline as
Phase 7. No play-by-play needed: pf/pa come straight from games.csv
scores, so training can reach back to 1999 (total_line has full coverage
back that far, per road_dog_system.py's own docstring).

Two things graded, since they answer different questions:
  1. ATS-style cover rate for a betting rule (edge = pred_total -
     total_line; bet the side edge points to), bucketed by |edge| --
     same shape as backtest_srs.py, so it's directly comparable.
  2. A raw cutoff table: bucket games by pred_total itself (not edge) and
     report the actual under-rate per bucket. This is the literal
     question asked -- "how bad do both teams have to be before under is
     more likely, and where's the cutoff" -- independent of whether the
     market has already priced it (the ATS test answers THAT).

USAGE
-----
  python backtest_totals.py                          # eval 2021-2025, expanding window
  python backtest_totals.py --eval_seasons 2024 2025
  python backtest_totals.py --k 4 --shrink 0.7 --rule 4
"""

import argparse

import numpy as np
import pandas as pd

from nfl_box_score_analysis import load_games, build_long_results
from backtest_srs import BREAKEVEN_COVER_PCT

FEATURES = ["intercept", "home_pf", "home_pa", "away_pf", "away_pa"]
EDGE_BUCKETS = [(0, 1), (1, 2), (2, 3), (3, 99)]  # |edge|, points
PRED_TOTAL_BUCKETS = [(0, 40), (40, 44), (44, 48), (48, 52), (52, 999)]


def profit_100(odds, won):
    """Flat 100 stake at American odds (same helper as backtest_baddogs.py)."""
    return (100 * 100 / -odds) if (won and odds < 0) else (odds if won else -100.0)


def trailing_pace(long, k=4.0, shrink=0.7):
    """As-of, cold-start-blended trailing points-for/points-against per
    team-game. See module docstring for the shrink-toward-league-average
    generalization."""
    long = long.sort_values(["season", "team", "week"]).reset_index(drop=True)
    grp = long.groupby(["season", "team"])
    long["n_prior"] = grp.cumcount()
    long["pf_trail"] = grp["team_score"].transform(lambda s: s.shift(1).cumsum())
    long["pa_trail"] = grp["opp_score"].transform(lambda s: s.shift(1).cumsum())

    lg = long.groupby("season")["team_score"].mean().rename("lg_pf")
    team_szn = long.groupby(["season", "team"]).agg(
        pf=("team_score", "mean"), pa=("opp_score", "mean")).reset_index()
    team_szn = team_szn.merge(lg, on="season")
    team_szn["prior_pf"] = team_szn["lg_pf"] + shrink * (team_szn["pf"] - team_szn["lg_pf"])
    team_szn["prior_pa"] = team_szn["lg_pf"] + shrink * (team_szn["pa"] - team_szn["lg_pf"])
    team_szn["season"] += 1  # this season's shrunk rate -> next season's prior
    long = long.merge(team_szn[["season", "team", "prior_pf", "prior_pa"]],
                      on=["season", "team"], how="left")

    overall_lg_pf = long["team_score"].mean()  # teams with no usable prior season
    long["prior_pf"] = long["prior_pf"].fillna(overall_lg_pf)
    long["prior_pa"] = long["prior_pa"].fillna(overall_lg_pf)

    w = long["n_prior"] / (long["n_prior"] + k)
    cur_pf = long["pf_trail"] / long["n_prior"].where(long["n_prior"] > 0)
    cur_pa = long["pa_trail"] / long["n_prior"].where(long["n_prior"] > 0)
    long["pf_asof"] = w * cur_pf.fillna(0) + (1 - w) * long["prior_pf"]
    long["pa_asof"] = w * cur_pa.fillna(0) + (1 - w) * long["prior_pa"]
    return long[["game_id", "season", "week", "team", "pf_asof", "pa_asof"]]


def assemble_games(games, seasons, pace):
    """One row per played game with the four as-of features, the actual
    total, the closing total_line, and the odds needed for ROI."""
    g = games.dropna(subset=["home_score", "away_score", "total_line"]).copy()
    g["season"] = g["season"].astype(int)
    g = g[g["season"].isin(list(seasons))]
    g["actual_total"] = g["home_score"] + g["away_score"]
    g["intercept"] = 1.0
    for side, tcol in [("home", "home_team"), ("away", "away_team")]:
        f = pace.rename(columns={"team": tcol, "pf_asof": f"{side}_pf", "pa_asof": f"{side}_pa"})
        g = g.merge(f[["game_id", tcol, f"{side}_pf", f"{side}_pa"]], on=["game_id", tcol], how="left")
    return g.dropna(subset=["home_pf", "home_pa", "away_pf", "away_pa"])


def fit_predict(g, eval_seasons, feature_cols):
    """Expanding window: season S predicted by a model trained on seasons < S."""
    out, coefs = [], {}
    for S in eval_seasons:
        train = g[g["season"] < S]
        test = g[g["season"] == S].copy()
        X = train[feature_cols].to_numpy(float)
        y = train["actual_total"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        test["pred_total"] = test[feature_cols].to_numpy(float) @ beta
        coefs[S] = {**{c: round(float(b), 3) for c, b in zip(feature_cols, beta)},
                    "train_n": len(train)}
        out.append(test)
    return pd.concat(out), coefs


def grade(df):
    """edge = pred_total - total_line (positive -> bet OVER). Grades ATS-
    style against the closing total, with ROI from the actual over/under
    odds in games.csv (falls back to flat -110 when odds are missing)."""
    df = df.copy()
    df["edge"] = df["pred_total"] - df["total_line"]
    df["abs_edge"] = df["edge"].abs()
    bets = df[df["edge"] != 0].copy()
    bets["bet_over"] = bets["edge"] > 0
    diff = bets["actual_total"] - bets["total_line"]
    bets["result"] = "push"
    bets.loc[diff > 0, "result"] = "over"
    bets.loc[diff < 0, "result"] = "under"
    bets["bet_won"] = ((bets["bet_over"] & (bets["result"] == "over"))
                       | ((~bets["bet_over"]) & (bets["result"] == "under")))
    odds_col = np.where(bets["bet_over"], bets.get("over_odds", -110.0), bets.get("under_odds", -110.0))
    odds_col = pd.to_numeric(pd.Series(odds_col, index=bets.index), errors="coerce").fillna(-110.0)
    bets["profit"] = [0.0 if r == "push" else profit_100(o, won)
                      for r, o, won in zip(bets["result"], odds_col, bets["bet_won"])]
    return bets


def cover_line(mask, bets, label):
    sub = bets[mask]
    decided = sub[sub["result"] != "push"]
    if len(decided) == 0:
        return f"  {label}: no bets"
    pct = 100 * decided["bet_won"].mean()
    roi = 100 * sub["profit"].sum() / (100 * len(sub))
    pushes = len(sub) - len(decided)
    verdict = "ABOVE breakeven" if pct >= BREAKEVEN_COVER_PCT else "below breakeven"
    push_note = f", {pushes} push" if pushes else ""
    return f"  {label}: {pct:5.1f}% cover | ROI {roi:+5.1f}% | n={len(decided)}{push_note}  -- {verdict}"


def raw_cutoff_table(bets):
    """Not a betting rule -- just: at a given predicted-total LEVEL (not
    edge vs the market), how often does the game actually go under? This
    is the literal "how bad do both teams have to be" question, separate
    from whether the market has already priced it in."""
    print(f"\nRaw under-rate by predicted-total level (n, not vs. market -- "
          f"just: does a low predicted total actually mean fewer points?):")
    for lo, hi in PRED_TOTAL_BUCKETS:
        m = (bets["pred_total"] >= lo) & (bets["pred_total"] < hi)
        sub = bets[m]
        decided = sub[sub["result"] != "push"]
        label = f"pred {lo}-{hi if hi < 999 else '+'}"
        if len(decided) == 0:
            print(f"  {label:<12}: no games")
            continue
        under_pct = 100 * (decided["result"] == "under").mean()
        print(f"  {label:<12}: {under_pct:5.1f}% actually went under  "
              f"(n={len(decided)}, avg actual total {decided['actual_total'].mean():.1f})")


def report(bets, coefs, eval_seasons, rule_edge):
    print(f"\n=== Totals (O/U) walk-forward backtest -- {min(eval_seasons)}-{max(eval_seasons)} ===")
    print(f"games graded: {len(bets)} | breakeven {BREAKEVEN_COVER_PCT}%")
    print("Fitted weights per eval season (expanding window):")
    for S, co in coefs.items():
        line_ = ", ".join(f"{c}={v}" for c, v in co.items() if c != "train_n")
        print(f"  {S}: {line_}  (train_n={co['train_n']})")

    print(f"\nATS-style cover rate by |edge| bucket (bet = side the edge points to):")
    for lo, hi in EDGE_BUCKETS:
        label = f"{lo}-{hi if hi < 99 else '+'} pt" if hi < 99 else f"{lo}+  pt"
        print(cover_line((bets["abs_edge"] >= lo) & (bets["abs_edge"] < hi), bets, label))
    print(f"\nSame buckets, OVER bets only:")
    for lo, hi in EDGE_BUCKETS:
        label = f"{lo}-{hi if hi < 99 else '+'} pt" if hi < 99 else f"{lo}+  pt"
        m = (bets["abs_edge"] >= lo) & (bets["abs_edge"] < hi) & bets["bet_over"]
        print(cover_line(m, bets, label))
    print(f"\nSame buckets, UNDER bets only:")
    for lo, hi in EDGE_BUCKETS:
        label = f"{lo}-{hi if hi < 99 else '+'} pt" if hi < 99 else f"{lo}+  pt"
        m = (bets["abs_edge"] >= lo) & (bets["abs_edge"] < hi) & (~bets["bet_over"])
        print(cover_line(m, bets, label))

    print(f"\nBetting rule (|edge| >= {rule_edge:g}):")
    print(cover_line(bets["abs_edge"] >= rule_edge, bets, f"|edge| >= {rule_edge:g}"))

    raw_cutoff_table(bets)

    print("\nNo leakage: pf/pa features are as-of (trailing, cold-start blended "
          "toward each team's shrunk prior-season rate); weights fit on seasons "
          "< the graded one (expanding window). Blend constants k/shrink are "
          "untuned placeholders, same as every other model in this project.")


def main():
    p = argparse.ArgumentParser(description="Phase 8: totals (O/U) walk-forward backtest")
    p.add_argument("--eval_seasons", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    p.add_argument("--k", type=float, default=4.0, help="cold-start blend, pseudo-games")
    p.add_argument("--shrink", type=float, default=0.7, help="prior-season regression")
    p.add_argument("--rule", type=float, default=2.0, help="betting-rule |edge| threshold")
    p.add_argument("--save_csv", action="store_true", help="save per-game grading to CSV")
    args = p.parse_args()

    games = load_games()
    lo = min(args.eval_seasons)
    train_from = 2000  # total_line has full coverage back to 1999; 2000 leaves a prior season
    long = build_long_results(games, list(range(train_from - 1, max(args.eval_seasons) + 1)))
    pace = trailing_pace(long, k=args.k, shrink=args.shrink)

    all_seasons = list(range(train_from, max(args.eval_seasons) + 1))
    g = assemble_games(games, all_seasons, pace)
    if g.empty:
        raise SystemExit("no graded games -- check --eval_seasons")

    pred, coefs = fit_predict(g, args.eval_seasons, FEATURES)
    bets = grade(pred)
    report(bets, coefs, args.eval_seasons, args.rule)

    if args.save_csv:
        bets.to_csv("backtest_totals_games.csv", index=False)
        print("\nSaved: backtest_totals_games.csv")


if __name__ == "__main__":
    main()
