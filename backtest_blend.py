"""
Phase 7: blend SRS with trailing QB EPA/dropback and pressure-allowed rate,
with the blend weights FITTED by regression — never hand-picked (the lesson
of the earlier blend that underperformed QB-EPA alone).

Honesty protocol, same discipline as the SRS backtest:
  - Every feature is as-of: trailing team-game sums with a cold-start blend
    against the prior season (identical k/shrink formula to
    blended_asof_ratings), so game N only sees games < N.
  - Weights are fit by EXPANDING-WINDOW OLS: to predict season S, train on
    seasons < S only. The 2024 model never sees a 2024 game.
  - Baseline is SRS-only refit through the SAME expanding window on the
    SAME games — the comparison is apples-to-apples.
  - EPA here is TEAM-level (offense's EPA/dropback on all dropbacks).
    Per-QB trailing (using games.csv's home_qb_name/away_qb_name to catch
    starter changes/injuries) is the Phase 7b upgrade if this shows life.

Model: actual_home_margin ~ b0*home_flag + b1*srs_diff + b2*epa_diff
       + b3*prate_diff   (no intercept; home_flag is 0 on neutral sites,
       so b0 IS the home-field term)

USAGE
-----
  python backtest_blend.py                            # eval 2022-2025, expanding window
  python backtest_blend.py --eval_seasons 2024 2025
  python backtest_blend.py --k 4 --shrink 0.7 --rule 4
"""

import argparse
import sys

import numpy as np
import pandas as pd

from nfl_box_score_analysis import load_games, load_pbp, build_long_results
from team_composite import qb_epa_by_team_game, pressure_allowed_by_team_game
from srs import blended_asof_ratings
from backtest_srs import normal_cdf, report

FEATURES = ["home_flag", "srs_diff", "epa_diff", "prate_diff"]


def trailing_features(games, pbp_seasons, k=4.0, shrink=0.7):
    """Per team-game, as-of EPA/dropback and pressure-allowed rate: trailing
    sums over that team's earlier games in the season, cold-start blended
    with the prior season (same g/(g+k) ramp as the SRS blend). Sums are
    trailed and rates computed only at lookup time, so dropback volume is
    weighted correctly."""
    long = build_long_results(games, pbp_seasons)
    frames = []
    for s in pbp_seasons:
        try:
            pbp = load_pbp(s)
        except Exception as e:  # season not available (e.g. pre-2021 pbp)
            print(f"  note: no play-by-play for {s} ({type(e).__name__}) -- "
                  f"its rows will rely on prior-season blend only", file=sys.stderr)
            continue
        frames.append(qb_epa_by_team_game(pbp).merge(
            pressure_allowed_by_team_game(pbp), on=["game_id", "posteam"]))
        print(f"  built EPA/pressure features for {s}", file=sys.stderr)
        del pbp
    feat = pd.concat(frames, ignore_index=True)

    tg = long.merge(feat, left_on=["game_id", "team"], right_on=["game_id", "posteam"],
                    how="left").drop(columns=["posteam"])
    tg[["total_epa", "dropbacks", "pressured"]] = tg[["total_epa", "dropbacks", "pressured"]].fillna(0)
    tg = tg.sort_values(["season", "team", "week"]).reset_index(drop=True)

    grp = tg.groupby(["season", "team"])
    tg["n_prior"] = grp.cumcount()
    for col in ["total_epa", "dropbacks", "pressured"]:
        tg[f"{col}_trail"] = grp[col].transform(lambda s: s.shift(1).cumsum()).fillna(0)

    pri = tg.groupby(["season", "team"]).agg(
        epa=("total_epa", "sum"), db=("dropbacks", "sum"), pr=("pressured", "sum")).reset_index()
    pri["prior_epa_db"] = pri["epa"] / pri["db"]
    pri["prior_prate"] = pri["pr"] / pri["db"]
    pri["season"] += 1  # this season's totals are NEXT season's prior
    tg = tg.merge(pri[["season", "team", "prior_epa_db", "prior_prate"]],
                  on=["season", "team"], how="left")

    w = tg["n_prior"] / (tg["n_prior"] + k)
    db_trail = tg["dropbacks_trail"].where(tg["dropbacks_trail"] > 0)
    tg["epa_db_asof"] = (w * (tg["total_epa_trail"] / db_trail)
                         + (1 - w) * shrink * tg["prior_epa_db"])
    tg["prate_asof"] = (w * (tg["pressured_trail"] / db_trail)
                        + (1 - w) * shrink * tg["prior_prate"])
    # week 1 (w=0, no trailing dropbacks) -> the regressed prior; missing
    # prior (no pbp that year) -> league-average-ish fallbacks
    tg["epa_db_asof"] = tg["epa_db_asof"].fillna(shrink * tg["prior_epa_db"]).fillna(0.0)
    tg["prate_asof"] = tg["prate_asof"].fillna(shrink * tg["prior_prate"]) \
        .fillna(tg["pressured"].sum() / max(1, tg["dropbacks"].sum()))
    return tg[["game_id", "season", "week", "team", "epa_db_asof", "prate_asof"]]


def assemble_games(games, seasons, srs_ratings, tg):
    """One row per played game with all four model inputs as-of."""
    g = games.dropna(subset=["home_score", "away_score"]).copy()
    g["season"] = g["season"].astype(int)
    g = g[g["season"].isin(list(seasons))]
    g["neutral"] = g.get("location", "Home").eq("Neutral")
    g["actual_margin"] = g["home_score"] - g["away_score"]
    g["home_won"] = (g["home_score"] > g["away_score"]).astype(int)
    g["home_flag"] = (~g["neutral"]).astype(float)
    g["srs_diff"] = [
        srs_ratings[(s, w)][h] - srs_ratings[(s, w)][a]
        for s, w, h, a in zip(g["season"], g["week"], g["home_team"], g["away_team"])
    ]
    for side, tcol in [("home", "home_team"), ("away", "away_team")]:
        f = tg.rename(columns={"team": tcol, "epa_db_asof": f"{side}_epa",
                               "prate_asof": f"{side}_prate"})
        g = g.merge(f[["game_id", tcol, f"{side}_epa", f"{side}_prate"]],
                    on=["game_id", tcol], how="left")
    g["epa_diff"] = g["home_epa"] - g["away_epa"]
    g["prate_diff"] = g["home_prate"] - g["away_prate"]
    return g.dropna(subset=["srs_diff", "epa_diff", "prate_diff"])


def fit_predict(g, eval_seasons, feature_cols):
    """Expanding window: season S is predicted by a model trained only on
    seasons < S. Returns (graded rows, fitted coefs per season)."""
    out, coefs = [], {}
    for S in eval_seasons:
        train = g[g["season"] < S]
        test = g[g["season"] == S].copy()
        X = train[feature_cols].to_numpy(float)
        y = train["actual_margin"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        std = (y - X @ beta).std()  # TRAIN residuals: the honest error scale
        test["pred_margin"] = test[feature_cols].to_numpy(float) @ beta
        test["win_std"] = std
        coefs[S] = {**{c: round(float(b), 3) for c, b in zip(feature_cols, beta)},
                    "std": round(float(std), 2), "train_n": len(train)}
        out.append(test)
    return pd.concat(out), coefs


def grade_model(df):
    """Same grading contract as backtest_srs.grade, but pred_margin comes
    from the fitted model and win-prob std from each season's train fit."""
    df = df.copy()
    df["home_win_prob"] = [normal_cdf(m / s) for m, s in zip(df["pred_margin"], df["win_std"])]
    df["brier"] = (df["home_win_prob"] - df["home_won"]) ** 2
    df["correct"] = (df["pred_margin"] > 0) == (df["home_won"] == 1)
    df["edge"] = df["pred_margin"] - df["spread_line"]
    df["abs_edge"] = df["edge"].abs()

    ats = df.dropna(subset=["spread_line"]).copy()
    ats = ats[ats["edge"] != 0]
    ats["bet_home"] = ats["edge"] > 0
    mvs = ats["actual_margin"] - ats["spread_line"]
    ats["ats_result"] = "push"
    ats.loc[mvs > 0, "ats_result"] = "home_covers"
    ats.loc[mvs < 0, "ats_result"] = "away_covers"
    ats["bet_won"] = (((ats["bet_home"]) & (ats["ats_result"] == "home_covers"))
                      | ((~ats["bet_home"]) & (ats["ats_result"] == "away_covers")))
    return df, ats


def main():
    p = argparse.ArgumentParser(description="Phase 7: fitted SRS + QB-EPA + pressure blend backtest")
    p.add_argument("--eval_seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025],
                   help="seasons to grade; each is predicted by a model fit on earlier seasons")
    p.add_argument("--k", type=float, default=4.0, help="cold-start blend, pseudo-games")
    p.add_argument("--shrink", type=float, default=0.7, help="prior-season regression")
    p.add_argument("--cap", type=float, default=None, help="SRS margin cap")
    p.add_argument("--rule", type=float, default=4.0, help="betting-rule |edge| threshold")
    args = p.parse_args()

    games = load_games()
    lo, hi = min(args.eval_seasons), max(args.eval_seasons)
    # features needed from lo-2 (prior for lo-1's rows) through hi
    pbp_seasons = list(range(lo - 2, hi + 1))
    tg = trailing_features(games, pbp_seasons, k=args.k, shrink=args.shrink)
    feat_seasons = sorted(tg["season"].unique())

    srs_ratings = blended_asof_ratings(games, feat_seasons, cap=args.cap,
                                       k=args.k, shrink=args.shrink)
    g = assemble_games(games, feat_seasons, srs_ratings, tg)

    print("\nFitted weights per eval season (expanding window):")
    for label, cols in [("M1 blend", FEATURES), ("M0 SRS-only", ["home_flag", "srs_diff"])]:
        _, coefs = fit_predict(g, args.eval_seasons, cols)
        line = " | ".join(f"{S}: " + ", ".join(f"{c}={v}" for c, v in co.items()
                                               if c not in ("std", "train_n"))
                          for S, co in coefs.items())
        print(f"  {label}: {line}")

    for label, cols in [("M0 SRS-only (fitted, same window)", ["home_flag", "srs_diff"]),
                        ("M1 blend: SRS + EPA/db + pressure (fitted)", FEATURES)]:
        pred, _ = fit_predict(g, args.eval_seasons, cols)
        df, ats = grade_model(pred)
        report(df, ats, args.eval_seasons,
               "HFA: fitted per season (home_flag coef above)",
               "win-prob std: per-season train residuals",
               args.rule, f"Phase 7 {label}",
               footer="\nNo leakage: features are as-of (trailing, cold-start blended); "
               "weights are fit on seasons < the graded one (expanding window). "
               "Blend constants k/shrink still untuned placeholders.")


if __name__ == "__main__":
    main()
