"""
grade_predictions.py -- Phase 6: grade the LIVE predictions log.

Reads predictions_log.csv (written by `predict_week.py --log` before
kickoff, append-only), joins to actual results, and reports the same
metrics as the backtests -- win accuracy, Brier score, ATS cover rates --
but for predictions that were made blind, per model_version. That's the
honest A/B test for any future tweak: bump MODEL_VERSION in
predict_week.py, log some weeks, re-run this, and the versions are
compared on the weeks they share.

USAGE
-----
  python grade_predictions.py                 # all versions
  python grade_predictions.py --version srs-v1
"""

import argparse
import sys

import pandas as pd

from nfl_box_score_analysis import load_games
from backtest_srs import BREAKEVEN_COVER_PCT
from predict_week import LOG_PATH


def grade(df):
    df = df.copy()
    df["actual_margin"] = df["home_score"] - df["away_score"]
    df["home_won"] = (df["home_score"] > df["away_score"]).astype(int)
    df["correct"] = (df["pred_margin"] > 0) == (df["home_won"] == 1)
    df["brier"] = (df["home_win_prob"] - df["home_won"]) ** 2

    ats = df.dropna(subset=["spread_line", "edge"]).copy()
    ats = ats[ats["edge"] != 0]
    ats["bet_home"] = ats["edge"] > 0
    mvs = ats["actual_margin"] - ats["spread_line"]
    ats["ats_result"] = "push"
    ats.loc[mvs > 0, "ats_result"] = "home_covers"
    ats.loc[mvs < 0, "ats_result"] = "away_covers"
    ats["bet_won"] = (((ats["bet_home"]) & (ats["ats_result"] == "home_covers"))
                      | ((~ats["bet_home"]) & (ats["ats_result"] == "away_covers")))
    return df, ats


def version_row(df, ats, version):
    decided = ats[ats["ats_result"] != "push"]
    cover = f"{100 * decided['bet_won'].mean():.1f}% (n={len(decided)})" if len(decided) else "-"
    big = decided[decided["edge"].abs() >= 4]
    big_cover = f"{100 * big['bet_won'].mean():.1f}% (n={len(big)})" if len(big) else "-"
    weeks = f"w{min(df['week'])}-w{max(df['week'])}"
    return {
        "version": version,
        "games": len(df),
        "weeks": weeks,
        "win_acc": f"{100 * df['correct'].mean():.1f}%",
        "brier": f"{df['brier'].mean():.3f}",
        "ats_cover": cover,
        "ats_|edge|>=4": big_cover,
    }


def main():
    p = argparse.ArgumentParser(description="Grade the live predictions log (Phase 6)")
    p.add_argument("--version", default=None, help="grade only this model_version")
    args = p.parse_args()

    try:
        log = pd.read_csv(LOG_PATH)
    except FileNotFoundError:
        sys.exit(f"no log yet -- run `python predict_week.py --log` before a week's games")

    if args.version:
        log = log[log["model_version"] == args.version]
        if log.empty:
            sys.exit(f"no rows for model_version={args.version}")

    games = load_games()
    scores = games[["game_id", "home_score", "away_score"]].dropna(subset=["home_score"])
    df = log.merge(scores, on="game_id", how="left")

    pending = df[df["home_score"].isna()]
    graded = df.dropna(subset=["home_score"]).copy()

    print(f"\n=== Live prediction log grading ===")
    print(f"Log: {LOG_PATH} | rows: {len(log)} | versions: "
          f"{', '.join(sorted(log['model_version'].unique()))}")
    if len(pending):
        wk = ", ".join(f"{s} wk{w}" for s, w in
                       sorted(pending[['season', 'week']].drop_duplicates().itertuples(index=False)))
        print(f"Pending (logged, not yet played): {len(pending)} games ({wk})")

    if graded.empty:
        print("\nNothing graded yet -- grades land once logged games are played.")
        return

    df, ats = grade(graded)
    rows = []
    for v, g in df.groupby("model_version"):
        a = ats[ats["model_version"] == v]
        rows.append(version_row(g, a, v))

    print(f"\nGraded games: {len(df)} (breakeven for ATS: {BREAKEVEN_COVER_PCT}%)\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nHome-always baseline on the same games: {100 * df['home_won'].mean():.1f}%")
    print("Compare versions only on weeks they SHARE (different week sets "
          "aren't a fair fight).")


if __name__ == "__main__":
    main()
