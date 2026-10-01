"""
Build a per-QB CSV (dropbacks, EPA/dropback, completions, catch%, team
record) from nflverse play-by-play, in the shape `team_composite.py
--qb_csv` and `qb_support_chart.py` expect.

No script previously built this -- the existing output/qb_adot_catchpct.csv
was put together ad hoc in an earlier session. This replaces that with a
reproducible pipeline, and swaps ADOT (a style stat: how far downfield a QB
throws) for EPA/dropback (a value stat: how much those throws actually
helped the team score) as the primary QB-quality number.

USAGE
-----
    python build_qb_csv.py --season 2026 --min_dropbacks 20 --out ~/Downloads/qb_epa_catchpct.csv

Weekly refresh (full pipeline):
    python build_qb_csv.py --season 2026 --out ~/Downloads/qb_epa_catchpct.csv
    python team_composite.py --seasons 2026 --qb_csv ~/Downloads/qb_epa_catchpct.csv
    /Users/stanleytan/anaconda3/envs/test/bin/python qb_support_chart.py \\
        --csv ~/Downloads/qb_epa_catchpct_with_game_performance.csv \\
        --out ~/Downloads/qb_epa_catchpct_game_performance.png
"""

import argparse

import pandas as pd

from nfl_box_score_analysis import build_long_results, load_games, load_pbp


def build_qb_table(season, min_dropbacks=20):
    pbp = load_pbp(season)
    games = load_games()

    db = pbp[pbp["qb_dropback"] == 1].copy()
    qb = db.groupby(["passer_player_name", "posteam"]).agg(
        dropbacks=("epa", "count"),
        total_epa=("epa", "sum"),
    ).reset_index()
    qb["epa_per_dropback"] = (qb["total_epa"] / qb["dropbacks"]).round(3)

    # Catch% is defined over pass attempts only (excludes sacks/scrambles),
    # same convention the old ADOT-based CSV used.
    passes = db[db["pass_attempt"] == 1]
    comp = passes.groupby(["passer_player_name", "posteam"]).agg(
        completions=("complete_pass", "sum"),
        attempts=("pass_attempt", "sum"),
    ).reset_index()
    comp["catch_pct"] = comp["completions"] / comp["attempts"]

    qb = qb.merge(
        comp[["passer_player_name", "posteam", "completions", "catch_pct"]],
        on=["passer_player_name", "posteam"], how="left",
    )
    qb = qb[qb["dropbacks"] >= min_dropbacks].rename(columns={"posteam": "team"})
    qb = qb.drop(columns=["total_epa"])

    # Team record so far this season, same shape the old CSV carried.
    long = build_long_results(games, [season]).sort_values(["team", "week"])
    long["margin"] = long["team_score"] - long["opp_score"]
    long = long[long["result"].isin(["W", "L", "T"])]

    rec = long.groupby("team").agg(
        wins=("result", lambda s: (s == "W").sum()),
        losses=("result", lambda s: (s == "L").sum()),
        avg_margin=("margin", "mean"),
    ).reset_index()
    rec["avg_margin"] = rec["avg_margin"].round(1)

    last = long.groupby("team").tail(1)[["team", "result", "margin"]].rename(
        columns={"result": "last_result", "margin": "last_margin"}
    )
    rec = rec.merge(last, on="team")
    rec["record_str"] = rec["wins"].astype(str) + "-" + rec["losses"].astype(str)

    qb = qb.merge(rec, on="team", how="left")
    return qb.sort_values("epa_per_dropback", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--min_dropbacks", type=int, default=20,
                     help="Drop QBs below this many dropbacks (default 20, low because early season)")
    ap.add_argument("--out", default="output/qb_epa_catchpct.csv")
    args = ap.parse_args()

    qb = build_qb_table(args.season, args.min_dropbacks)
    qb.to_csv(args.out, index=False)
    print(f"Saved {len(qb)} QBs (min {args.min_dropbacks} dropbacks) -> {args.out}")
    print(qb[["passer_player_name", "team", "dropbacks", "epa_per_dropback", "catch_pct", "record_str"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
