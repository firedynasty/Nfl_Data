"""
Game Performance Score: one number per team summarizing how well the offense
moves the ball and protects its QB.

Components (season totals, then percentile-ranked across the 32 teams):
  + win percentage            (ties count as half a win; more = better)
  + turnover margin per game  (takeaways - giveaways; more = better)
  + first downs per game      (more = better)
  + yards per play            (higher = better)
  + rush yards per attempt    (higher = better)
  + red-zone TD rate          (higher = better)
  - pressure allowed rate     ((qb_hit | sack) / dropbacks; less = better, flipped)

Each component becomes a 0-100 percentile vs. the league ("better than X%
of teams"), and the weighted average is the Game Performance Score (0-100).
Percentiles are used instead of z-scores because they're bounded and
robust to one outlier team skewing the scale.

The score is built to be joined onto per-QB rows (e.g. qb_adot_catchpct.csv)
as context for QB ratings: a QB producing with a low game performance score is
doing more with less.

USAGE
-----
1. Team table for a season:
     python team_composite.py --seasons 2026

2. Same, but also attach the score to a QB CSV (joined on team):
     python team_composite.py --seasons 2026 --qb_csv ~/Downloads/qb_adot_catchpct.csv
"""

import argparse
import sys

import pandas as pd

from nfl_box_score_analysis import (
    build_long_results,
    first_downs_by_team_game,
    load_games,
    load_pbp_multi,
    plays_yards_by_team_game,
    redzone_by_team_game,
    team_rushing_by_team_game,
    turnover_margin_by_team_game,
)

# Weights for each component (must sum to 1.0). Edit to taste.
WEIGHTS = {
    "win_pct": 0.20,
    "to_margin_pg": 0.15,
    "first_downs_pg": 0.13,
    "yards_per_play": 0.13,
    "rush_yds_per_att": 0.13,
    "rz_td_pct": 0.13,
    "pressure_rate": 0.13,  # flipped below: less pressure = higher score
}


def pressure_allowed_by_team_game(pbp):
    """Dropbacks and pressures (qb_hit OR sack) per team-game. The `team` side
    is the team whose QB dropped back, so this is pressure *allowed*."""
    db = pbp[pbp["qb_dropback"] == 1].copy()
    db["pressured"] = ((db["qb_hit"] == 1) | (db["sack"] == 1)).astype(int)
    return db.groupby(["game_id", "posteam"]).agg(
        dropbacks=("pressured", "count"),
        pressured=("pressured", "sum"),
    ).reset_index()


def team_game_stats(seasons):
    """One row per team-game with every component stat merged on."""
    games = load_games()
    pbp = load_pbp_multi(seasons, tag="composite")
    long = build_long_results(games, seasons)

    df = long.copy()
    for stats in (
        first_downs_by_team_game(pbp),
        plays_yards_by_team_game(pbp).drop(columns=["yards_per_play"]),
        redzone_by_team_game(pbp)[["game_id", "posteam", "rz_trips", "rz_td"]],
        team_rushing_by_team_game(pbp).drop(columns=["yds_per_attempt"]),
        pressure_allowed_by_team_game(pbp),
    ):
        df = df.merge(stats, left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left")
        df = df.drop(columns=["posteam"])

    # Turnover margin is built on the long frame itself (team-keyed, no posteam)
    to = turnover_margin_by_team_game(pbp, long)
    df = df.merge(to[["game_id", "team", "turnover_margin"]], on=["game_id", "team"], how="left")

    df["first_downs"] = df["first_downs"].fillna(0)
    df[["rz_trips", "rz_td"]] = df[["rz_trips", "rz_td"]].fillna(0)
    return df


def team_season_table(team_game):
    """Aggregate team-games to season totals, then compute rate stats."""
    t = team_game.groupby("team").agg(
        games=("game_id", "count"),
        wins=("result", lambda s: (s == "W").sum()),
        ties=("result", lambda s: (s == "T").sum()),
        first_downs=("first_downs", "sum"),
        plays=("plays", "sum"),
        yards=("yards", "sum"),
        rush_attempts=("rush_attempts", "sum"),
        rush_yards=("rush_yards", "sum"),
        rz_trips=("rz_trips", "sum"),
        rz_td=("rz_td", "sum"),
        dropbacks=("dropbacks", "sum"),
        pressured=("pressured", "sum"),
        to_margin=("turnover_margin", "sum"),
    ).reset_index()

    t["win_pct"] = (t["wins"] + 0.5 * t["ties"]) / t["games"]
    t["to_margin_pg"] = t["to_margin"] / t["games"]
    t["first_downs_pg"] = t["first_downs"] / t["games"]
    t["yards_per_play"] = t["yards"] / t["plays"]
    t["rush_yds_per_att"] = t["rush_yards"] / t["rush_attempts"]
    # Teams with zero RZ trips get the league-average TD rate (neutral), not NaN
    t["rz_td_pct"] = (t["rz_td"] / t["rz_trips"]).fillna(t["rz_td"].sum() / t["rz_trips"].sum())
    t["pressure_rate"] = t["pressured"] / t["dropbacks"]
    return t


def add_game_performance_score(t):
    """Percentile-rank each component across teams and combine."""
    for col in WEIGHTS:
        t[f"{col}_pctl"] = (t[col].rank(pct=True) * 100).round(1)

    t["game_performance_score"] = sum(
        w * (100 - t[f"{col}_pctl"] if col == "pressure_rate" else t[f"{col}_pctl"])
        for col, w in WEIGHTS.items()
    ).round(1)
    return t.sort_values("game_performance_score", ascending=False).reset_index(drop=True)


def validate_against_results(team_game, seasons):
    """Sanity check: a game-level version of the score should track scoring
    margin and separate winners from losers (same idea as the CLI trends)."""
    g = team_game.copy()
    g["first_downs_pg"] = g["first_downs"]
    g["to_margin_pg"] = g["turnover_margin"]
    g["yards_per_play"] = g["yards"] / g["plays"]
    g["rush_yds_per_att"] = g["rush_yards"] / g["rush_attempts"]
    g["rz_td_pct"] = (g["rz_td"] / g["rz_trips"].where(g["rz_trips"] != 0)).fillna(
        g["rz_td"].sum() / g["rz_trips"].sum()
    )
    g["pressure_rate"] = g["pressured"] / g["dropbacks"]
    g["margin"] = g["team_score"] - g["opp_score"]

    score = 0
    # win_pct is skipped here: at the single-game level it *is* the result,
    # so including it would make this check circular. Validate the stat
    # components only, renormalized to sum to 1.
    stat_weights = {c: w for c, w in WEIGHTS.items() if c != "win_pct"}
    total_w = sum(stat_weights.values())
    for col, w in stat_weights.items():
        pct = g[col].rank(pct=True)
        score = score + (w / total_w) * (1 - pct if col == "pressure_rate" else pct)
    g["game_score"] = (score * 100).round(1)

    valid = g[g["result"].isin(["W", "L"])]
    corr = valid["game_score"].corr(valid["margin"])
    by_result = valid.groupby("result")["game_score"].mean().round(1)
    print("\n--- Validation (game-level score vs. results, stat components only) ---")
    print(f"Correlation with scoring margin: {corr:.2f}")
    print(f"Avg score in wins: {by_result.get('W')}  |  in losses: {by_result.get('L')}")
    print("(win% excluded here -- at game level it IS the result, so including")
    print("it would be circular; expect positive correlation and wins above losses)")


def main():
    parser = argparse.ArgumentParser(description="Game Performance Score composite")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2026])
    parser.add_argument("--qb_csv", default=None,
                        help="Optional QB CSV with a 'team' column; the score is joined onto it")
    parser.add_argument("--out", default="game_performance_score.csv")
    args = parser.parse_args()

    if abs(sum(WEIGHTS.values()) - 1.0) > 1e-9:
        sys.exit(f"WEIGHTS must sum to 1.0, got {sum(WEIGHTS.values())}")

    team_game = team_game_stats(args.seasons)
    t = add_game_performance_score(team_season_table(team_game))

    cols = ["team", "games", "game_performance_score", "win_pct", "to_margin_pg",
            "first_downs_pg", "yards_per_play", "rush_yds_per_att",
            "rz_td_pct", "pressure_rate"]
    show = t[cols].copy()
    for c in ["first_downs_pg", "yards_per_play", "rush_yds_per_att", "to_margin_pg"]:
        show[c] = show[c].round(2)
    show["win_pct"] = (show["win_pct"] * 100).round(1)
    show["rz_td_pct"] = (show["rz_td_pct"] * 100).round(1)
    show["pressure_rate"] = (show["pressure_rate"] * 100).round(1)
    print(f"\n=== Game Performance Score — season(s): {args.seasons} ===")
    print("(rate columns shown as raw values; RZ TD% and pressure rate as percentages)\n")
    print(show.to_string(index=False))

    t.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")

    validate_against_results(team_game, args.seasons)

    if args.qb_csv:
        qb = pd.read_csv(args.qb_csv)
        before = len(qb)
        qb = qb.merge(
            t[["team", "game_performance_score"] + [f"{c}_pctl" for c in WEIGHTS]],
            on="team", how="left",
        )
        unmatched = qb[qb["game_performance_score"].isna()]["team"].unique()
        out_path = args.qb_csv.replace(".csv", "_with_game_performance.csv")
        qb.to_csv(out_path, index=False)
        print(f"\nAttached score to {before} QB rows -> {out_path}")
        if len(unmatched):
            print(f"WARNING: no team match for: {list(unmatched)}")
        print(qb[["passer_player_name", "team", "game_performance_score"]]
              .sort_values("game_performance_score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
