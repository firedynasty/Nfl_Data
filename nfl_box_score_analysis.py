"""
NFL box score + stat comparison toolkit.

Pulls official schedule/results and play-by-play data straight from the
nflverse-data GitHub releases (no scraping, no API key, no rate limits).

Requires: pandas
    pip install pandas

USAGE
-----
1. Single game box score (who won + team stat comparison):
     python nfl_box_score_analysis.py boxscore --game_id 2025_02_PHI_KC
   (game_id format: SEASON_WEEK_AWAYTEAM_HOMETEAM)

2. List games for a team/season to find a game_id:
     python nfl_box_score_analysis.py list-games --season 2025 --team KC

3. Trend commands — compare winners vs losers across seasons for a given stat:
     python nfl_box_score_analysis.py first-downs-trend --seasons 2021 2022 2023 2024 2025
     python nfl_box_score_analysis.py redzone-trend     --seasons 2021 2022 2023 2024 2025
     python nfl_box_score_analysis.py rushing-trend     --seasons 2021 2022 2023 2024 2025 --min_attempts 10 --ypc_threshold 4.0
     python nfl_box_score_analysis.py turnover-trend    --seasons 2021 2022 2023 2024 2025

4. Run all four trends at once and print a combined summary:
     python nfl_box_score_analysis.py all-trends --seasons 2021 2022 2023 2024 2025
"""

import argparse
import sys
import pandas as pd

GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
PBP_URL_TMPL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"


# ---------------------------------------------------------------------------
# Shared loaders / helpers
# ---------------------------------------------------------------------------

def load_games():
    return pd.read_csv(GAMES_URL)


def load_pbp(season):
    return pd.read_csv(PBP_URL_TMPL.format(season=season), compression="gzip", low_memory=False)


def load_pbp_multi(seasons, tag=""):
    frames = []
    for s in seasons:
        pbp = load_pbp(s)
        frames.append(pbp)
        label = f" ({tag})" if tag else ""
        print(f"  loaded {s} play-by-play{label} ({len(pbp)} plays)", file=sys.stderr)
    return pd.concat(frames, ignore_index=True)


def build_long_results(games, seasons):
    """Reshape games.csv (one row per game) into one row per team per game,
    with a W/L/T result column."""
    g = games.dropna(subset=["home_score", "away_score"]).copy()
    g["season"] = g["season"].astype(int)
    g = g[g["season"].isin(seasons)]

    cols = ["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"]
    home = g[cols].copy()
    home["team"], home["opp"] = home["home_team"], home["away_team"]
    home["team_score"], home["opp_score"] = home["home_score"], home["away_score"]

    away = g[cols].copy()
    away["team"], away["opp"] = away["away_team"], away["home_team"]
    away["team_score"], away["opp_score"] = away["away_score"], away["home_score"]

    long = pd.concat([home, away], ignore_index=True)[
        ["game_id", "season", "week", "team", "opp", "team_score", "opp_score"]
    ]
    long["result"] = long.apply(
        lambda r: "W" if r.team_score > r.opp_score else ("L" if r.team_score < r.opp_score else "T"),
        axis=1,
    )
    return long


def print_summary_table(title, merged, value_cols, seasons):
    print(f"\n=== {title} ===")
    print(f"Seasons: {seasons}  |  team-games: {len(merged)}\n")
    print(merged.groupby("result")[value_cols].agg(["mean", "median", "count"]).round(2).to_string())


# ---------------------------------------------------------------------------
# Stat builders (one row per game_id + team)
# ---------------------------------------------------------------------------

def first_downs_by_team_game(pbp):
    return (
        pbp[pbp["first_down"] == 1]
        .groupby(["game_id", "posteam"])
        .size()
        .reset_index(name="first_downs")
    )


def plays_yards_by_team_game(pbp):
    # Yards per play = total yards on run/pass plays / number of run/pass plays
    # (excludes penalties-only plays, kneels, spikes, etc.)
    scrim = pbp[pbp["play_type"].isin(["run", "pass"])]
    out = scrim.groupby(["game_id", "posteam"]).agg(
        plays=("play_type", "count"),
        yards=("yards_gained", "sum"),
    ).reset_index()
    out["yards_per_play"] = out["yards"] / out["plays"]
    return out


def redzone_by_team_game(pbp):
    drives = pbp.dropna(subset=["posteam", "fixed_drive"]).groupby(
        ["game_id", "posteam", "fixed_drive"]
    ).agg(
        inside20=("drive_inside20", "max"),
        result=("fixed_drive_result", "first"),
    ).reset_index()

    rz = drives[drives["inside20"] == 1].copy()
    rz["td"] = (rz["result"] == "Touchdown").astype(int)
    rz["fg"] = (rz["result"] == "Field goal").astype(int)
    rz["scored"] = ((rz["td"] == 1) | (rz["fg"] == 1)).astype(int)

    out = rz.groupby(["game_id", "posteam"]).agg(
        rz_trips=("td", "count"),
        rz_td=("td", "sum"),
        rz_fg=("fg", "sum"),
        rz_scored=("scored", "sum"),
    ).reset_index()
    out["rz_td_pct"] = out["rz_td"] / out["rz_trips"]
    out["rz_score_pct"] = out["rz_scored"] / out["rz_trips"]
    return out


def main_rusher_by_team_game(pbp, min_attempts=10):
    rush = pbp[(pbp["rush_attempt"] == 1) & pbp["rusher_player_id"].notna()]
    per_rusher = rush.groupby(
        ["game_id", "posteam", "rusher_player_id", "rusher_player_name"]
    ).agg(
        attempts=("rush_attempt", "sum"),
        yards=("rushing_yards", "sum"),
    ).reset_index()
    per_rusher["ypc"] = per_rusher["yards"] / per_rusher["attempts"]

    idx = per_rusher.groupby(["game_id", "posteam"])["attempts"].idxmax()
    main_rusher = per_rusher.loc[idx].reset_index(drop=True)
    main_rusher["qualified"] = main_rusher["attempts"] >= min_attempts
    return main_rusher


def team_rushing_by_team_game(pbp):
    """Team-level rushing: every rush attempt credited to that team, not just the lead back."""
    rush = pbp[pbp["rush_attempt"] == 1]
    out = rush.groupby(["game_id", "posteam"]).agg(
        rush_attempts=("rush_attempt", "sum"),
        rush_yards=("rushing_yards", "sum"),
    ).reset_index()
    out["yds_per_attempt"] = (out["rush_yards"] / out["rush_attempts"]).round(1)
    return out


def qb_hits_sacks_by_team_game(pbp):
    # qb_hit and sack are reported as separate columns in nflverse charting --
    # hits usually include sacks, but not with 100% consistency, so we keep
    # them as two separate counts rather than assuming one contains the other.
    db = pbp[pbp["qb_dropback"] == 1]
    return db.groupby(["game_id", "posteam"]).agg(
        qb_hits=("qb_hit", "sum"),
        sacks=("sack", "sum"),
    ).reset_index()


def turnover_margin_by_team_game(pbp, long):
    pbp = pbp.copy()
    pbp["turnover"] = pbp["interception"].fillna(0) + pbp["fumble_lost"].fillna(0)
    give = pbp[pbp["posteam"].notna()].groupby(["game_id", "posteam"])["turnover"].sum().reset_index(
        name="giveaways"
    )

    merged = long.merge(give, left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left")
    merged = merged.drop(columns=["posteam"]).rename(columns={"giveaways": "own_giveaways"})
    merged = merged.merge(give, left_on=["game_id", "opp"], right_on=["game_id", "posteam"], how="left")
    merged = merged.drop(columns=["posteam"]).rename(columns={"giveaways": "opp_giveaways"})

    merged["own_giveaways"] = merged["own_giveaways"].fillna(0)
    merged["opp_giveaways"] = merged["opp_giveaways"].fillna(0)
    merged["takeaways"] = merged["opp_giveaways"]
    merged["turnover_margin"] = merged["takeaways"] - merged["own_giveaways"]
    return merged


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_first_downs_trend(seasons, save_csv=True):
    games = load_games()
    pbp = load_pbp_multi(seasons, tag="first-downs")
    fd = first_downs_by_team_game(pbp)

    long = build_long_results(games, seasons)
    merged = long.merge(fd, left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left")
    merged["first_downs"] = merged["first_downs"].fillna(0)
    merged = merged[merged["result"].isin(["W", "L"])]

    print_summary_table("First downs: winners vs losers", merged, ["first_downs"], seasons)
    summary = merged.groupby("result")["first_downs"].mean()

    if save_csv:
        merged.to_csv("team_game_first_downs.csv", index=False)
        print("\nSaved: team_game_first_downs.csv")

    print("\n--- Paste-ready summary ---")
    print(
        f"Winning teams averaged {summary['W']:.1f} first downs per game; "
        f"losing teams averaged {summary['L']:.1f}."
    )
    return merged


def cmd_efficiency_trend(seasons, save_csv=True):
    games = load_games()
    pbp = load_pbp_multi(seasons, tag="efficiency")
    eff = plays_yards_by_team_game(pbp)

    long = build_long_results(games, seasons)
    merged = long.merge(eff, left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left")
    merged = merged.drop(columns=["posteam"])
    merged = merged[merged["result"].isin(["W", "L"])]

    print_summary_table(
        "Plays / yards / yards-per-play: winners vs losers",
        merged, ["plays", "yards", "yards_per_play"], seasons,
    )
    summary = merged.groupby("result")[["plays", "yards", "yards_per_play"]].mean()

    if save_csv:
        merged.to_csv("team_game_efficiency.csv", index=False)
        print("\nSaved: team_game_efficiency.csv")

    print("\n--- Paste-ready summary ---")
    print(
        f"Winning teams averaged {summary.loc['W','plays']:.1f} plays and "
        f"{summary.loc['W','yards']:.1f} yards/game ({summary.loc['W','yards_per_play']:.2f} yds/play); "
        f"losing teams averaged {summary.loc['L','plays']:.1f} plays and "
        f"{summary.loc['L','yards']:.1f} yards/game ({summary.loc['L','yards_per_play']:.2f} yds/play)."
    )
    return merged


def cmd_redzone_trend(seasons, save_csv=True):
    games = load_games()
    pbp = load_pbp_multi(seasons, tag="red-zone")
    rz = redzone_by_team_game(pbp)

    long = build_long_results(games, seasons)
    merged = long.merge(rz, left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="inner")
    merged = merged[merged["result"].isin(["W", "L"])]
    merged = merged[merged["rz_trips"] >= 1]

    print_summary_table(
        "Red zone trips / TD% / score%: winners vs losers",
        merged, ["rz_trips", "rz_td_pct", "rz_score_pct"], seasons,
    )
    summary = merged.groupby("result")[["rz_trips", "rz_td_pct"]].mean()

    if save_csv:
        merged.to_csv("team_game_redzone.csv", index=False)
        print("\nSaved: team_game_redzone.csv")

    print("\n--- Paste-ready summary ---")
    print(
        f"Winning teams averaged {summary.loc['W','rz_trips']:.1f} red zone trips per game and "
        f"scored touchdowns on {summary.loc['W','rz_td_pct']*100:.1f}% of them; losing teams averaged "
        f"{summary.loc['L','rz_trips']:.1f} trips at a {summary.loc['L','rz_td_pct']*100:.1f}% TD rate."
    )
    return merged


def cmd_rushing_trend(seasons, min_attempts=10, ypc_threshold=4.0, save_csv=True):
    games = load_games()
    pbp = load_pbp_multi(seasons, tag="rushing")
    main_rusher = main_rusher_by_team_game(pbp, min_attempts=min_attempts)

    long = build_long_results(games, seasons)
    merged = main_rusher.merge(long, left_on=["game_id", "posteam"], right_on=["game_id", "team"], how="inner")
    merged = merged[merged["result"].isin(["W", "L"])]

    print(f"\n=== Main rusher YPC: winners vs losers (attempts >= {min_attempts}) ===")
    print(f"Seasons: {seasons}  |  team-games: {len(merged)}\n")
    qualified_only = merged[merged["attempts"] >= min_attempts]
    print(qualified_only.groupby("result")["ypc"].agg(["mean", "median", "count"]).round(2).to_string())

    hit = merged[(merged["attempts"] >= min_attempts) & (merged["ypc"] >= ypc_threshold)]
    miss = merged[(merged["attempts"] >= min_attempts) & (merged["ypc"] < ypc_threshold)]
    few = merged[merged["attempts"] < min_attempts]

    print(f"\nWin rate, main rusher {min_attempts}+ attempts & {ypc_threshold}+ ypc: "
          f"{(hit['result']=='W').mean()*100:.1f}% (n={len(hit)})")
    print(f"Win rate, main rusher {min_attempts}+ attempts & under {ypc_threshold} ypc: "
          f"{(miss['result']=='W').mean()*100:.1f}% (n={len(miss)})")
    print(f"Win rate, main rusher under {min_attempts} attempts: "
          f"{(few['result']=='W').mean()*100:.1f}% (n={len(few)})  "
          f"[caution: mostly a game-script effect, not a cause]")

    if save_csv:
        merged.to_csv("team_game_rushing.csv", index=False)
        print("\nSaved: team_game_rushing.csv")

    print("\n--- Paste-ready summary ---")
    print(
        f"When a team's lead rusher had {min_attempts}+ carries at {ypc_threshold}+ yards/carry, "
        f"that team won {(hit['result']=='W').mean()*100:.1f}% of the time, vs "
        f"{(miss['result']=='W').mean()*100:.1f}% when the same carry volume came at a lower ypc."
    )
    return merged


def cmd_turnover_trend(seasons, save_csv=True):
    games = load_games()
    pbp = load_pbp_multi(seasons, tag="turnovers")
    long = build_long_results(games, seasons)
    merged = turnover_margin_by_team_game(pbp, long)
    merged = merged[merged["result"].isin(["W", "L"])]

    print_summary_table("Turnover margin: winners vs losers", merged, ["turnover_margin"], seasons)

    def bucket(m):
        if m > 0:
            return "positive (+)"
        if m < 0:
            return "negative (-)"
        return "even (0)"

    merged["bucket"] = merged["turnover_margin"].apply(bucket)
    win_rate = merged.groupby("bucket")["result"].apply(lambda s: (s == "W").mean() * 100).round(1)
    counts = merged["bucket"].value_counts()
    print("\nWin rate by turnover margin bucket:")
    print(pd.DataFrame({"win_rate_pct": win_rate, "n": counts}).to_string())

    if save_csv:
        merged.to_csv("team_game_turnover_margin.csv", index=False)
        print("\nSaved: team_game_turnover_margin.csv")

    pos_rate = win_rate.get("positive (+)", float("nan"))
    neg_rate = win_rate.get("negative (-)", float("nan"))
    print("\n--- Paste-ready summary ---")
    print(
        f"Teams with a positive turnover margin won {pos_rate:.1f}% of their games; "
        f"teams with a negative turnover margin won only {neg_rate:.1f}%."
    )
    return merged


def cmd_all_trends(seasons, min_attempts=10, ypc_threshold=4.0):
    print("Running all five trend analyses. This loads play-by-play data multiple times")
    print("(once per analysis) and may take a little while for many seasons.\n")
    cmd_efficiency_trend(seasons, save_csv=False)
    cmd_first_downs_trend(seasons, save_csv=False)
    cmd_redzone_trend(seasons, save_csv=False)
    cmd_rushing_trend(seasons, min_attempts=min_attempts, ypc_threshold=ypc_threshold, save_csv=False)
    cmd_turnover_trend(seasons, save_csv=False)


def cmd_boxscore(game_id):
    season = int(game_id.split("_")[0])
    games = load_games()
    pbp = load_pbp(season)

    game_row = games[games["game_id"] == game_id]
    if game_row.empty:
        print(f"game_id '{game_id}' not found in {season} schedule.")
        return
    row = game_row.iloc[0]

    home, away = row["home_team"], row["away_team"]
    home_score, away_score = row["home_score"], row["away_score"]
    winner = home if home_score > away_score else away if away_score > home_score else "TIE"

    g_pbp = pbp[pbp["game_id"] == game_id]
    fd = g_pbp[g_pbp["first_down"] == 1].groupby("posteam").size()

    # Red zone for this single game
    drives = g_pbp.dropna(subset=["posteam", "fixed_drive"]).groupby(
        ["posteam", "fixed_drive"]
    ).agg(inside20=("drive_inside20", "max"), result=("fixed_drive_result", "first")).reset_index()
    rz = drives[drives["inside20"] == 1]
    team_rushing = team_rushing_by_team_game(g_pbp).set_index("posteam")

    stats = {}
    for team in [home, away]:
        t = g_pbp[g_pbp["posteam"] == team]
        team_rz = rz[rz["posteam"] == team]
        rz_trips = len(team_rz)
        rz_td = int((team_rz["result"] == "Touchdown").sum())
        rushing = team_rushing.loc[team] if team in team_rushing.index else None
        stats[team] = {
            "first_downs": int(fd.get(team, 0)),
            "total_yards": int(t["yards_gained"].sum()),
            "pass_yards": int(t.loc[t["play_type"] == "pass", "yards_gained"].sum()),
            "rush_attempts": int(rushing["rush_attempts"]) if rushing is not None else 0,
            "rush_yards": int(rushing["rush_yards"]) if rushing is not None else 0,
            "yds_per_attempt": rushing["yds_per_attempt"] if rushing is not None else 0.0,
            "turnovers": int(t["interception"].sum() + t["fumble_lost"].sum()),
            "third_down_pct": round(
                100 * t["third_down_converted"].sum()
                / max(1, (t["third_down_converted"].sum() + t["third_down_failed"].sum())),
                1,
            ),
            "rz_trips": rz_trips,
            "rz_td_pct": round(100 * rz_td / rz_trips, 1) if rz_trips else 0.0,
        }

    print(f"\n{away} @ {home}  —  {row['gameday']}  (game_id: {game_id})")
    print(f"Final: {away} {away_score} — {home} {home_score}   Winner: {winner}\n")
    stat_df = pd.DataFrame(stats).T
    print(stat_df.to_string())

    print("\n--- Paste-ready summary ---")
    print(
        f"{away} @ {home} on {row['gameday']}: final score {away} {away_score}, {home} {home_score} "
        f"({winner} won). First downs: {away} {stats[away]['first_downs']}, {home} {stats[home]['first_downs']}. "
        f"Total yards: {away} {stats[away]['total_yards']}, {home} {stats[home]['total_yards']}. "
        f"Turnovers: {away} {stats[away]['turnovers']}, {home} {stats[home]['turnovers']}. "
        f"Red zone TD%: {away} {stats[away]['rz_td_pct']}%, {home} {stats[home]['rz_td_pct']}%."
    )


def cmd_list_games(season, team):
    games = load_games()
    g = games[games["season"] == season]
    if team:
        g = g[(g["home_team"] == team) | (g["away_team"] == team)]
    cols = ["game_id", "week", "gameday", "away_team", "home_team", "away_score", "home_score"]
    print(g[cols].sort_values("week").to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFL box score + stat comparison toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_box = sub.add_parser("boxscore", help="Show one game's box score and winner")
    p_box.add_argument("--game_id", required=True)

    p_list = sub.add_parser("list-games", help="List game_ids for a season/team")
    p_list.add_argument("--season", type=int, required=True)
    p_list.add_argument("--team", default=None)

    p_eff = sub.add_parser("efficiency-trend", help="Plays/yards/yards-per-play: winners vs losers")
    p_eff.add_argument("--seasons", nargs="+", type=int, default=[2025])

    p_fd = sub.add_parser("first-downs-trend", help="First downs: winners vs losers")
    p_fd.add_argument("--seasons", nargs="+", type=int, default=[2025])

    p_rz = sub.add_parser("redzone-trend", help="Red zone trips/TD%%: winners vs losers")
    p_rz.add_argument("--seasons", nargs="+", type=int, default=[2025])

    p_rush = sub.add_parser("rushing-trend", help="Main rusher YPC/attempts: winners vs losers")
    p_rush.add_argument("--seasons", nargs="+", type=int, default=[2025])
    p_rush.add_argument("--min_attempts", type=int, default=10)
    p_rush.add_argument("--ypc_threshold", type=float, default=4.0)

    p_to = sub.add_parser("turnover-trend", help="Turnover margin (INT+fumbles lost): winners vs losers")
    p_to.add_argument("--seasons", nargs="+", type=int, default=[2025])

    p_all = sub.add_parser("all-trends", help="Run all four trend analyses in sequence")
    p_all.add_argument("--seasons", nargs="+", type=int, default=[2025])
    p_all.add_argument("--min_attempts", type=int, default=10)
    p_all.add_argument("--ypc_threshold", type=float, default=4.0)

    args = parser.parse_args()

    if args.cmd == "boxscore":
        cmd_boxscore(args.game_id)
    elif args.cmd == "list-games":
        cmd_list_games(args.season, args.team)
    elif args.cmd == "efficiency-trend":
        cmd_efficiency_trend(args.seasons)
    elif args.cmd == "first-downs-trend":
        cmd_first_downs_trend(args.seasons)
    elif args.cmd == "redzone-trend":
        cmd_redzone_trend(args.seasons)
    elif args.cmd == "rushing-trend":
        cmd_rushing_trend(args.seasons, args.min_attempts, args.ypc_threshold)
    elif args.cmd == "turnover-trend":
        cmd_turnover_trend(args.seasons)
    elif args.cmd == "all-trends":
        cmd_all_trends(args.seasons, args.min_attempts, args.ypc_threshold)
