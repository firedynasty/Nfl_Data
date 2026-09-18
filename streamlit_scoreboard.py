#!/usr/bin/env python3
"""
Streamlit scoreboard built on nfl_box_score_analysis.py.

One page: pick a season/week and scroll through every game that week, each
with its final score and the box-score-style team stats (plays/yards/yards-
per-play, turnover margin, red zone trips/TD%/score%) listed right underneath
-- same stat logic as the CLI's efficiency-trend / turnover-trend /
redzone-trend commands, just computed per game instead of aggregated across
a season.

Usage:
    streamlit run streamlit_scoreboard.py
"""

import pandas as pd
import streamlit as st

from nfl_box_score_analysis import (
    build_long_results,
    first_downs_by_team_game,
    load_games,
    load_pbp,
    plays_yards_by_team_game,
    qb_hits_sacks_by_team_game,
    redzone_by_team_game,
    team_rushing_by_team_game,
    turnover_margin_by_team_game,
)
from predict_week import build_week_table, detail_for_games, line_string, HFA_POINTS, MARGIN_STD
from srs import blended_asof_ratings

LOGOS_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/logos.csv"
TEAMS_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/teams.csv"

st.set_page_config(page_title="NFL Scoreboard", page_icon="🏈", layout="wide")
st.markdown(
    '<style>section[data-testid="stSidebar"] { width: 220px !important; }</style>',
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_games():
    return load_games()


@st.cache_data(show_spinner=False)
def cached_logos() -> dict[str, str]:
    logos = pd.read_csv(LOGOS_URL)
    return dict(zip(logos["team"], logos["team_logo"]))


@st.cache_data(show_spinner=False)
def cached_team_names() -> dict[str, str]:
    teams = pd.read_csv(TEAMS_URL)
    latest = teams.sort_values("season").drop_duplicates("team", keep="last")
    return dict(zip(latest["team"], latest["nickname"]))


@st.cache_data(show_spinner="Loading play-by-play for this season...")
def season_game_stats(season: int) -> pd.DataFrame:
    """One row per team-game: efficiency, turnover margin, and red zone stats."""
    games = cached_games()
    pbp = load_pbp(season)
    long = build_long_results(games, [season])

    eff = plays_yards_by_team_game(pbp)
    rz = redzone_by_team_game(pbp)
    to = turnover_margin_by_team_game(pbp, long)
    rush = team_rushing_by_team_game(pbp)
    fd = first_downs_by_team_game(pbp)
    pressure = qb_hits_sacks_by_team_game(pbp)

    df = to[["game_id", "week", "team", "opp", "team_score", "opp_score", "result",
             "own_giveaways", "opp_giveaways", "turnover_margin"]]
    df = df.merge(eff, left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left").drop(columns=["posteam"])
    df = df.merge(
        rz[["game_id", "posteam", "rz_trips", "rz_td", "rz_scored"]],
        left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left",
    ).drop(columns=["posteam"])
    df = df.merge(
        rush[["game_id", "posteam", "rush_attempts", "rush_yards", "yds_per_attempt"]],
        left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left",
    ).drop(columns=["posteam"])
    df = df.merge(fd, left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left").drop(columns=["posteam"])
    df["first_downs"] = df["first_downs"].fillna(0)
    df = df.merge(
        pressure, left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left",
    ).drop(columns=["posteam"])
    return df


@st.cache_data(ttl=3600, show_spinner="Computing SRS ratings...")
def cached_srs_ratings(season: int):
    """No-leakage blended SRS per week for one season. Scores-only (no pbp),
    so this is fast; ttl lets newly played games flow into the ratings."""
    return blended_asof_ratings(cached_games(), [season])


def current_season_and_week(games: pd.DataFrame) -> tuple[int, int]:
    """Latest season, and the most recent week that has kicked off (so the
    default view is 'this week', including in-progress/upcoming games in it)."""
    season = int(games["season"].max())
    sub = games[games["season"] == season].copy()
    sub["gameday"] = pd.to_datetime(sub["gameday"])
    started = sub[sub["gameday"] <= pd.Timestamp.now()]
    week = int(started["week"].max()) if len(started) else 1
    return season, week


def day_label(gameday: pd.Timestamp) -> str:
    today = pd.Timestamp.now().normalize()
    delta = (gameday.normalize() - today).days
    if delta == 0:
        return "Today"
    if delta == -1:
        return "Yesterday"
    if delta == 1:
        return "Tomorrow"
    return gameday.strftime("%a, %b %-d")


def stats_table(game_stats: pd.DataFrame, game_id: str, away: str, home: str, away_label: str, home_label: str) -> pd.DataFrame:
    g = game_stats[game_stats["game_id"] == game_id].set_index("team")
    rows = {}
    for team, label in ((away, away_label), (home, home_label)):
        if team not in g.index:
            rows[label] = {"1st": "-", "Plays": "-", "Yards": "-", "Yds/Play": "-", "TO's": "-",
                            "Rush #": "-", "Yds": "-", "Yds/Att": "-",
                            "RZ TD": "-", "RZ Score": "-", "QB Hits": "-", "Sacks": "-"}
            continue
        r = g.loc[team]
        rz_trips = int(r["rz_trips"]) if pd.notna(r["rz_trips"]) else 0
        rz_td = int(r["rz_td"]) if pd.notna(r["rz_td"]) else 0
        rz_scored = int(r["rz_scored"]) if pd.notna(r["rz_scored"]) else 0
        rows[label] = {
            "1st": int(r["first_downs"]) if pd.notna(r["first_downs"]) else "-",
            "Plays": int(r["plays"]) if pd.notna(r["plays"]) else "-",
            "Yards": int(r["yards"]) if pd.notna(r["yards"]) else "-",
            "Yds/Play": f"{r['yards_per_play']:.1f}" if pd.notna(r["yards_per_play"]) else "-",
            "TO's": f"{r['turnover_margin']:+.0f}",
            "Rush #": int(r["rush_attempts"]) if pd.notna(r["rush_attempts"]) else "-",
            "Yds": int(r["rush_yards"]) if pd.notna(r["rush_yards"]) else "-",
            "Yds/Att": f"{r['yds_per_attempt']:.1f}" if pd.notna(r["yds_per_attempt"]) else "-",
            "RZ TD": f"{rz_td}/{rz_trips}" if rz_trips else "-",
            "RZ Score": f"{rz_scored}/{rz_trips}" if rz_trips else "-",
            "QB Hits": int(r["qb_hits"]) if pd.notna(r["qb_hits"]) else "-",
            "Sacks": int(r["sacks"]) if pd.notna(r["sacks"]) else "-",
        }
    return pd.DataFrame(rows).T


def main():
    st.title("🏈 NFL Scoreboard")
    st.caption("Score, then the box-score stats right underneath: plays/yards/yards-per-play, turnover margin, red zone conversion.")

    games = cached_games()
    logos = cached_logos()
    names = cached_team_names()
    default_season, default_week = current_season_and_week(games)

    with st.sidebar:
        st.header("Setup")
        seasons_available = sorted(games["season"].unique(), reverse=True)
        season = st.selectbox("Season", seasons_available, index=seasons_available.index(default_season))

        view_by = st.segmented_control("View by", ["Week", "Team", "Predictions"], default="Week")

        season_games = games[games["season"] == season]
        # Predictions needs only final scores (SRS) -- skip the heavy pbp load
        game_stats = None if view_by == "Predictions" else season_game_stats(season)
        week = None
        team_filter = None
        if view_by == "Team":
            teams_available = sorted(
                set(season_games["home_team"]).union(season_games["away_team"]),
                key=lambda t: names.get(t, t),
            )
            default_team = "LAC" if "LAC" in teams_available else teams_available[0]
            team_filter = st.selectbox(
                "Team", teams_available,
                index=teams_available.index(default_team),
                format_func=lambda t: names.get(t, t),
            )
        else:
            weeks_available = sorted(season_games["week"].unique())
            week = st.selectbox(
                "Week", weeks_available,
                index=weeks_available.index(default_week) if default_week in weeks_available else 0,
            )

        show_preds = False
        if view_by != "Predictions":
            show_preds = st.toggle(
                "Attach prediction row to games", value=False,
                help="Adds each game's SRS ratings, model line, market line, and "
                     "predicted winner right under the score.",
            )

        st.divider()
        st.caption(
            "Note on QB Hits/Sacks: hits usually include sacks, but not with "
            "100% consistency in the raw nflverse charting -- the two are kept "
            "as separate reported columns rather than assuming one strictly "
            "contains the other."
        )

    if view_by == "Predictions":
        ratings = cached_srs_ratings(season)
        pred_df, _ = build_week_table(games, season, week, ratings, k=4.0, shrink=0.7,
                                      names=names)

        st.caption(f"Season {season}, Week {week} — {len(pred_df)} games · "
                   f"SRS ratings as of before week {week} (no games from this week or later)")
        st.sidebar.download_button(
            "⬇️ Download this view (CSV)",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name=f"nfl_{season}_week{week:02d}_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(pred_df, hide_index=True, use_container_width=True)
        st.caption(
            f"away_srs/home_srs: opponent-adjusted team rating in points "
            f"(0 = league average; HFA {HFA_POINTS:+.1f}, win-prob std {MARGIN_STD:.1f}). "
            f"model_line: SRS diff + home field. edge: model line minus market line, "
            f"home-positive (+ = model likes home more than the market)."
        )
        st.caption(
            "Reminder: in honest walk-forward backtests (2021-2025, spread and "
            "moneyline) no edge bucket cleared the 52.4% breakeven — this sheet "
            "says how good the teams are, not what to bet."
        )
        with st.expander("How are these predictions made?"):
            st.markdown(
                "**away_srs / home_srs** — each team's Simple Rating System number, "
                "in points: average scoring margin plus the average rating of the "
                "opponents faced (opponent-adjusted; 0 = league average). Uses only "
                "games *before* this week, blended with last season early in the year.\n\n"
                "**model_line** — home_srs − away_srs + 2.2 (home-field constant "
                "estimated from the 2021-2025 backtest): the margin the model "
                "considers fair, in betting notation.\n\n"
                "**market_line** — the sportsbook's actual spread.\n\n"
                "**edge** — model_line − market_line (home-positive): + means the "
                "model likes the home team *more than the market does*.\n\n"
                "**pred / win_pct** — the team the model thinks **wins the game "
                "outright** (bigger predicted margin), and its calibrated "
                "probability (normal curve, 13.4-point error std from the backtest). "
                "This is about *winning*, never about covering the spread.\n\n"
                "**final / pred_right** — the actual score, and whether the "
                "predicted winner actually won."
            )
        return

    if view_by == "Team":
        shown_games = season_games[
            (season_games["home_team"] == team_filter) | (season_games["away_team"] == team_filter)
        ].copy()
        st.caption(f"Season {season}, {names.get(team_filter, team_filter)} — {len(shown_games)} games")
    else:
        shown_games = season_games[season_games["week"] == week].copy()
        st.caption(f"Season {season}, Week {week} — {len(shown_games)} games")

    shown_games["gameday"] = pd.to_datetime(shown_games["gameday"])
    shown_games = shown_games.sort_values(["week", "gameday", "gametime"])

    pred_by_id = None
    if show_preds:
        pred_by_id = detail_for_games(shown_games, cached_srs_ratings(season)).set_index("game_id")

    export_df = game_stats[game_stats["game_id"].isin(shown_games["game_id"])].copy()
    export_df.insert(0, "season", season)
    export_df.insert(3, "team_name", export_df["team"].map(names))
    export_df.insert(5, "opp_name", export_df["opp"].map(names))
    export_scope = names.get(team_filter, team_filter) if view_by == "Team" else f"week{week}"
    st.sidebar.download_button(
        "⬇️ Download this view (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"nfl_{season}_{export_scope}_team_game_stats.csv",
        mime="text/csv",
        help="Every stat shown on the scoreboard cards, for exactly the games "
             "currently on screen -- handy for a groupby/mean in pandas or "
             "handing to Claude Code.",
        use_container_width=True,
    )

    for game in shown_games.itertuples():
        final = pd.notna(game.away_score) and pd.notna(game.home_score)
        away_win = final and game.away_score > game.home_score
        home_win = final and game.home_score > game.away_score
        away_label = names.get(game.away_team, game.away_team)
        home_label = names.get(game.home_team, game.home_team)

        with st.container(border=True):
            for team, label, score, won in (
                (game.away_team, away_label, game.away_score, away_win),
                (game.home_team, home_label, game.home_score, home_win),
            ):
                logo_col, name_col, score_col = st.columns([1, 6, 1])
                logo_url = logos.get(team)
                if logo_url:
                    logo_col.image(logo_url, width=32)
                name_col.markdown(f"{'**' if won else ''}{label}{'**' if won else ''}")
                score_text = f"{int(score)}" if pd.notna(score) else "-"
                score_col.markdown(f"{'**' if won else ''}{score_text}{'**' if won else ''}")

            status = "Final" if final else "Upcoming"
            time_suffix = f" {game.gametime}" if not final and pd.notna(game.gametime) else ""
            week_prefix = f"Week {game.week} · " if view_by == "Team" else ""
            st.caption(f"{week_prefix}{status} · {day_label(game.gameday)}{time_suffix}")

            if pred_by_id is not None and game.game_id in pred_by_id.index:
                p = pred_by_id.loc[game.game_id]
                model_line = line_string(game.home_team, game.away_team, p["pred_margin"])
                market_line = (line_string(game.home_team, game.away_team, p["spread_line"])
                               if pd.notna(p["spread_line"]) else "-")
                win_pct = 100 * max(p["home_win_prob"], 1 - p["home_win_prob"])
                right = ""
                if final:
                    correct = (game.home_score > game.away_score) == (p["pred_winner"] == game.home_team)
                    right = f" · {'✓ right' if correct else '✗ wrong'}"
                st.caption(
                    f"🔮 SRS {game.away_team} {p['away_srs']:+.1f} @ {game.home_team} "
                    f"{p['home_srs']:+.1f} · model {model_line} · market {market_line} · "
                    f"pred {p['pred_winner']} {win_pct:.0f}%{right}"
                )

            if final:
                st.table(
                    stats_table(game_stats, game.game_id, game.away_team, game.home_team, away_label, home_label)
                )


if __name__ == "__main__":
    main()
